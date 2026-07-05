// Phase 7A: AOT native backends — C and LLVM textual IR.
//
// A program is fully-AOT-able when every function is in the JIT's i64/f64
// eligible sets and `main` is int-statements + `print(<int expr>)` /
// `print("literal")`. The emitted code goes through `cc -O2` (C) or
// `clang -O2` (.ll) into a pure native binary: no runtime, no warm-up.
// AOT has no deopt path, so `nova build --aot` verifies the binary's output
// against the tiered VM at build time and falls back to the embed build on
// ANY divergence (overflow->BigInt, shift edge cases, ...) — the shipped
// binary is byte-identical or it doesn't ship as AOT at all.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use crate::ast::*;
use crate::jit::{eligible_set, float_eligible_set};

pub enum Backend { C, Llvm, Wasm, Arm }

#[derive(Clone, Copy, PartialEq)]
pub enum Tier { Typed, Boxed }

impl Tier {
    pub fn name(&self) -> &'static str {
        match self { Tier::Typed => "typed", Tier::Boxed => "boxed" }
    }
}

pub struct AotProgram<'p> {
    int_fns: Vec<&'p Func>,
    float_fns: Vec<&'p Func>,
    // main's body with the parser's implicit trailing `return <expr>`
    // rewritten back to plain expression statements
    main_body: Vec<Stmt>,
}

// dry check: which functions exist, whether main qualifies
pub fn analyze(prog: &Program) -> Option<AotProgram<'_>> {
    let ints = eligible_set(prog);
    let floats = float_eligible_set(prog, &ints);
    let mut int_fns = Vec::new();
    let mut float_fns = Vec::new();
    let mut main = None;
    for item in &prog.items {
        let f = match item {
            Item::Func(f) => f,
            // consts/tests are ignored by the native binary; anything else
            // (structs, enums, machines, impls) means heap features -> no AOT
            Item::Const { .. } | Item::Test(_) => continue,
            _ => return None,
        };
        if f.name == "main" { main = Some(f); continue; }
        if ints.contains(&f.name) { int_fns.push(f); }
        else if floats.contains(&f.name) { float_fns.push(f); }
        else { return None; }
    }
    let main = main?;
    if !main.params.is_empty() { return None; }
    let mut main_body = main.body.clone();
    fix_main_tail(&mut main_body);
    let mut scopes = vec![HashSet::new()];
    if !main_body.iter().all(|s| main_stmt_ok(s, &ints, &mut scopes)) { return None; }
    Some(AotProgram { int_fns, float_fns, main_body })
}

// the parser turns main's trailing expression (and the tails of a trailing
// if/else) into `return <expr>`; in a binary's main that is just "evaluate it"
fn fix_main_tail(body: &mut Vec<Stmt>) {
    match body.last_mut() {
        Some(Stmt::Return(Some(_))) => {
            if let Some(Stmt::Return(Some(e))) = body.pop() {
                body.push(Stmt::Expr(e));
            }
        }
        Some(Stmt::If { then, els, .. }) => {
            fix_main_tail(then);
            if let Some(e) = els { fix_main_tail(e); }
        }
        _ => {}
    }
}

// main may use the int-pure statement set (plus print) — with two extra
// restrictions the C/LLVM lowering needs: no `**`, and no shadowing `let`
fn main_stmt_ok(s: &Stmt, ints: &HashSet<String>, scopes: &mut Vec<HashSet<String>>) -> bool {
    stmt_ok(s, ints, scopes, false)
}

fn stmt_ok(s: &Stmt, ints: &HashSet<String>, scopes: &mut Vec<HashSet<String>>, allow_ret: bool) -> bool {
    match s {
        Stmt::Return(Some(e)) => allow_ret && int_expr_ok(e, ints, scopes),
        Stmt::Expr(e) => {
            let mut e = e;
            while let Expr::At { expr, .. } = e { e = expr; }
            if let Expr::Call { callee, args } = e {
                if callee == "print" && args.len() == 1 {
                    let mut a = &args[0];
                    while let Expr::At { expr, .. } = a { a = expr; }
                    return matches!(a, Expr::Str(_)) || int_expr_ok(a, ints, scopes);
                }
            }
            int_expr_ok(e, ints, scopes)
        }
        Stmt::Let { name, value, .. } => {
            if !int_expr_ok(value, ints, scopes) { return false; }
            if scopes.iter().any(|sc| sc.contains(name)) { return false; } // shadowing
            scopes.last_mut().unwrap().insert(name.clone());
            true
        }
        Stmt::Assign { name, value } => {
            if !int_expr_ok(value, ints, scopes) { return false; }
            if !scopes.iter().any(|sc| sc.contains(name)) {
                scopes.last_mut().unwrap().insert(name.clone());
            }
            true
        }
        Stmt::If { cond, then, els } => {
            if !int_expr_ok(cond, ints, scopes) { return false; }
            scopes.push(HashSet::new());
            let a = then.iter().all(|s| stmt_ok(s, ints, scopes, allow_ret));
            scopes.pop();
            scopes.push(HashSet::new());
            let b = els.as_ref().map_or(true, |e| e.iter().all(|s| stmt_ok(s, ints, scopes, allow_ret)));
            scopes.pop();
            a && b
        }
        Stmt::While { cond, body } => {
            if !int_expr_ok(cond, ints, scopes) { return false; }
            scopes.push(HashSet::new());
            let r = body.iter().all(|s| stmt_ok(s, ints, scopes, allow_ret));
            scopes.pop();
            r
        }
        Stmt::ForRange { var, start, end, body, .. } => {
            if !int_expr_ok(start, ints, scopes) || !int_expr_ok(end, ints, scopes) { return false; }
            if scopes.iter().any(|sc| sc.contains(var)) { return false; }
            scopes.push(HashSet::new());
            scopes.last_mut().unwrap().insert(var.clone());
            let r = body.iter().all(|s| stmt_ok(s, ints, scopes, allow_ret));
            scopes.pop();
            r
        }
        Stmt::Break(None) | Stmt::Continue => true,
        Stmt::Return(None) => true,
        _ => false,
    }
}

fn int_expr_ok(e: &Expr, ints: &HashSet<String>, scopes: &Vec<HashSet<String>>) -> bool {
    match e {
        Expr::At { expr, .. } => int_expr_ok(expr, ints, scopes),
        Expr::Int(_) => true,
        Expr::Ident(n) => scopes.iter().any(|sc| sc.contains(n)),
        Expr::Unary { op, expr } =>
            matches!(op, UnOp::Neg | UnOp::Not | UnOp::BitNot) && int_expr_ok(expr, ints, scopes),
        Expr::Binary { op, lhs, rhs } =>
            !matches!(op, BinOp::Pow)
            && int_expr_ok(lhs, ints, scopes) && int_expr_ok(rhs, ints, scopes),
        Expr::If { cond, then, els } =>
            int_expr_ok(cond, ints, scopes) && int_expr_ok(then, ints, scopes)
            && int_expr_ok(els, ints, scopes),
        Expr::Call { callee, args } =>
            ints.contains(callee) && args.iter().all(|a| int_expr_ok(a, ints, scopes)),
        _ => false,
    }
}

// same restrictions for function bodies (the JIT allows Pow/shadowing because
// it can deopt; AOT cannot). Checked per-function before emission.
fn fn_body_ok(f: &Func, ints: &HashSet<String>) -> bool {
    let mut scopes = vec![HashSet::new()];
    for p in &f.params { scopes.last_mut().unwrap().insert(p.clone()); }
    f.body.iter().all(|s| fn_stmt_ok(s, ints, &mut scopes))
}

fn fn_stmt_ok(s: &Stmt, ints: &HashSet<String>, scopes: &mut Vec<HashSet<String>>) -> bool {
    stmt_ok(s, ints, scopes, true)
}

pub fn emit(prog: &Program, backend: &Backend) -> Option<(String, Tier)> {
    if let Some(code) = emit_typed(prog, backend) {
        return Some((code, Tier::Typed));
    }
    // boxed tier (strings/arrays via the refcounted runtime). Both backends
    // lower against the same `runtime/nova_rt.c`: the C backend #includes it;
    // the LLVM backend emits textual IR that calls it through the struct ABI
    // clang uses for `NV` (a value passed as (i8 tag, i64 payload), returned as
    // `{i8,i64}`). Either way the byte-diff gate verifies the result.
    match backend {
        // ARM cross-compiles the same portable C (typed + boxed): nova_rt.c is
        // ordinary libc C, so the aarch64 cross gcc links it fine.
        Backend::C | Backend::Arm => {
            if let Some(code) = emit_boxed(prog) { return Some((code, Tier::Boxed)); }
        }
        Backend::Llvm => {
            if let Some(code) = emit_boxed_llvm(prog) { return Some((code, Tier::Boxed)); }
        }
        // WASM has no boxed tier: the refcounted runtime needs libc/malloc, which
        // means a wasi-sysroot. The typed (pure int/float + string-literal) subset
        // compiles freestanding and is the honest, verifiable WASM surface today.
        Backend::Wasm => {}
    }
    None
}

fn emit_typed(prog: &Program, backend: &Backend) -> Option<String> {
    let a = analyze(prog)?;
    let ints = eligible_set(prog);
    for f in &a.int_fns {
        if !fn_body_ok(f, &ints) { return None; }
    }
    // float bodies: the float track already forbids %, **, ForRange, shadowing
    // is still possible -> check with an empty int set (idents resolve via scopes)
    for f in &a.float_fns {
        let mut scopes = vec![HashSet::new()];
        for p in &f.params { scopes.last_mut().unwrap().insert(p.clone()); }
        if !float_body_no_shadow(&f.body, &mut scopes) { return None; }
    }
    Some(match backend {
        // ARM reuses the portable C codegen; only the compiler + run harness differ.
        Backend::C | Backend::Arm => CEmit::new(&a).emit(),
        Backend::Wasm => CEmit::new_wasm(&a).emit(),
        Backend::Llvm => LlEmit::new(&a).emit(),
    })
}

fn float_body_no_shadow(body: &[Stmt], scopes: &mut Vec<HashSet<String>>) -> bool {
    for s in body {
        match s {
            Stmt::Let { name, .. } => {
                if scopes.iter().any(|sc| sc.contains(name)) { return false; }
                scopes.last_mut().unwrap().insert(name.clone());
            }
            Stmt::Assign { name, .. } => {
                if !scopes.iter().any(|sc| sc.contains(name)) {
                    scopes.last_mut().unwrap().insert(name.clone());
                }
            }
            Stmt::If { then, els, .. } => {
                scopes.push(HashSet::new());
                if !float_body_no_shadow(then, scopes) { return false; }
                scopes.pop();
                scopes.push(HashSet::new());
                if let Some(e) = els { if !float_body_no_shadow(e, scopes) { return false; } }
                scopes.pop();
            }
            Stmt::While { body, .. } => {
                scopes.push(HashSet::new());
                if !float_body_no_shadow(body, scopes) { return false; }
                scopes.pop();
            }
            _ => {}
        }
    }
    true
}

// ---------------------------------------------------------------------------
// C backend
// ---------------------------------------------------------------------------

struct CEmit<'p> {
    a: &'p AotProgram<'p>,
    out: String,
    // true when targeting freestanding wasm32: no <stdio.h>, `print` routes to
    // JS-imported functions, and `main` is exported as `main` (no libc entry).
    wasm: bool,
}

fn mangle(n: &str) -> String { format!("nv_{}", n) }

impl<'p> CEmit<'p> {
    fn new(a: &'p AotProgram<'p>) -> Self { CEmit { a, out: String::new(), wasm: false } }
    fn new_wasm(a: &'p AotProgram<'p>) -> Self { CEmit { a, out: String::new(), wasm: true } }

    fn emit(mut self) -> String {
        if self.wasm {
            // Freestanding wasm: declare the two host imports (integer + string
            // printing) and pull in stdint for i64. The JS host formats values to
            // match `nova run` exactly; the byte-diff gate rejects any mismatch.
            self.out.push_str(
                "#include <stdint.h>\ntypedef int64_t i64;\n\
                 __attribute__((import_module(\"env\"),import_name(\"print_i64\")))\n\
                 void nova_print_i64(long long);\n\
                 __attribute__((import_module(\"env\"),import_name(\"print_str\")))\n\
                 void nova_print_str(const char*, int);\n\n");
        } else {
            self.out.push_str("#include <stdio.h>\n#include <stdint.h>\ntypedef int64_t i64;\n\n");
        }
        for f in &self.a.int_fns { self.proto(f, "i64"); }
        for f in &self.a.float_fns { self.proto(f, "double"); }
        self.out.push('\n');
        for f in self.a.int_fns.clone() { self.func(f, false); }
        for f in self.a.float_fns.clone() { self.func(f, true); }
        self.main();
        self.out
    }

    fn proto(&mut self, f: &Func, ty: &str) {
        let ps: Vec<String> = f.params.iter().map(|p| format!("{} {}", ty, mangle(p))).collect();
        let _ = writeln!(self.out, "static {} {}({});", ty, mangle(&f.name), ps.join(", "));
    }

    fn func(&mut self, f: &Func, is_float: bool) {
        let ty = if is_float { "double" } else { "i64" };
        let ps: Vec<String> = f.params.iter().map(|p| format!("{} {}", ty, mangle(p))).collect();
        let _ = writeln!(self.out, "static {} {}({}) {{", ty, mangle(&f.name), ps.join(", "));
        let mut vars: Vec<String> = Vec::new();
        collect_vars(&f.body, &mut vars);
        for v in &vars {
            if !f.params.contains(v) {
                let _ = writeln!(self.out, "  {} {} = 0;", ty, mangle(v));
            }
        }
        for s in &f.body { self.stmt(s, 1, is_float); }
        let _ = writeln!(self.out, "  return 0;\n}}\n");
    }

    fn main(&mut self) {
        if self.wasm {
            self.out.push_str("__attribute__((export_name(\"main\"))) void nova_main(void) {\n");
        } else {
            self.out.push_str("int main(void) {\n");
        }
        let mut vars: Vec<String> = Vec::new();
        collect_vars(&self.a.main_body, &mut vars);
        for v in &vars { let _ = writeln!(self.out, "  i64 {} = 0;", mangle(v)); }
        for s in &self.a.main_body.clone() { self.stmt(s, 1, false); }
        if self.wasm { self.out.push_str("}\n"); } else { self.out.push_str("  return 0;\n}\n"); }
    }

    fn stmt(&mut self, s: &Stmt, d: usize, fl: bool) {
        let ind = "  ".repeat(d);
        match s {
            Stmt::Expr(e) => {
                let mut inner = e;
                while let Expr::At { expr, .. } = inner { inner = expr; }
                if let Expr::Call { callee, args } = inner {
                    if callee == "print" && args.len() == 1 {
                        let mut a = &args[0];
                        while let Expr::At { expr, .. } = a { a = expr; }
                        if let Expr::Str(s) = a {
                            if self.wasm {
                                let _ = writeln!(self.out, "{}nova_print_str(\"{}\", {});", ind, c_escape(s), s.len());
                            } else {
                                let _ = writeln!(self.out, "{}printf(\"%s\\n\", \"{}\");", ind, c_escape(s));
                            }
                        } else {
                            let v = self.expr(a, fl);
                            if self.wasm {
                                let _ = writeln!(self.out, "{}nova_print_i64((long long)({}));", ind, v);
                            } else {
                                let _ = writeln!(self.out, "{}printf(\"%lld\\n\", (long long)({}));", ind, v);
                            }
                        }
                        return;
                    }
                }
                let v = self.expr(inner, fl);
                let _ = writeln!(self.out, "{}(void)({});", ind, v);
            }
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let v = self.expr(value, fl);
                let _ = writeln!(self.out, "{}{} = {};", ind, mangle(name), v);
            }
            Stmt::Return(Some(e)) => {
                let v = self.expr(e, fl);
                let _ = writeln!(self.out, "{}return {};", ind, v);
            }
            Stmt::Return(None) => { let _ = writeln!(self.out, "{}return 0;", ind); }
            Stmt::If { cond, then, els } => {
                let c = self.expr(cond, fl);
                let _ = writeln!(self.out, "{}if ({}) {{", ind, c);
                for s in then { self.stmt(s, d + 1, fl); }
                if let Some(els) = els {
                    let _ = writeln!(self.out, "{}}} else {{", ind);
                    for s in els { self.stmt(s, d + 1, fl); }
                }
                let _ = writeln!(self.out, "{}}}", ind);
            }
            Stmt::While { cond, body } => {
                let c = self.expr(cond, fl);
                let _ = writeln!(self.out, "{}while ({}) {{", ind, c);
                for s in body { self.stmt(s, d + 1, fl); }
                let _ = writeln!(self.out, "{}}}", ind);
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                // hidden counter: body may reassign `var` without affecting iteration
                let sv = self.expr(start, fl);
                let ev = self.expr(end, fl);
                let cmp = if *inclusive { "<=" } else { "<" };
                let _ = writeln!(self.out,
                    "{}for (i64 __c = {}, __lim = {}; __c {} __lim; __c++) {{ {} = __c;",
                    ind, sv, ev, cmp, mangle(var));
                for s in body { self.stmt(s, d + 1, fl); }
                let _ = writeln!(self.out, "{}}}", ind);
            }
            Stmt::Break(None) => { let _ = writeln!(self.out, "{}break;", ind); }
            Stmt::Continue => { let _ = writeln!(self.out, "{}continue;", ind); }
            _ => unreachable!("checked by analyze"),
        }
    }

    fn expr(&mut self, e: &Expr, fl: bool) -> String {
        match e {
            Expr::At { expr, .. } => self.expr(expr, fl),
            Expr::Int(n) => {
                if *n == i64::MIN { "(-9223372036854775807LL - 1)".into() }
                else { format!("{}LL", n) }
            }
            Expr::Float(x) => c_float(*x),
            Expr::Ident(n) => mangle(n),
            Expr::Unary { op, expr } => {
                let v = self.expr(expr, fl);
                match op {
                    UnOp::Neg => format!("(-({}))", v),
                    UnOp::Not => format!("((i64)(({}) == 0))", v),
                    UnOp::BitNot => format!("(~({}))", v),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let a = self.expr(lhs, fl);
                let b = self.expr(rhs, fl);
                match op {
                    BinOp::Add => format!("(({}) + ({}))", a, b),
                    BinOp::Sub => format!("(({}) - ({}))", a, b),
                    BinOp::Mul => format!("(({}) * ({}))", a, b),
                    BinOp::Div => format!("(({}) / ({}))", a, b),
                    BinOp::Rem => format!("(({}) % ({}))", a, b),
                    BinOp::BitOr => format!("(({}) | ({}))", a, b),
                    BinOp::BitXor => format!("(({}) ^ ({}))", a, b),
                    BinOp::BitAnd => format!("(({}) & ({}))", a, b),
                    // Nova: wrapping_shl/shr(b as u32) — mask like Rust does
                    BinOp::Shl => format!("((i64)((uint64_t)({}) << (({}) & 63)))", a, b),
                    BinOp::Shr => format!("(({}) >> (({}) & 63))", a, b),
                    // Nova compares numbers as f64
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let c = match op {
                            BinOp::Lt => "<", BinOp::Le => "<=",
                            BinOp::Gt => ">", _ => ">=",
                        };
                        if fl { format!("((i64)(({}) {} ({})))", a, c, b) }
                        else { format!("((i64)((double)({}) {} (double)({})))", a, c, b) }
                    }
                    BinOp::Eq => format!("((i64)(({}) == ({})))", a, b),
                    BinOp::Ne => format!("((i64)(({}) != ({})))", a, b),
                    BinOp::And => format!("((i64)((({}) != {}) && (({}) != {})))", a, zero(fl), b, zero(fl)),
                    BinOp::Or => format!("((i64)((({}) != {}) || (({}) != {})))", a, zero(fl), b, zero(fl)),
                    BinOp::Pow => unreachable!("rejected by analyze"),
                }
            }
            Expr::If { cond, then, els } => {
                let c = self.expr(cond, fl);
                let t = self.expr(then, fl);
                let e2 = self.expr(els, fl);
                format!("(({}) ? ({}) : ({}))", c, t, e2)
            }
            Expr::Call { callee, args } => {
                let is_float_fn = self.a.float_fns.iter().any(|f| &f.name == callee);
                let vals: Vec<String> = args.iter().map(|a| self.expr(a, is_float_fn)).collect();
                format!("{}({})", mangle(callee), vals.join(", "))
            }
            Expr::Block { stmts, tail } => {
                // GNU statement expression (gcc + clang)
                let mut inner = String::from("({ ");
                let saved = std::mem::take(&mut self.out);
                for s in stmts { self.stmt(s, 0, fl); }
                inner.push_str(&self.out.replace('\n', " "));
                self.out = saved;
                if let Some(t) = tail {
                    let v = self.expr(t, fl);
                    let _ = write!(inner, "{}; }})", v);
                } else {
                    inner.push_str("0; })");
                }
                inner
            }
            _ => unreachable!("checked by analyze"),
        }
    }
}

fn zero(fl: bool) -> &'static str { if fl { "0.0" } else { "0" } }

// a C expression with the exact f64 value (fold_program can produce inf/NaN)
fn c_float(x: f64) -> String {
    if x.is_nan() { "(0.0/0.0)".into() }
    else if x == f64::INFINITY { "(1.0/0.0)".into() }
    else if x == f64::NEG_INFINITY { "(-1.0/0.0)".into() }
    else { format!("{:?}", x) }
}

fn c_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            '\r' => o.push_str("\\r"),
            '%' => o.push('%'), // printed via %s, not a format string
            c => o.push(c),
        }
    }
    o
}

fn collect_vars(body: &[Stmt], out: &mut Vec<String>) {
    for s in body {
        match s {
            Stmt::Let { name, .. } | Stmt::Assign { name, .. } => {
                if !out.contains(name) { out.push(name.clone()); }
            }
            Stmt::ForRange { var, body, .. } | Stmt::ForEach { var, body, .. } => {
                if !out.contains(var) { out.push(var.clone()); }
                collect_vars(body, out);
            }
            Stmt::If { then, els, .. } => {
                collect_vars(then, out);
                if let Some(e) = els { collect_vars(e, out); }
            }
            Stmt::While { body, .. } => collect_vars(body, out),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// LLVM textual IR backend (.ll compiled by system clang; alloca/load/store
// style, mem2reg promotes to registers at -O2)
// ---------------------------------------------------------------------------

struct LlEmit<'p> {
    a: &'p AotProgram<'p>,
    out: String,
    body: String,
    n: usize,
    strs: Vec<String>,
    vars: HashMap<String, (String, bool)>, // name -> (%slot, is_float)
}

impl<'p> LlEmit<'p> {
    fn new(a: &'p AotProgram<'p>) -> Self {
        LlEmit { a, out: String::new(), body: String::new(), n: 0, strs: Vec::new(), vars: HashMap::new() }
    }

    fn tmp(&mut self) -> String { self.n += 1; format!("%t{}", self.n) }
    fn label(&mut self) -> String { self.n += 1; format!("L{}", self.n) }

    fn emit(mut self) -> String {
        self.out.push_str("declare i32 @printf(ptr, ...)\n");
        self.out.push_str("@.fi = private unnamed_addr constant [6 x i8] c\"%lld\\0A\\00\"\n");
        self.out.push_str("@.fs = private unnamed_addr constant [4 x i8] c\"%s\\0A\\00\"\n\n");
        for f in self.a.int_fns.clone() { self.func(f, false); }
        for f in self.a.float_fns.clone() { self.func(f, true); }
        self.func_main();
        let mut header = String::new();
        for (i, s) in self.strs.iter().enumerate() {
            let bytes = ll_string_bytes(s);
            let _ = writeln!(header, "@.s{} = private unnamed_addr constant [{} x i8] c\"{}\"",
                i, s.len() + 1, bytes);
        }
        format!("{}{}", header, self.out)
    }

    fn func(&mut self, f: &Func, fl: bool) {
        let ty = if fl { "double" } else { "i64" };
        self.vars.clear();
        self.body.clear();
        self.n = 0;
        let ps: Vec<String> = (0..f.params.len()).map(|i| format!("{} %a{}", ty, i)).collect();
        let mut head = format!("define internal {} @{}({}) {{\nentry:\n", ty, mangle(&f.name), ps.join(", "));
        let mut names: Vec<String> = f.params.clone();
        collect_vars(&f.body, &mut names);
        for (i, v) in names.iter().enumerate() {
            let slot = format!("%v{}", i);
            let _ = writeln!(head, "  {} = alloca {}", slot, ty);
            self.vars.insert(v.clone(), (slot, fl));
        }
        for (i, p) in f.params.iter().enumerate() {
            let slot = self.vars[p].0.clone();
            let _ = writeln!(head, "  store {} %a{}, ptr {}", ty, i, slot);
        }
        for v in names.iter() {
            if !f.params.contains(v) {
                let slot = self.vars[v].0.clone();
                let z = if fl { "0.0" } else { "0" };
                let _ = writeln!(head, "  store {} {}, ptr {}", ty, z, slot);
            }
        }
        let mut loops = Vec::new();
        for s in &f.body { self.stmt(s, fl, &mut loops); }
        let z = if fl { "0.0" } else { "0" };
        let _ = writeln!(self.body, "  ret {} {}", ty, z);
        self.out.push_str(&head);
        self.out.push_str(&self.body.clone());
        self.out.push_str("}\n\n");
    }

    fn func_main(&mut self) {
        self.vars.clear();
        self.body.clear();
        self.n = 0;
        let mut head = String::from("define i32 @main() {\nentry:\n");
        let mut names: Vec<String> = Vec::new();
        collect_vars(&self.a.main_body, &mut names);
        for (i, v) in names.iter().enumerate() {
            let slot = format!("%v{}", i);
            let _ = writeln!(head, "  {} = alloca i64", slot);
            let _ = writeln!(head, "  store i64 0, ptr {}", slot);
            self.vars.insert(v.clone(), (slot, false));
        }
        let mut loops = Vec::new();
        for s in &self.a.main_body.clone() { self.stmt(s, false, &mut loops); }
        self.body.push_str("  ret i32 0\n");
        self.out.push_str(&head);
        self.out.push_str(&self.body.clone());
        self.out.push_str("}\n\n");
    }

    fn stmt(&mut self, s: &Stmt, fl: bool, loops: &mut Vec<(String, String)>) {
        match s {
            Stmt::Expr(e) => {
                let mut inner = e;
                while let Expr::At { expr, .. } = inner { inner = expr; }
                if let Expr::Call { callee, args } = inner {
                    if callee == "print" && args.len() == 1 {
                        let mut a = &args[0];
                        while let Expr::At { expr, .. } = a { a = expr; }
                        if let Expr::Str(st) = a {
                            let idx = self.strs.len();
                            self.strs.push(st.clone());
                            let t = self.tmp();
                            let _ = writeln!(self.body,
                                "  {} = call i32 (ptr, ...) @printf(ptr @.fs, ptr @.s{})", t, idx);
                        } else {
                            let v = self.expr(a, fl);
                            let t = self.tmp();
                            let _ = writeln!(self.body,
                                "  {} = call i32 (ptr, ...) @printf(ptr @.fi, i64 {})", t, v);
                        }
                        return;
                    }
                }
                self.expr(inner, fl);
            }
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let v = self.expr(value, fl);
                let (slot, sf) = self.vars[name].clone();
                let ty = if sf { "double" } else { "i64" };
                let _ = writeln!(self.body, "  store {} {}, ptr {}", ty, v, slot);
            }
            Stmt::Return(Some(e)) => {
                let v = self.expr(e, fl);
                let ty = if fl { "double" } else { "i64" };
                let _ = writeln!(self.body, "  ret {} {}", ty, v);
                let dead = self.label();
                let _ = writeln!(self.body, "{}:", dead);
            }
            Stmt::Return(None) => {
                let ty = if fl { "double" } else { "i64" };
                let z = if fl { "0.0" } else { "0" };
                let _ = writeln!(self.body, "  ret {} {}", ty, z);
                let dead = self.label();
                let _ = writeln!(self.body, "{}:", dead);
            }
            Stmt::If { cond, then, els } => {
                let c = self.cond_i1(cond, fl);
                let (lt, le, lend) = (self.label(), self.label(), self.label());
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", c, lt, le);
                let _ = writeln!(self.body, "{}:", lt);
                for s in then { self.stmt(s, fl, loops); }
                let _ = writeln!(self.body, "  br label %{}", lend);
                let _ = writeln!(self.body, "{}:", le);
                if let Some(els) = els { for s in els { self.stmt(s, fl, loops); } }
                let _ = writeln!(self.body, "  br label %{}", lend);
                let _ = writeln!(self.body, "{}:", lend);
            }
            Stmt::While { cond, body } => {
                let (lh, lb, lx) = (self.label(), self.label(), self.label());
                let _ = writeln!(self.body, "  br label %{}", lh);
                let _ = writeln!(self.body, "{}:", lh);
                let c = self.cond_i1(cond, fl);
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", c, lb, lx);
                let _ = writeln!(self.body, "{}:", lb);
                loops.push((lh.clone(), lx.clone()));
                for s in body { self.stmt(s, fl, loops); }
                loops.pop();
                let _ = writeln!(self.body, "  br label %{}", lh);
                let _ = writeln!(self.body, "{}:", lx);
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                let sv = self.expr(start, false);
                let ev = self.expr(end, false);
                let cslot = { self.n += 1; format!("%c{}", self.n) };
                let lslot = { self.n += 1; format!("%c{}", self.n) };
                let _ = writeln!(self.body, "  {} = alloca i64", cslot);
                let _ = writeln!(self.body, "  {} = alloca i64", lslot);
                let _ = writeln!(self.body, "  store i64 {}, ptr {}", sv, cslot);
                let _ = writeln!(self.body, "  store i64 {}, ptr {}", ev, lslot);
                let (lh, lb, li, lx) = (self.label(), self.label(), self.label(), self.label());
                let _ = writeln!(self.body, "  br label %{}", lh);
                let _ = writeln!(self.body, "{}:", lh);
                let cv = self.tmp();
                let lv = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", cv, cslot);
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", lv, lslot);
                let cc = if *inclusive { "sle" } else { "slt" };
                let t = self.tmp();
                let _ = writeln!(self.body, "  {} = icmp {} i64 {}, {}", t, cc, cv, lv);
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", t, lb, lx);
                let _ = writeln!(self.body, "{}:", lb);
                let (vslot, _) = self.vars[var].clone();
                let _ = writeln!(self.body, "  store i64 {}, ptr {}", cv, vslot);
                loops.push((li.clone(), lx.clone()));
                for s in body { self.stmt(s, fl, loops); }
                loops.pop();
                let _ = writeln!(self.body, "  br label %{}", li);
                let _ = writeln!(self.body, "{}:", li);
                let cv2 = self.tmp();
                let cv3 = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", cv2, cslot);
                let _ = writeln!(self.body, "  {} = add i64 {}, 1", cv3, cv2);
                let _ = writeln!(self.body, "  store i64 {}, ptr {}", cv3, cslot);
                let _ = writeln!(self.body, "  br label %{}", lh);
                let _ = writeln!(self.body, "{}:", lx);
            }
            Stmt::Break(None) => {
                let lx = loops.last().unwrap().1.clone();
                let _ = writeln!(self.body, "  br label %{}", lx);
                let dead = self.label();
                let _ = writeln!(self.body, "{}:", dead);
            }
            Stmt::Continue => {
                let lh = loops.last().unwrap().0.clone();
                let _ = writeln!(self.body, "  br label %{}", lh);
                let dead = self.label();
                let _ = writeln!(self.body, "{}:", dead);
            }
            _ => unreachable!("checked by analyze"),
        }
    }

    // condition as i1
    fn cond_i1(&mut self, e: &Expr, fl: bool) -> String {
        let v = self.expr(e, fl);
        let t = self.tmp();
        let _ = writeln!(self.body, "  {} = icmp ne i64 {}, 0", t, v);
        t
    }

    fn expr(&mut self, e: &Expr, fl: bool) -> String {
        match e {
            Expr::At { expr, .. } => self.expr(expr, fl),
            Expr::Int(n) => format!("{}", n),
            // exact bit pattern — LLVM accepts hex64 double constants
            Expr::Float(x) => format!("0x{:016X}", x.to_bits()),
            Expr::Ident(n) => {
                let (slot, sf) = self.vars[n].clone();
                let ty = if sf { "double" } else { "i64" };
                let t = self.tmp();
                let _ = writeln!(self.body, "  {} = load {}, ptr {}", t, ty, slot);
                t
            }
            Expr::Unary { op, expr } => {
                let v = self.expr(expr, fl);
                let t = self.tmp();
                match op {
                    UnOp::Neg => {
                        if fl { let _ = writeln!(self.body, "  {} = fneg double {}", t, v); }
                        else { let _ = writeln!(self.body, "  {} = sub i64 0, {}", t, v); }
                    }
                    UnOp::Not => {
                        let c = self.tmp();
                        let _ = writeln!(self.body, "  {} = icmp eq i64 {}, 0", t, v);
                        let _ = writeln!(self.body, "  {} = zext i1 {} to i64", c, t);
                        return c;
                    }
                    UnOp::BitNot => { let _ = writeln!(self.body, "  {} = xor i64 {}, -1", t, v); }
                }
                t
            }
            Expr::Binary { op, lhs, rhs } => self.binop(*op, lhs, rhs, fl),
            Expr::If { cond, then, els } => {
                let ty = if fl { "double" } else { "i64" };
                let slot = { self.n += 1; format!("%c{}", self.n) };
                let _ = writeln!(self.body, "  {} = alloca {}", slot, ty);
                let c = self.cond_i1(cond, fl);
                let (lt, le, lend) = (self.label(), self.label(), self.label());
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", c, lt, le);
                let _ = writeln!(self.body, "{}:", lt);
                let tv = self.expr(then, fl);
                let _ = writeln!(self.body, "  store {} {}, ptr {}", ty, tv, slot);
                let _ = writeln!(self.body, "  br label %{}", lend);
                let _ = writeln!(self.body, "{}:", le);
                let ev = self.expr(els, fl);
                let _ = writeln!(self.body, "  store {} {}, ptr {}", ty, ev, slot);
                let _ = writeln!(self.body, "  br label %{}", lend);
                let _ = writeln!(self.body, "{}:", lend);
                let t = self.tmp();
                let _ = writeln!(self.body, "  {} = load {}, ptr {}", t, ty, slot);
                t
            }
            Expr::Call { callee, args } => {
                let is_float_fn = self.a.float_fns.iter().any(|f| &f.name == callee);
                let ty = if is_float_fn { "double" } else { "i64" };
                let vals: Vec<String> = args.iter()
                    .map(|a| format!("{} {}", ty, self.expr(a, is_float_fn))).collect();
                let t = self.tmp();
                let _ = writeln!(self.body, "  {} = call {} @{}({})", t, ty, mangle(callee), vals.join(", "));
                t
            }
            Expr::Block { stmts, tail } => {
                let mut loops = Vec::new();
                for s in stmts { self.stmt(s, fl, &mut loops); }
                match tail {
                    Some(t) => self.expr(t, fl),
                    None => "0".into(),
                }
            }
            _ => unreachable!("checked by analyze"),
        }
    }

    fn binop(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr, fl: bool) -> String {
        use BinOp::*;
        if matches!(op, And | Or) {
            let a = self.expr(lhs, fl);
            let b = self.expr(rhs, fl);
            let (ta, tb, t, r) = (self.tmp(), self.tmp(), self.tmp(), self.tmp());
            let _ = writeln!(self.body, "  {} = icmp ne i64 {}, 0", ta, a);
            let _ = writeln!(self.body, "  {} = icmp ne i64 {}, 0", tb, b);
            let o = if matches!(op, And) { "and" } else { "or" };
            let _ = writeln!(self.body, "  {} = {} i1 {}, {}", t, o, ta, tb);
            let _ = writeln!(self.body, "  {} = zext i1 {} to i64", r, t);
            return r;
        }
        let a = self.expr(lhs, fl);
        let b = self.expr(rhs, fl);
        let t = self.tmp();
        if fl {
            let ins = match op {
                Add => "fadd", Sub => "fsub", Mul => "fmul", Div => "fdiv",
                Eq | Ne | Lt | Le | Gt | Ge => {
                    let cc = match op {
                        Eq => "oeq", Ne => "une", Lt => "olt",
                        Le => "ole", Gt => "ogt", _ => "oge",
                    };
                    let r = self.tmp();
                    let _ = writeln!(self.body, "  {} = fcmp {} double {}, {}", t, cc, a, b);
                    let _ = writeln!(self.body, "  {} = zext i1 {} to i64", r, t);
                    return r;
                }
                _ => unreachable!(),
            };
            let _ = writeln!(self.body, "  {} = {} double {}, {}", t, ins, a, b);
            return t;
        }
        match op {
            Add | Sub | Mul | Div | Rem | BitOr | BitXor | BitAnd => {
                let ins = match op {
                    Add => "add", Sub => "sub", Mul => "mul",
                    Div => "sdiv", Rem => "srem",
                    BitOr => "or", BitXor => "xor", _ => "and",
                };
                let _ = writeln!(self.body, "  {} = {} i64 {}, {}", t, ins, a, b);
                t
            }
            Shl | Shr => {
                let m = self.tmp();
                let _ = writeln!(self.body, "  {} = and i64 {}, 63", m, b);
                let ins = if matches!(op, Shl) { "shl" } else { "ashr" };
                let _ = writeln!(self.body, "  {} = {} i64 {}, {}", t, ins, a, m);
                t
            }
            Eq | Ne => {
                let cc = if matches!(op, Eq) { "eq" } else { "ne" };
                let r = self.tmp();
                let _ = writeln!(self.body, "  {} = icmp {} i64 {}, {}", t, cc, a, b);
                let _ = writeln!(self.body, "  {} = zext i1 {} to i64", r, t);
                r
            }
            Lt | Le | Gt | Ge => {
                // as f64, like the interpreter
                let (fa, fb, r) = (self.tmp(), self.tmp(), self.tmp());
                let _ = writeln!(self.body, "  {} = sitofp i64 {} to double", fa, a);
                let _ = writeln!(self.body, "  {} = sitofp i64 {} to double", fb, b);
                let cc = match op { Lt => "olt", Le => "ole", Gt => "ogt", _ => "oge" };
                let _ = writeln!(self.body, "  {} = fcmp {} double {}, {}", t, cc, fa, fb);
                let _ = writeln!(self.body, "  {} = zext i1 {} to i64", r, t);
                r
            }
            _ => unreachable!(),
        }
    }
}

fn ll_string_bytes(s: &str) -> String {
    let mut o = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b == b' ' { o.push(b as char); }
        else { let _ = write!(o, "\\{:02X}", b); }
    }
    o.push_str("\\00");
    o
}

// ---------------------------------------------------------------------------
// Boxed tier (Phase 8): whole-program compilation against the refcounted C
// runtime (runtime/nova_rt.c, #included into the generated file so everything
// is one translation unit and clang/gcc -O3 inlines the runtime into program
// code). Ownership convention: every emitted expression yields an OWNED value;
// the x* wrapper functions consume their arguments; stores release the old
// slot value; functions take owned params and release everything on return.
// ---------------------------------------------------------------------------

struct BoxedProgram<'p> {
    ints: HashSet<String>,
    floats: HashSet<String>,
    boxed: Vec<&'p Func>,
    int_fns: Vec<&'p Func>,
    float_fns: Vec<&'p Func>,
    main_body: Vec<Stmt>,
}

fn analyze_boxed(prog: &Program) -> Option<BoxedProgram<'_>> {
    let ints = eligible_set(prog);
    let floats = float_eligible_set(prog, &ints);
    let mut int_fns = Vec::new();
    let mut float_fns = Vec::new();
    let mut rest: Vec<&Func> = Vec::new();
    let mut main = None;
    for item in &prog.items {
        let f = match item {
            Item::Func(f) => f,
            Item::Const { .. } | Item::Test(_) => continue,
            _ => return None,
        };
        if f.name == "main" { main = Some(f); continue; }
        if ints.contains(&f.name) && fn_body_ok(f, &ints) { int_fns.push(f); }
        else if floats.contains(&f.name) {
            let mut sc = vec![HashSet::new()];
            for p in &f.params { sc.last_mut().unwrap().insert(p.clone()); }
            if float_body_no_shadow(&f.body, &mut sc) { float_fns.push(f); } else { rest.push(f); }
        }
        else { rest.push(f); }
    }
    let main = main?;
    if !main.params.is_empty() { return None; }
    // fixpoint: a boxed fn may call ints/floats/boxed fns
    let mut boxed_names: HashSet<String> = rest.iter().map(|f| f.name.clone()).collect();
    loop {
        let mut drop_name = None;
        for f in &rest {
            if !boxed_names.contains(&f.name) { continue; }
            let mut known: HashSet<String> = ints.clone();
            known.extend(floats.iter().cloned());
            known.extend(boxed_names.iter().cloned());
            let mut scopes = vec![HashSet::new()];
            for p in &f.params { scopes.last_mut().unwrap().insert(p.clone()); }
            let ok = f.body.iter().all(|s| bx_stmt(s, &known, &mut scopes, true));
            if !ok { drop_name = Some(f.name.clone()); break; }
        }
        match drop_name { Some(n) => { boxed_names.remove(&n); } None => break }
    }
    if boxed_names.len() != rest.len() { return None; } // some fn fits no tier
    let mut main_body = main.body.clone();
    fix_main_tail(&mut main_body);
    let mut known: HashSet<String> = ints.clone();
    known.extend(floats.iter().cloned());
    known.extend(boxed_names.iter().cloned());
    let mut scopes = vec![HashSet::new()];
    if !main_body.iter().all(|s| bx_stmt(s, &known, &mut scopes, false)) { return None; }
    Some(BoxedProgram { ints, floats, boxed: rest, int_fns, float_fns, main_body })
}

fn bx_stmt(s: &Stmt, known: &HashSet<String>, scopes: &mut Vec<HashSet<String>>, allow_ret: bool) -> bool {
    match s {
        Stmt::Return(Some(e)) => allow_ret && bx_expr(e, known, scopes),
        Stmt::Return(None) => true,
        Stmt::Expr(e) => bx_expr(e, known, scopes),
        Stmt::Let { name, value, .. } => {
            if !bx_expr(value, known, scopes) { return false; }
            if scopes.iter().any(|sc| sc.contains(name)) { return false; }
            scopes.last_mut().unwrap().insert(name.clone());
            true
        }
        Stmt::Assign { name, value } => {
            if !bx_expr(value, known, scopes) { return false; }
            if !scopes.iter().any(|sc| sc.contains(name)) {
                scopes.last_mut().unwrap().insert(name.clone());
            }
            true
        }
        Stmt::IndexAssign { base, index, value } =>
            bx_expr(base, known, scopes) && bx_expr(index, known, scopes)
            && bx_expr(value, known, scopes),
        Stmt::If { cond, then, els } => {
            if !bx_expr(cond, known, scopes) { return false; }
            scopes.push(HashSet::new());
            let a = then.iter().all(|s| bx_stmt(s, known, scopes, allow_ret));
            scopes.pop();
            scopes.push(HashSet::new());
            let b = els.as_ref().map_or(true, |e| e.iter().all(|s| bx_stmt(s, known, scopes, allow_ret)));
            scopes.pop();
            a && b
        }
        Stmt::While { cond, body } => {
            if !bx_expr(cond, known, scopes) { return false; }
            scopes.push(HashSet::new());
            let r = body.iter().all(|s| bx_stmt(s, known, scopes, allow_ret));
            scopes.pop();
            r
        }
        Stmt::ForRange { var, start, end, body, .. } => {
            if !bx_expr(start, known, scopes) || !bx_expr(end, known, scopes) { return false; }
            if scopes.iter().any(|sc| sc.contains(var)) { return false; }
            scopes.push(HashSet::new());
            scopes.last_mut().unwrap().insert(var.clone());
            let r = body.iter().all(|s| bx_stmt(s, known, scopes, allow_ret));
            scopes.pop();
            r
        }
        Stmt::ForEach { var, iter, body } => {
            if !bx_expr(iter, known, scopes) { return false; }
            if scopes.iter().any(|sc| sc.contains(var)) { return false; }
            scopes.push(HashSet::new());
            scopes.last_mut().unwrap().insert(var.clone());
            let r = body.iter().all(|s| bx_stmt(s, known, scopes, allow_ret));
            scopes.pop();
            r
        }
        Stmt::Break(None) | Stmt::Continue => true,
        _ => false,
    }
}

fn bx_expr(e: &Expr, known: &HashSet<String>, scopes: &Vec<HashSet<String>>) -> bool {
    match e {
        Expr::At { expr, .. } => bx_expr(expr, known, scopes),
        Expr::Int(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::Null => true,
        Expr::Ident(n) => scopes.iter().any(|sc| sc.contains(n)),
        Expr::Unary { op, expr } =>
            matches!(op, UnOp::Neg | UnOp::Not | UnOp::BitNot) && bx_expr(expr, known, scopes),
        Expr::Binary { op, lhs, rhs } =>
            !matches!(op, BinOp::Pow) && bx_expr(lhs, known, scopes) && bx_expr(rhs, known, scopes),
        Expr::If { cond, then, els } =>
            bx_expr(cond, known, scopes) && bx_expr(then, known, scopes) && bx_expr(els, known, scopes),
        // if-expression branches are tail-only Blocks; statement-carrying
        // blocks in expression position stay on the VM
        Expr::Block { stmts, tail } =>
            stmts.is_empty() && tail.as_ref().map_or(false, |t| bx_expr(t, known, scopes)),
        Expr::Array(xs) => xs.iter().all(|x| bx_expr(x, known, scopes)),
        Expr::Index { base, index } => {
            if !bx_expr(base, known, scopes) { return false; }
            if let Expr::RangeLit { lo, hi, .. } = &**index {
                lo.as_ref().map_or(true, |e| bx_expr(e, known, scopes))
                    && hi.as_ref().map_or(true, |e| bx_expr(e, known, scopes))
            } else {
                bx_expr(index, known, scopes)
            }
        }
        Expr::RangeLit { lo: Some(lo), hi: Some(hi), .. } =>
            bx_expr(lo, known, scopes) && bx_expr(hi, known, scopes),
        Expr::FmtStr(parts) => parts.iter().all(|p| match p {
            FmtPart::Lit(_) => true,
            FmtPart::Expr(e) => bx_expr(e, known, scopes),
        }),
        Expr::Call { callee, args } => {
            let ok_callee = known.contains(callee)
                || matches!(callee.as_str(), "print" | "len" | "push" | "pop");
            ok_callee && args.iter().all(|a| bx_expr(a, known, scopes))
                && !(callee == "print" && args.len() != 1)
        }
        _ => false,
    }
}

fn emit_boxed(prog: &Program) -> Option<String> {
    let b = analyze_boxed(prog)?;
    let mut e = BxEmit { b: &b, out: String::new() };
    Some(e.emit())
}

struct BxEmit<'p, 'a> {
    b: &'a BoxedProgram<'p>,
    out: String,
}

impl<'p, 'a> BxEmit<'p, 'a> {
    fn emit(&mut self) -> String {
        self.out.push_str("#include \"nova_rt.c\"\n\n");
        // consuming wrappers: arguments owned, result owned
        self.out.push_str(r#"static NV xadd(NV a, NV b){ NV r=nv_add(a,b); nv_release(a); nv_release(b); return r; }
static NV xsub(NV a, NV b){ NV r=nv_sub(a,b); nv_release(a); nv_release(b); return r; }
static NV xmul(NV a, NV b){ NV r=nv_mul(a,b); nv_release(a); nv_release(b); return r; }
static NV xdiv(NV a, NV b){ NV r=nv_div(a,b); nv_release(a); nv_release(b); return r; }
static NV xrem(NV a, NV b){ NV r=nv_rem(a,b); nv_release(a); nv_release(b); return r; }
static NV xlt(NV a, NV b){ NV r=nv_cmp_lt(a,b); nv_release(a); nv_release(b); return r; }
static NV xle(NV a, NV b){ NV r=nv_cmp_le(a,b); nv_release(a); nv_release(b); return r; }
static NV xgt(NV a, NV b){ NV r=nv_cmp_gt(a,b); nv_release(a); nv_release(b); return r; }
static NV xge(NV a, NV b){ NV r=nv_cmp_ge(a,b); nv_release(a); nv_release(b); return r; }
static NV xeq(NV a, NV b){ NV r=nv_bool(nv_eq(a,b)); nv_release(a); nv_release(b); return r; }
static NV xne(NV a, NV b){ NV r=nv_bool(!nv_eq(a,b)); nv_release(a); nv_release(b); return r; }
static NV xbit(NV a, NV b, char op){ NV r=nv_bit(a,b,op); nv_release(a); nv_release(b); return r; }
static NV xneg(NV a){ NV r=nv_neg(a); nv_release(a); return r; }
static NV xnot(NV a){ NV r=nv_not(a); nv_release(a); return r; }
static NV xbnot(NV a){ NV r=nv_bitnot(a); nv_release(a); return r; }
static i64 xtruthy(NV a){ i64 r=nv_truthy(a); nv_release(a); return r; }
static NV xindex(NV b, NV i){ NV r=nv_index(b,i); nv_release(b); nv_release(i); return r; }
static NV xslice(NV b, NV lo, int hl, NV hi, int hh, int inc){
  i64 l = hl ? nv_as_int(lo) : 0; i64 h = hh ? nv_as_int(hi) : 0;
  NV r = nv_slice(b, l, hl, h, hh, inc); nv_release(b); return r; }
static NV xrange(NV lo, NV hi, int inc){ NV r=nv_range(nv_as_int(lo), nv_as_int(hi), inc); return r; }
static NV xlen(NV a){ i64 n=nv_len(a); nv_release(a); return nv_int(n); }
static NV xcat(NV a, NV b){ NV r=nv_concat2(a,b); nv_release(a); nv_release(b); return r; }
static NV xtostr(NV a){ NV r=nv_tostr(a); nv_release(a); return r; }
static i64 xint(NV a){ i64 r=nv_as_int(a); nv_release(a); return r; }
static double xflt(NV a){ double r=nv_as_float(a); nv_release(a); return r; }
"#);
        self.out.push('\n');
        // typed-tier functions keep their fast unboxed form
        let ap = AotProgram {
            int_fns: self.b.int_fns.clone(),
            float_fns: self.b.float_fns.clone(),
            main_body: Vec::new(),
        };
        let mut typed = CEmit::new(&ap);
        for f in &self.b.int_fns { typed.proto(f, "i64"); }
        for f in &self.b.float_fns { typed.proto(f, "double"); }
        for f in self.b.int_fns.clone() { typed.func(f, false); }
        for f in self.b.float_fns.clone() { typed.func(f, true); }
        self.out.push_str(&typed.out);
        // boxed prototypes then bodies
        for f in &self.b.boxed {
            let ps: Vec<String> = f.params.iter().map(|p| format!("NV {}", mangle(p))).collect();
            let _ = writeln!(self.out, "static NV {}({});", mangle(&f.name), ps.join(", "));
        }
        self.out.push('\n');
        for f in self.b.boxed.clone() { self.func(f); }
        self.main();
        std::mem::take(&mut self.out)
    }

    fn func(&mut self, f: &Func) {
        let ps: Vec<String> = f.params.iter().map(|p| format!("NV {}", mangle(p))).collect();
        let _ = writeln!(self.out, "static NV {}({}) {{", mangle(&f.name), ps.join(", "));
        let mut vars: Vec<String> = Vec::new();
        collect_vars(&f.body, &mut vars);
        for v in &vars {
            if !f.params.contains(v) {
                let _ = writeln!(self.out, "  NV {} = nv_null();", mangle(v));
            }
        }
        let all: Vec<String> = f.params.iter().cloned().chain(
            vars.iter().filter(|v| !f.params.contains(v)).cloned()).collect();
        for s in &f.body { self.stmt(s, 1, &all); }
        for v in &all { let _ = writeln!(self.out, "  nv_release({});", mangle(v)); }
        let _ = writeln!(self.out, "  return nv_null();\n}}\n");
    }

    fn main(&mut self) {
        self.out.push_str("int main(void) {\n");
        let mut vars: Vec<String> = Vec::new();
        collect_vars(&self.b.main_body, &mut vars);
        for v in &vars { let _ = writeln!(self.out, "  NV {} = nv_null();", mangle(v)); }
        let body = self.b.main_body.clone();
        for s in &body { self.stmt(s, 1, &vars); }
        // release every local before exit, mirroring `func`; without this a
        // main-scope heap value (array/string) is leaked at program exit
        for v in &vars { let _ = writeln!(self.out, "  nv_release({});", mangle(v)); }
        self.out.push_str("  return 0;\n}\n");
    }

    fn stmt(&mut self, s: &Stmt, d: usize, all_vars: &[String]) {
        let ind = "  ".repeat(d);
        match s {
            Stmt::Expr(e) => {
                let mut inner = e;
                while let Expr::At { expr, .. } = inner { inner = expr; }
                if let Expr::Call { callee, args } = inner {
                    match callee.as_str() {
                        "print" => {
                            let v = self.expr(&args[0]);
                            let _ = writeln!(self.out, "{}{{ NV __p = {}; nv_print(__p); nv_release(__p); }}", ind, v);
                            return;
                        }
                        "push" if args.len() == 2 => {
                            let a = self.expr(&args[0]);
                            let v = self.expr(&args[1]);
                            let _ = writeln!(self.out, "{}{{ NV __a = {}; nv_arr_push(__a, {}); nv_release(__a); }}", ind, a, v);
                            return;
                        }
                        _ => {}
                    }
                }
                let v = self.expr(inner);
                let _ = writeln!(self.out, "{}nv_release({});", ind, v);
            }
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let v = self.expr(value);
                let _ = writeln!(self.out, "{}{{ NV __t = {}; nv_release({}); {} = __t; }}", ind, v, mangle(name), mangle(name));
            }
            Stmt::IndexAssign { base, index, value } => {
                let bexp = self.expr(base);
                let i = self.expr(index);
                let v = self.expr(value);
                let _ = writeln!(self.out,
                    "{}{{ NV __b = {}; NV __i = {}; nv_index_set(__b, __i, {}); nv_release(__b); nv_release(__i); }}",
                    ind, bexp, i, v);
            }
            Stmt::Return(Some(e)) => {
                let v = self.expr(e);
                let _ = write!(self.out, "{}{{ NV __r = {};", ind, v);
                for v in all_vars { let _ = write!(self.out, " nv_release({});", mangle(v)); }
                let _ = writeln!(self.out, " return __r; }}");
            }
            Stmt::Return(None) => {
                let _ = write!(self.out, "{}{{", ind);
                for v in all_vars { let _ = write!(self.out, " nv_release({});", mangle(v)); }
                let _ = writeln!(self.out, " return nv_null(); }}");
            }
            Stmt::If { cond, then, els } => {
                let c = self.expr(cond);
                let _ = writeln!(self.out, "{}if (xtruthy({})) {{", ind, c);
                for s in then { self.stmt(s, d + 1, all_vars); }
                if let Some(els) = els {
                    let _ = writeln!(self.out, "{}}} else {{", ind);
                    for s in els { self.stmt(s, d + 1, all_vars); }
                }
                let _ = writeln!(self.out, "{}}}", ind);
            }
            Stmt::While { cond, body } => {
                let c = self.expr(cond);
                let _ = writeln!(self.out, "{}while (xtruthy({})) {{", ind, c);
                for s in body { self.stmt(s, d + 1, all_vars); }
                let _ = writeln!(self.out, "{}}}", ind);
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                let sv = self.expr(start);
                let ev = self.expr(end);
                let cmp = if *inclusive { "<=" } else { "<" };
                let _ = writeln!(self.out,
                    "{}for (i64 __c = xint({}), __lim = xint({}); __c {} __lim; __c++) {{ nv_release({}); {} = nv_int(__c);",
                    ind, sv, ev, cmp, mangle(var), mangle(var));
                for s in body { self.stmt(s, d + 1, all_vars); }
                let _ = writeln!(self.out, "{}}}", ind);
            }
            Stmt::ForEach { var, iter, body } => {
                // iterate over a snapshot, like the interpreter (well-defined
                // even if the body mutates the array)
                let it = self.expr(iter);
                let _ = writeln!(self.out,
                    "{}{{ NV __it0 = {}; NV __it = (__it0.tag == NV_ARR) ? xslice(__it0, nv_int(0), 0, nv_int(0), 0, 0) : __it0;",
                    ind, it);
                let _ = writeln!(self.out,
                    "{}  for (i64 __i = 0, __n = nv_len(__it); __i < __n; __i++) {{ nv_release({}); {} = nv_index(__it, nv_int(__i));",
                    ind, mangle(var), mangle(var));
                for s in body { self.stmt(s, d + 2, all_vars); }
                let _ = writeln!(self.out, "{}  }}\n{}  nv_release(__it); }}", ind, ind);
            }
            Stmt::Break(None) => { let _ = writeln!(self.out, "{}break;", ind); }
            Stmt::Continue => { let _ = writeln!(self.out, "{}continue;", ind); }
            _ => unreachable!("checked by analyze_boxed"),
        }
    }

    fn expr(&mut self, e: &Expr) -> String {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Int(n) => {
                if *n == i64::MIN { "nv_int(-9223372036854775807LL - 1)".into() }
                else { format!("nv_int({}LL)", n) }
            }
            Expr::Float(x) => format!("nv_float({})", c_float(*x)),
            Expr::Bool(b) => format!("nv_bool({})", if *b { 1 } else { 0 }),
            Expr::Null => "nv_null()".into(),
            Expr::Str(s) => format!("nv_str(\"{}\")", c_escape(s)),
            Expr::Ident(n) => format!("nv_retain({})", mangle(n)),
            Expr::Unary { op, expr } => {
                let v = self.expr(expr);
                match op {
                    UnOp::Neg => format!("xneg({})", v),
                    UnOp::Not => format!("xnot({})", v),
                    UnOp::BitNot => format!("xbnot({})", v),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    let a = self.expr(lhs);
                    let b = self.expr(rhs);
                    // short-circuit via GNU statement expression
                    return if matches!(op, BinOp::And) {
                        format!("({{ NV __r; if (xtruthy({})) __r = nv_bool(xtruthy({})); else __r = nv_bool(0); __r; }})", a, b)
                    } else {
                        format!("({{ NV __r; if (xtruthy({})) __r = nv_bool(1); else __r = nv_bool(xtruthy({})); __r; }})", a, b)
                    };
                }
                let a = self.expr(lhs);
                let b = self.expr(rhs);
                match op {
                    BinOp::Add => format!("xadd({}, {})", a, b),
                    BinOp::Sub => format!("xsub({}, {})", a, b),
                    BinOp::Mul => format!("xmul({}, {})", a, b),
                    BinOp::Div => format!("xdiv({}, {})", a, b),
                    BinOp::Rem => format!("xrem({}, {})", a, b),
                    BinOp::Lt => format!("xlt({}, {})", a, b),
                    BinOp::Le => format!("xle({}, {})", a, b),
                    BinOp::Gt => format!("xgt({}, {})", a, b),
                    BinOp::Ge => format!("xge({}, {})", a, b),
                    BinOp::Eq => format!("xeq({}, {})", a, b),
                    BinOp::Ne => format!("xne({}, {})", a, b),
                    BinOp::BitAnd => format!("xbit({}, {}, '&')", a, b),
                    BinOp::BitOr => format!("xbit({}, {}, '|')", a, b),
                    BinOp::BitXor => format!("xbit({}, {}, '^')", a, b),
                    BinOp::Shl => format!("xbit({}, {}, '<')", a, b),
                    BinOp::Shr => format!("xbit({}, {}, '>')", a, b),
                    _ => unreachable!(),
                }
            }
            Expr::If { cond, then, els } => {
                let c = self.expr(cond);
                let t = self.expr(then);
                let e2 = self.expr(els);
                format!("(xtruthy({}) ? ({}) : ({}))", c, t, e2)
            }
            Expr::Array(xs) => {
                let mut s = format!("({{ NV __a = nv_arr({});", xs.len().max(1));
                for x in xs {
                    let v = self.expr(x);
                    let _ = write!(s, " nv_arr_push(__a, {});", v);
                }
                s.push_str(" __a; })");
                s
            }
            Expr::Index { base, index } => {
                if let Expr::RangeLit { lo, hi, inclusive } = &**index {
                    let b = self.expr(base);
                    let lo_s = lo.as_ref().map(|e| self.expr(e)).unwrap_or("nv_int(0)".into());
                    let hi_s = hi.as_ref().map(|e| self.expr(e)).unwrap_or("nv_int(0)".into());
                    format!("xslice({}, {}, {}, {}, {}, {})", b, lo_s,
                        lo.is_some() as u8, hi_s, hi.is_some() as u8, *inclusive as u8)
                } else {
                    let b = self.expr(base);
                    let i = self.expr(index);
                    format!("xindex({}, {})", b, i)
                }
            }
            Expr::RangeLit { lo: Some(lo), hi: Some(hi), inclusive } => {
                let l = self.expr(lo);
                let h = self.expr(hi);
                format!("xrange({}, {}, {})", l, h, *inclusive as u8)
            }
            Expr::FmtStr(parts) => {
                let mut s = String::from("nv_str(\"\")");
                for p in parts {
                    let piece = match p {
                        FmtPart::Lit(t) => format!("nv_str(\"{}\")", c_escape(t)),
                        FmtPart::Expr(e) => { let v = self.expr(e); format!("xtostr({})", v) }
                    };
                    s = format!("xcat({}, {})", s, piece);
                }
                s
            }
            Expr::Call { callee, args } => {
                if callee == "len" {
                    let a = self.expr(&args[0]);
                    return format!("xlen({})", a);
                }
                if callee == "pop" {
                    let a = self.expr(&args[0]);
                    return format!("({{ NV __a = {}; NV __v = nv_pop(__a); nv_release(__a); __v; }})", a);
                }
                if callee == "push" {
                    let a = self.expr(&args[0]);
                    let v = self.expr(&args[1]);
                    return format!("({{ NV __a = {}; nv_arr_push(__a, {}); nv_release(__a); nv_null(); }})", a, v);
                }
                if callee == "print" {
                    let v = self.expr(&args[0]);
                    return format!("({{ NV __p = {}; nv_print(__p); nv_release(__p); nv_null(); }})", v);
                }
                if self.b.ints.contains(callee) {
                    let vals: Vec<String> = args.iter().map(|a| { let v = self.expr(a); format!("xint({})", v) }).collect();
                    return format!("nv_int({}({}))", mangle(callee), vals.join(", "));
                }
                if self.b.floats.contains(callee) {
                    let vals: Vec<String> = args.iter().map(|a| { let v = self.expr(a); format!("xflt({})", v) }).collect();
                    return format!("nv_float({}({}))", mangle(callee), vals.join(", "));
                }
                let vals: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                format!("{}({})", mangle(callee), vals.join(", "))
            }
            Expr::Block { stmts, tail } => {
                let mut s = String::from("({ ");
                let saved = std::mem::take(&mut self.out);
                for st in stmts { self.stmt(st, 0, &[]); }
                s.push_str(&self.out.replace('\n', " "));
                self.out = saved;
                match tail {
                    Some(t) => { let v = self.expr(t); let _ = write!(s, "{}; }})", v); }
                    None => s.push_str("nv_null(); })"),
                }
                s
            }
            _ => unreachable!("checked by analyze_boxed"),
        }
    }
}

// ---- LLVM textual-IR boxed backend -------------------------------------
//
// Mirrors the C boxed emitter (`BxEmit`) but emits LLVM IR that calls the same
// `runtime/nova_rt.c` through clang's ABI for the `NV` value type: an NV passed
// by value becomes two arguments `(i8 tag, i64 payload)`, and an NV return is
// the aggregate `{i8, i64}`. To keep the per-expression logic simple, every
// runtime call is funnelled through a fixed set of `x*`/`nvb_*` wrapper
// functions that take and return the `{i8,i64}` aggregate directly and do the
// (tag,payload) splitting internally; the emitter itself only ever moves
// `{i8,i64}` SSA values around. Ownership matches `BxEmit` exactly (expressions
// yield owned values; wrappers consume their arguments; slots release the old
// value on store; functions release every local on return) so output — and heap
// accounting — is identical, which the build-time byte-diff gate confirms.

fn emit_boxed_llvm(prog: &Program) -> Option<String> {
    let b = analyze_boxed(prog)?;
    Some(LlBox::new(&b).emit())
}

const NV: &str = "{ i8, i64 }";

struct LlBox<'p, 'a> {
    b: &'a BoxedProgram<'p>,
    out: String,
    body: String,
    n: usize,
    strs: Vec<String>,
    slots: HashMap<String, String>, // var -> %slot (alloca {i8,i64})
    loops: Vec<(String, String)>,   // (continue-label, break-label)
}

impl<'p, 'a> LlBox<'p, 'a> {
    fn new(b: &'a BoxedProgram<'p>) -> Self {
        LlBox { b, out: String::new(), body: String::new(), n: 0,
                strs: Vec::new(), slots: HashMap::new(), loops: Vec::new() }
    }
    fn tmp(&mut self) -> String { self.n += 1; format!("%t{}", self.n) }
    fn lbl(&mut self, s: &str) -> String { self.n += 1; format!("{}{}", s, self.n) }

    fn emit(mut self) -> String {
        // typed-tier functions reuse the proven unboxed .ll backend
        let ap = AotProgram {
            int_fns: self.b.int_fns.clone(),
            float_fns: self.b.float_fns.clone(),
            main_body: Vec::new(),
        };
        let mut typed = LlEmit::new(&ap);
        typed.out.clear();
        for f in ap.int_fns.clone() { typed.func(f, false); }
        for f in ap.float_fns.clone() { typed.func(f, true); }
        let typed_fns = typed.out.clone();

        // boxed function bodies + main (fills self.strs as a side effect)
        for f in self.b.boxed.clone() { self.func(f); }
        self.func_main();

        // assemble: declarations, wrappers, string constants, then bodies
        let mut head = String::new();
        head.push_str(&ll_boxed_preamble());
        for (i, s) in self.strs.iter().enumerate() {
            let bytes = ll_string_bytes(s);
            let _ = writeln!(head, "@.sb{} = private unnamed_addr constant [{} x i8] c\"{}\"",
                i, s.len() + 1, bytes);
        }
        head.push('\n');
        format!("{}{}\n{}", head, typed_fns, self.out)
    }

    fn func(&mut self, f: &Func) {
        self.slots.clear();
        self.body.clear();
        self.n = 0;
        let ps: Vec<String> = (0..f.params.len()).map(|i| format!("{} %a{}", NV, i)).collect();
        let mut head = format!("define internal {} @{}({}) {{\nentry:\n", NV, mangle(&f.name), ps.join(", "));
        let mut names: Vec<String> = f.params.clone();
        collect_vars(&f.body, &mut names);
        for (i, v) in names.iter().enumerate() {
            let slot = format!("%v{}", i);
            let _ = writeln!(head, "  {} = alloca {}", slot, NV);
            self.slots.insert(v.clone(), slot);
        }
        for (i, p) in f.params.iter().enumerate() {
            let slot = self.slots[p].clone();
            let _ = writeln!(head, "  store {} %a{}, ptr {}", NV, i, slot);
        }
        for v in names.iter().filter(|v| !f.params.contains(v)) {
            let slot = self.slots[v].clone();
            let _ = writeln!(head, "  store {} zeroinitializer, ptr {}", NV, slot);
        }
        let all: Vec<String> = names.clone();
        for s in &f.body { self.stmt(s, &all); }
        // fall-through return: release every local, return null
        for v in &all { let sl = self.slots[v].clone(); self.release_slot(&sl); }
        let z = self.call(NV, "@nv_null()");
        let _ = writeln!(self.body, "  ret {} {}", NV, z);
        self.out.push_str(&head);
        self.out.push_str(&self.body.clone());
        self.out.push_str("}\n\n");
    }

    fn func_main(&mut self) {
        self.slots.clear();
        self.body.clear();
        self.n = 0;
        let mut head = String::from("define i32 @main() {\nentry:\n");
        let mut names: Vec<String> = Vec::new();
        collect_vars(&self.b.main_body, &mut names);
        for (i, v) in names.iter().enumerate() {
            let slot = format!("%v{}", i);
            let _ = writeln!(head, "  {} = alloca {}", slot, NV);
            let _ = writeln!(head, "  store {} zeroinitializer, ptr {}", NV, slot);
            self.slots.insert(v.clone(), slot);
        }
        let body = self.b.main_body.clone();
        for s in &body { self.stmt(s, &names); }
        for v in &names { let sl = self.slots[v].clone(); self.release_slot(&sl); }
        self.body.push_str("  ret i32 0\n");
        self.out.push_str(&head);
        self.out.push_str(&self.body.clone());
        self.out.push_str("}\n\n");
    }

    // emit `%t = <rhs>` and return the temp name
    fn call(&mut self, ty: &str, rhs: &str) -> String {
        let t = self.tmp();
        let _ = writeln!(self.body, "  {} = call {} {}", t, ty, rhs);
        t
    }
    fn release(&mut self, v: &str) {
        let _ = writeln!(self.body, "  call void @nvb_rel({} {})", NV, v);
    }
    fn release_slot(&mut self, slot: &str) {
        let old = self.tmp();
        let _ = writeln!(self.body, "  {} = load {}, ptr {}", old, NV, slot);
        self.release(&old);
    }
    // store an owned value into a slot, releasing the previous occupant
    fn store_slot(&mut self, slot: &str, v: &str) {
        self.release_slot(slot);
        let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, v, slot);
    }
    fn truthy(&mut self, v: &str) -> String {
        let i = self.call("i64", &format!("@xtruthy({} {})", NV, v));
        let c = self.tmp();
        let _ = writeln!(self.body, "  {} = icmp ne i64 {}, 0", c, i);
        c
    }

    fn stmt(&mut self, s: &Stmt, all_vars: &[String]) {
        match s {
            Stmt::Expr(e) => {
                let mut inner = e;
                while let Expr::At { expr, .. } = inner { inner = expr; }
                let v = self.expr(inner);
                self.release(&v);
            }
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let v = self.expr(value);
                let slot = self.slots[name].clone();
                self.store_slot(&slot, &v);
            }
            Stmt::IndexAssign { base, index, value } => {
                let bexp = self.expr(base);
                let i = self.expr(index);
                let v = self.expr(value);
                let _ = writeln!(self.body, "  call void @xidxset({} {}, {} {}, {} {})",
                    NV, bexp, NV, i, NV, v);
            }
            Stmt::Return(Some(e)) => {
                let v = self.expr(e);
                // stash the return value, release locals, then return it
                let slot = self.tmp();
                let _ = writeln!(self.body, "  {} = alloca {}", slot, NV);
                let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, v, slot);
                for x in all_vars { let sl = self.slots[x].clone(); self.release_slot(&sl); }
                let r = self.tmp();
                let _ = writeln!(self.body, "  {} = load {}, ptr {}", r, NV, slot);
                let _ = writeln!(self.body, "  ret {} {}", NV, r);
                let dead = self.lbl("dead");
                let _ = writeln!(self.body, "{}:", dead);
            }
            Stmt::Return(None) => {
                for x in all_vars { let sl = self.slots[x].clone(); self.release_slot(&sl); }
                let z = self.call(NV, "@nv_null()");
                let _ = writeln!(self.body, "  ret {} {}", NV, z);
                let dead = self.lbl("dead");
                let _ = writeln!(self.body, "{}:", dead);
            }
            Stmt::If { cond, then, els } => {
                let c = self.expr(cond);
                let cc = self.truthy(&c);
                self.release(&c);
                let lt = self.lbl("then");
                let le = self.lbl("else");
                let ld = self.lbl("endif");
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", cc, lt, le);
                let _ = writeln!(self.body, "{}:", lt);
                for s in then { self.stmt(s, all_vars); }
                let _ = writeln!(self.body, "  br label %{}", ld);
                let _ = writeln!(self.body, "{}:", le);
                if let Some(els) = els { for s in els { self.stmt(s, all_vars); } }
                let _ = writeln!(self.body, "  br label %{}", ld);
                let _ = writeln!(self.body, "{}:", ld);
            }
            Stmt::While { cond, body } => {
                let lc = self.lbl("wcond");
                let lb = self.lbl("wbody");
                let le = self.lbl("wend");
                let _ = writeln!(self.body, "  br label %{}", lc);
                let _ = writeln!(self.body, "{}:", lc);
                let c = self.expr(cond);
                let cc = self.truthy(&c);
                self.release(&c);
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", cc, lb, le);
                let _ = writeln!(self.body, "{}:", lb);
                self.loops.push((lc.clone(), le.clone()));
                for s in body { self.stmt(s, all_vars); }
                self.loops.pop();
                let _ = writeln!(self.body, "  br label %{}", lc);
                let _ = writeln!(self.body, "{}:", le);
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                let sv = self.expr(start);
                let lo = self.call("i64", &format!("@xint({} {})", NV, sv));
                let ev = self.expr(end);
                let hi = self.call("i64", &format!("@xint({} {})", NV, ev));
                let ctr = self.tmp();
                let _ = writeln!(self.body, "  {} = alloca i64", ctr);
                let _ = writeln!(self.body, "  store i64 {}, ptr {}", lo, ctr);
                let lc = self.lbl("rcond");
                let lb = self.lbl("rbody");
                let le = self.lbl("rend");
                let cmp = if *inclusive { "sle" } else { "slt" };
                let slot = self.slots[var].clone();
                let _ = writeln!(self.body, "  br label %{}", lc);
                let _ = writeln!(self.body, "{}:", lc);
                let cv = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", cv, ctr);
                let cc = self.tmp();
                let _ = writeln!(self.body, "  {} = icmp {} i64 {}, {}", cc, cmp, cv, hi);
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", cc, lb, le);
                let _ = writeln!(self.body, "{}:", lb);
                let iv = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", iv, ctr);
                let boxed = self.call(NV, &format!("@nv_int(i64 {})", iv));
                self.store_slot(&slot, &boxed);
                self.loops.push((lc.clone(), le.clone()));
                for s in body { self.stmt(s, all_vars); }
                self.loops.pop();
                let nv = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", nv, ctr);
                let ni = self.tmp();
                let _ = writeln!(self.body, "  {} = add i64 {}, 1", ni, nv);
                let _ = writeln!(self.body, "  store i64 {}, ptr {}", ni, ctr);
                let _ = writeln!(self.body, "  br label %{}", lc);
                let _ = writeln!(self.body, "{}:", le);
            }
            Stmt::ForEach { var, iter, body } => {
                // snapshot arrays (like the interpreter), then index by position
                let it0 = self.expr(iter);
                let tag = self.tmp();
                let _ = writeln!(self.body, "  {} = extractvalue {} {}, 0", tag, NV, it0);
                let isarr = self.tmp();
                let _ = writeln!(self.body, "  {} = icmp eq i8 {}, 5", isarr, tag); // NV_ARR
                let itslot = self.tmp();
                let _ = writeln!(self.body, "  {} = alloca {}", itslot, NV);
                let ls = self.lbl("fslice");
                let ln = self.lbl("fnoslice");
                let la = self.lbl("fafter");
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", isarr, ls, ln);
                let _ = writeln!(self.body, "{}:", ls);
                let z0 = self.call(NV, "@nv_int(i64 0)");
                let z1 = self.call(NV, "@nv_int(i64 0)");
                let snap = self.call(NV, &format!("@xslice({0} {1}, {0} {2}, i32 0, {0} {3}, i32 0, i32 0)",
                    NV, it0, z0, z1));
                let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, snap, itslot);
                let _ = writeln!(self.body, "  br label %{}", la);
                let _ = writeln!(self.body, "{}:", ln);
                let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, it0, itslot);
                let _ = writeln!(self.body, "  br label %{}", la);
                let _ = writeln!(self.body, "{}:", la);
                let it = self.tmp();
                let _ = writeln!(self.body, "  {} = load {}, ptr {}", it, NV, itslot);
                let n = self.call("i64", &format!("@nvb_len({} {})", NV, it));
                let idx = self.tmp();
                let _ = writeln!(self.body, "  {} = alloca i64", idx);
                let _ = writeln!(self.body, "  store i64 0, ptr {}", idx);
                let lc = self.lbl("fcond");
                let lb = self.lbl("fbody");
                let le = self.lbl("fend");
                let slot = self.slots[var].clone();
                let _ = writeln!(self.body, "  br label %{}", lc);
                let _ = writeln!(self.body, "{}:", lc);
                let iv = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", iv, idx);
                let cc = self.tmp();
                let _ = writeln!(self.body, "  {} = icmp slt i64 {}, {}", cc, iv, n);
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", cc, lb, le);
                let _ = writeln!(self.body, "{}:", lb);
                let iv2 = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", iv2, idx);
                let ib = self.call(NV, &format!("@nv_int(i64 {})", iv2));
                let elem = self.call(NV, &format!("@nvb_index({0} {1}, {0} {2})", NV, it, ib));
                self.store_slot(&slot, &elem);
                self.loops.push((lc.clone(), le.clone()));
                for s in body { self.stmt(s, all_vars); }
                self.loops.pop();
                let iv3 = self.tmp();
                let _ = writeln!(self.body, "  {} = load i64, ptr {}", iv3, idx);
                let ni = self.tmp();
                let _ = writeln!(self.body, "  {} = add i64 {}, 1", ni, iv3);
                let _ = writeln!(self.body, "  store i64 {}, ptr {}", ni, idx);
                let _ = writeln!(self.body, "  br label %{}", lc);
                let _ = writeln!(self.body, "{}:", le);
                self.release(&it);
            }
            Stmt::Break(None) => {
                if let Some((_, le)) = self.loops.last().cloned() {
                    let _ = writeln!(self.body, "  br label %{}", le);
                    let d = self.lbl("dead"); let _ = writeln!(self.body, "{}:", d);
                }
            }
            Stmt::Continue => {
                if let Some((lc, _)) = self.loops.last().cloned() {
                    let _ = writeln!(self.body, "  br label %{}", lc);
                    let d = self.lbl("dead"); let _ = writeln!(self.body, "{}:", d);
                }
            }
            _ => unreachable!("checked by analyze_boxed"),
        }
    }

    // returns an SSA temp holding an owned {i8,i64}
    fn expr(&mut self, e: &Expr) -> String {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Int(n) => self.call(NV, &format!("@nv_int(i64 {})", n)),
            Expr::Float(x) => self.call(NV, &format!("@nv_float(double {})", ll_float(*x))),
            Expr::Bool(b) => self.call(NV, &format!("@nv_bool(i64 {})", if *b { 1 } else { 0 })),
            Expr::Null => self.call(NV, "@nv_null()"),
            Expr::Str(s) => {
                let idx = self.strs.len();
                self.strs.push(s.clone());
                let p = self.tmp();
                let _ = writeln!(self.body,
                    "  {} = getelementptr inbounds [{} x i8], ptr @.sb{}, i64 0, i64 0",
                    p, s.len() + 1, idx);
                self.call(NV, &format!("@nv_str(ptr {})", p))
            }
            Expr::Ident(n) => {
                let slot = self.slots[n].clone();
                let v = self.tmp();
                let _ = writeln!(self.body, "  {} = load {}, ptr {}", v, NV, slot);
                self.call(NV, &format!("@xret({} {})", NV, v))
            }
            Expr::Unary { op, expr } => {
                let v = self.expr(expr);
                let f = match op { UnOp::Neg => "xneg", UnOp::Not => "xnot", UnOp::BitNot => "xbnot" };
                self.call(NV, &format!("@{}({} {})", f, NV, v))
            }
            Expr::Binary { op, lhs, rhs } => {
                if matches!(op, BinOp::And | BinOp::Or) {
                    return self.short_circuit(*op, lhs, rhs);
                }
                let a = self.expr(lhs);
                let b = self.expr(rhs);
                let (f, extra) = match op {
                    BinOp::Add => ("xadd", ""), BinOp::Sub => ("xsub", ""),
                    BinOp::Mul => ("xmul", ""), BinOp::Div => ("xdiv", ""),
                    BinOp::Rem => ("xrem", ""), BinOp::Lt => ("xlt", ""),
                    BinOp::Le => ("xle", ""), BinOp::Gt => ("xgt", ""),
                    BinOp::Ge => ("xge", ""), BinOp::Eq => ("xeq", ""),
                    BinOp::Ne => ("xne", ""),
                    BinOp::BitAnd => ("xbit", ", i8 38"), BinOp::BitOr => ("xbit", ", i8 124"),
                    BinOp::BitXor => ("xbit", ", i8 94"), BinOp::Shl => ("xbit", ", i8 60"),
                    BinOp::Shr => ("xbit", ", i8 62"),
                    BinOp::Pow | BinOp::And | BinOp::Or => unreachable!(),
                };
                self.call(NV, &format!("@{}({} {}, {} {}{})", f, NV, a, NV, b, extra))
            }
            Expr::If { cond, then, els } => {
                let slot = self.tmp();
                let _ = writeln!(self.body, "  {} = alloca {}", slot, NV);
                let c = self.expr(cond);
                let cc = self.truthy(&c);
                self.release(&c);
                let lt = self.lbl("ithen");
                let le = self.lbl("ielse");
                let ld = self.lbl("iend");
                let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", cc, lt, le);
                let _ = writeln!(self.body, "{}:", lt);
                let tv = self.expr(then);
                let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, tv, slot);
                let _ = writeln!(self.body, "  br label %{}", ld);
                let _ = writeln!(self.body, "{}:", le);
                let ev = self.expr(els);
                let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, ev, slot);
                let _ = writeln!(self.body, "  br label %{}", ld);
                let _ = writeln!(self.body, "{}:", ld);
                let r = self.tmp();
                let _ = writeln!(self.body, "  {} = load {}, ptr {}", r, NV, slot);
                r
            }
            Expr::Array(xs) => {
                let a = self.call(NV, &format!("@nv_arr(i64 {})", xs.len().max(1)));
                for x in xs {
                    let v = self.expr(x);
                    // non-releasing push: `a` is the array we are building and return
                    let _ = writeln!(self.body, "  call void @nvb_push({} {}, {} {})", NV, a, NV, v);
                }
                a
            }
            Expr::Index { base, index } => {
                if let Expr::RangeLit { lo, hi, inclusive } = &**index {
                    let bexp = self.expr(base);
                    let lo_v = match lo.as_ref() {
                        Some(e) => self.expr(e), None => self.call(NV, "@nv_int(i64 0)"),
                    };
                    let hi_v = match hi.as_ref() {
                        Some(e) => self.expr(e), None => self.call(NV, "@nv_int(i64 0)"),
                    };
                    self.call(NV, &format!("@xslice({0} {1}, {0} {2}, i32 {3}, {0} {4}, i32 {5}, i32 {6})",
                        NV, bexp, lo_v, lo.is_some() as u8, hi_v, hi.is_some() as u8, *inclusive as u8))
                } else {
                    let bexp = self.expr(base);
                    let i = self.expr(index);
                    self.call(NV, &format!("@xindex({0} {1}, {0} {2})", NV, bexp, i))
                }
            }
            Expr::RangeLit { lo: Some(lo), hi: Some(hi), inclusive } => {
                let l = self.expr(lo);
                let h = self.expr(hi);
                self.call(NV, &format!("@xrange({0} {1}, {0} {2}, i32 {3})", NV, l, h, *inclusive as u8))
            }
            Expr::FmtStr(parts) => {
                let mut acc = self.call(NV, "@nv_str(ptr @.sbempty)");
                for p in parts {
                    let piece = match p {
                        FmtPart::Lit(t) => {
                            let idx = self.strs.len();
                            self.strs.push(t.clone());
                            let pp = self.tmp();
                            let _ = writeln!(self.body,
                                "  {} = getelementptr inbounds [{} x i8], ptr @.sb{}, i64 0, i64 0",
                                pp, t.len() + 1, idx);
                            self.call(NV, &format!("@nv_str(ptr {})", pp))
                        }
                        FmtPart::Expr(e) => {
                            let v = self.expr(e);
                            self.call(NV, &format!("@xtostr({} {})", NV, v))
                        }
                    };
                    acc = self.call(NV, &format!("@xcat({0} {1}, {0} {2})", NV, acc, piece));
                }
                acc
            }
            Expr::Call { callee, args } => self.call_expr(callee, args),
            Expr::Block { stmts, tail } if stmts.is_empty() => match tail {
                Some(t) => self.expr(t),
                None => self.call(NV, "@nv_null()"),
            },
            _ => unreachable!("checked by analyze_boxed"),
        }
    }

    fn call_expr(&mut self, callee: &str, args: &[Expr]) -> String {
        match callee {
            "len" => {
                let a = self.expr(&args[0]);
                self.call(NV, &format!("@xlen({} {})", NV, a))
            }
            "pop" => {
                let a = self.expr(&args[0]);
                self.call(NV, &format!("@xpop({} {})", NV, a))
            }
            "push" => {
                let a = self.expr(&args[0]);
                let v = self.expr(&args[1]);
                let _ = writeln!(self.body, "  call void @xpush({} {}, {} {})", NV, a, NV, v);
                self.call(NV, "@nv_null()")
            }
            "print" => {
                let v = self.expr(&args[0]);
                let _ = writeln!(self.body, "  call void @xprint({} {})", NV, v);
                self.call(NV, "@nv_null()")
            }
            _ if self.b.ints.contains(callee) => {
                let vals: Vec<String> = args.iter().map(|a| {
                    let v = self.expr(a);
                    self.call("i64", &format!("@xint({} {})", NV, v))
                }).collect();
                let ps: Vec<String> = vals.iter().map(|v| format!("i64 {}", v)).collect();
                let r = self.call("i64", &format!("@{}({})", mangle(callee), ps.join(", ")));
                self.call(NV, &format!("@nv_int(i64 {})", r))
            }
            _ if self.b.floats.contains(callee) => {
                let vals: Vec<String> = args.iter().map(|a| {
                    let v = self.expr(a);
                    self.call("double", &format!("@xflt({} {})", NV, v))
                }).collect();
                let ps: Vec<String> = vals.iter().map(|v| format!("double {}", v)).collect();
                let r = self.call("double", &format!("@{}({})", mangle(callee), ps.join(", ")));
                self.call(NV, &format!("@nv_float(double {})", r))
            }
            _ => {
                let vals: Vec<String> = args.iter().map(|a| self.expr(a)).collect();
                let ps: Vec<String> = vals.iter().map(|v| format!("{} {}", NV, v)).collect();
                self.call(NV, &format!("@{}({})", mangle(callee), ps.join(", ")))
            }
        }
    }

    fn short_circuit(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr) -> String {
        let slot = self.tmp();
        let _ = writeln!(self.body, "  {} = alloca {}", slot, NV);
        let a = self.expr(lhs);
        let ca = self.truthy(&a);
        self.release(&a);
        let lrhs = self.lbl("scrhs");
        let lshort = self.lbl("scshort");
        let ld = self.lbl("scend");
        // AND: if a truthy eval rhs else false. OR: if a truthy true else eval rhs.
        let (then_lbl, else_lbl) = if matches!(op, BinOp::And) { (&lrhs, &lshort) } else { (&lshort, &lrhs) };
        let _ = writeln!(self.body, "  br i1 {}, label %{}, label %{}", ca, then_lbl, else_lbl);
        let _ = writeln!(self.body, "{}:", lrhs);
        let b = self.expr(rhs);
        let cb = self.truthy(&b);
        self.release(&b);
        let bb = self.tmp();
        let _ = writeln!(self.body, "  {} = zext i1 {} to i64", bb, cb);
        let bv = self.call(NV, &format!("@nv_bool(i64 {})", bb));
        let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, bv, slot);
        let _ = writeln!(self.body, "  br label %{}", ld);
        let _ = writeln!(self.body, "{}:", lshort);
        let lit = if matches!(op, BinOp::And) { 0 } else { 1 };
        let sv = self.call(NV, &format!("@nv_bool(i64 {})", lit));
        let _ = writeln!(self.body, "  store {} {}, ptr {}", NV, sv, slot);
        let _ = writeln!(self.body, "  br label %{}", ld);
        let _ = writeln!(self.body, "{}:", ld);
        let r = self.tmp();
        let _ = writeln!(self.body, "  {} = load {}, ptr {}", r, NV, slot);
        r
    }
}

// Format an f64 as an LLVM `double` literal. LLVM accepts C99 hex float, which
// is exact and side-steps decimal round-trip issues.
fn ll_float(x: f64) -> String {
    if x.is_nan() { return "0x7FF8000000000000".into(); }
    if x.is_infinite() {
        return if x < 0.0 { "0xFFF0000000000000".into() } else { "0x7FF0000000000000".into() };
    }
    // LLVM's "0x" double form is the raw 64-bit IEEE pattern
    format!("0x{:016X}", x.to_bits())
}

// Fixed IR preamble: runtime declarations (clang's NV ABI) + the aggregate-in,
// aggregate-out wrapper functions the emitter targets.
fn ll_boxed_preamble() -> String {
    let mut s = String::new();
    s.push_str("; runtime declarations — NV passed as (i8 tag, i64 payload), returned as {i8,i64}\n");
    let agg = |name: &str, params: &str| format!("declare {{ i8, i64 }} @{}({})\n", name, params);
    for (n, p) in [
        ("nv_int", "i64"), ("nv_float", "double"), ("nv_bool", "i64"), ("nv_null", ""),
        ("nv_str", "ptr"), ("nv_arr", "i64"), ("nv_retain", "i8,i64"),
        ("nv_add", "i8,i64,i8,i64"), ("nv_sub", "i8,i64,i8,i64"), ("nv_mul", "i8,i64,i8,i64"),
        ("nv_div", "i8,i64,i8,i64"), ("nv_rem", "i8,i64,i8,i64"),
        ("nv_cmp_lt", "i8,i64,i8,i64"), ("nv_cmp_le", "i8,i64,i8,i64"),
        ("nv_cmp_gt", "i8,i64,i8,i64"), ("nv_cmp_ge", "i8,i64,i8,i64"),
        ("nv_concat2", "i8,i64,i8,i64"), ("nv_bit", "i8,i64,i8,i64,i8"),
        ("nv_neg", "i8,i64"), ("nv_not", "i8,i64"), ("nv_bitnot", "i8,i64"),
        ("nv_tostr", "i8,i64"), ("nv_index", "i8,i64,i8,i64"), ("nv_pop", "i8,i64"),
        ("nv_slice", "i8,i64,i64,i32,i64,i32,i32"), ("nv_range", "i64,i64,i32"),
    ] {
        s.push_str(&agg(n, p));
    }
    s.push_str("declare i64 @nv_eq(i8,i64,i8,i64)\n");
    s.push_str("declare i64 @nv_truthy(i8,i64)\n");
    s.push_str("declare i64 @nv_len(i8,i64)\n");
    s.push_str("declare i64 @nv_as_int(i8,i64)\n");
    s.push_str("declare double @nv_as_float(i8,i64)\n");
    s.push_str("declare void @nv_print(i8,i64)\n");
    s.push_str("declare void @nv_release(i8,i64)\n");
    s.push_str("declare void @nv_arr_push(i8,i64,i8,i64)\n");
    s.push_str("declare void @nv_index_set(i8,i64,i8,i64,i8,i64)\n\n");
    s.push_str("@.sbempty = private unnamed_addr constant [1 x i8] c\"\\00\"\n\n");
    // wrapper helpers: split the aggregate, call the runtime, manage ownership
    s.push_str(&ll_wrappers());
    s
}

// The `{i8,i64}`-in/out wrapper functions. Each unpacks its NV aggregate(s) into
// (tag,payload) for the runtime call and mirrors BxEmit's ownership discipline.
fn ll_wrappers() -> String {
    let mut s = String::new();
    // unpack %0 -> (%a0,%a1), %1 -> (%b0,%b1)
    let split_a = "  %a0 = extractvalue { i8, i64 } %0, 0\n  %a1 = extractvalue { i8, i64 } %0, 1\n";
    let split_b = "  %b0 = extractvalue { i8, i64 } %1, 0\n  %b1 = extractvalue { i8, i64 } %1, 1\n";
    let rel_a = "  call void @nv_release(i8 %a0, i64 %a1)\n";
    let rel_b = "  call void @nv_release(i8 %b0, i64 %b1)\n";

    // binary, consume both, return NV
    for (x, nv) in [("xadd","nv_add"),("xsub","nv_sub"),("xmul","nv_mul"),("xdiv","nv_div"),
                    ("xrem","nv_rem"),("xlt","nv_cmp_lt"),("xle","nv_cmp_le"),("xgt","nv_cmp_gt"),
                    ("xge","nv_cmp_ge"),("xcat","nv_concat2")] {
        let _ = write!(s, "define internal {{ i8, i64 }} @{}({{ i8, i64 }} %0, {{ i8, i64 }} %1) {{\n{}{}  %r = call {{ i8, i64 }} @{}(i8 %a0, i64 %a1, i8 %b0, i64 %b1)\n{}{}  ret {{ i8, i64 }} %r\n}}\n",
            x, split_a, split_b, nv, rel_a, rel_b);
    }
    // equality: nv_eq -> i64 -> nv_bool
    for (x, invert) in [("xeq", false), ("xne", true)] {
        let cmp = if invert { "  %b = icmp eq i64 %e, 0\n  %z = zext i1 %b to i64\n" }
                  else       { "  %z = icmp ne i64 %e, 0\n  %z2 = zext i1 %z to i64\n" };
        let zv = if invert { "%z" } else { "%z2" };
        let _ = write!(s, "define internal {{ i8, i64 }} @{}({{ i8, i64 }} %0, {{ i8, i64 }} %1) {{\n{}{}  %e = call i64 @nv_eq(i8 %a0, i64 %a1, i8 %b0, i64 %b1)\n{}  %r = call {{ i8, i64 }} @nv_bool(i64 {})\n{}{}  ret {{ i8, i64 }} %r\n}}\n",
            x, split_a, split_b, cmp, zv, rel_a, rel_b);
    }
    // bitwise: extra i8 op arg, consume both
    let _ = write!(s, "define internal {{ i8, i64 }} @xbit({{ i8, i64 }} %0, {{ i8, i64 }} %1, i8 %2) {{\n{}{}  %r = call {{ i8, i64 }} @nv_bit(i8 %a0, i64 %a1, i8 %b0, i64 %b1, i8 %2)\n{}{}  ret {{ i8, i64 }} %r\n}}\n",
        split_a, split_b, rel_a, rel_b);
    // unary, consume, return NV
    for (x, nv) in [("xneg","nv_neg"),("xnot","nv_not"),("xbnot","nv_bitnot"),("xtostr","nv_tostr")] {
        let _ = write!(s, "define internal {{ i8, i64 }} @{}({{ i8, i64 }} %0) {{\n{}  %r = call {{ i8, i64 }} @{}(i8 %a0, i64 %a1)\n{}  ret {{ i8, i64 }} %r\n}}\n",
            x, split_a, nv, rel_a);
    }
    // truthy/xint/xflt: consume, return scalar
    let _ = write!(s, "define internal i64 @xtruthy({{ i8, i64 }} %0) {{\n{}  %r = call i64 @nv_truthy(i8 %a0, i64 %a1)\n{}  ret i64 %r\n}}\n", split_a, rel_a);
    let _ = write!(s, "define internal i64 @xint({{ i8, i64 }} %0) {{\n{}  %r = call i64 @nv_as_int(i8 %a0, i64 %a1)\n{}  ret i64 %r\n}}\n", split_a, rel_a);
    let _ = write!(s, "define internal double @xflt({{ i8, i64 }} %0) {{\n{}  %r = call double @nv_as_float(i8 %a0, i64 %a1)\n{}  ret double %r\n}}\n", split_a, rel_a);
    // xlen: len -> nv_int, consume
    let _ = write!(s, "define internal {{ i8, i64 }} @xlen({{ i8, i64 }} %0) {{\n{}  %n = call i64 @nv_len(i8 %a0, i64 %a1)\n{}  %r = call {{ i8, i64 }} @nv_int(i64 %n)\n  ret {{ i8, i64 }} %r\n}}\n", split_a, rel_a);
    // xindex: consume both
    let _ = write!(s, "define internal {{ i8, i64 }} @xindex({{ i8, i64 }} %0, {{ i8, i64 }} %1) {{\n{}{}  %r = call {{ i8, i64 }} @nv_index(i8 %a0, i64 %a1, i8 %b0, i64 %b1)\n{}{}  ret {{ i8, i64 }} %r\n}}\n", split_a, split_b, rel_a, rel_b);
    // nvb_index: borrow (no release) — for-each stepping
    let _ = write!(s, "define internal {{ i8, i64 }} @nvb_index({{ i8, i64 }} %0, {{ i8, i64 }} %1) {{\n{}{}  %r = call {{ i8, i64 }} @nv_index(i8 %a0, i64 %a1, i8 %b0, i64 %b1)\n  ret {{ i8, i64 }} %r\n}}\n", split_a, split_b);
    // nvb_len: borrow
    let _ = write!(s, "define internal i64 @nvb_len({{ i8, i64 }} %0) {{\n{}  %r = call i64 @nv_len(i8 %a0, i64 %a1)\n  ret i64 %r\n}}\n", split_a);
    // xret: retain (no release)
    let _ = write!(s, "define internal {{ i8, i64 }} @xret({{ i8, i64 }} %0) {{\n{}  %r = call {{ i8, i64 }} @nv_retain(i8 %a0, i64 %a1)\n  ret {{ i8, i64 }} %r\n}}\n", split_a);
    // nvb_rel: release
    let _ = write!(s, "define internal void @nvb_rel({{ i8, i64 }} %0) {{\n{}  call void @nv_release(i8 %a0, i64 %a1)\n  ret void\n}}\n", split_a);
    // xprint: consume
    let _ = write!(s, "define internal void @xprint({{ i8, i64 }} %0) {{\n{}  call void @nv_print(i8 %a0, i64 %a1)\n{}  ret void\n}}\n", split_a, rel_a);
    // xpush(a, v): push v into a, then release a (a is a retained handle from a
    // `push(arr, x)` call); v moves into the array
    let _ = write!(s, "define internal void @xpush({{ i8, i64 }} %0, {{ i8, i64 }} %1) {{\n{}{}  call void @nv_arr_push(i8 %a0, i64 %a1, i8 %b0, i64 %b1)\n{}  ret void\n}}\n", split_a, split_b, rel_a);
    // nvb_push(a, v): push v into a WITHOUT releasing a — for building an array
    // literal, where a is the array under construction (returned owned); v moves in
    let _ = write!(s, "define internal void @nvb_push({{ i8, i64 }} %0, {{ i8, i64 }} %1) {{\n{}{}  call void @nv_arr_push(i8 %a0, i64 %a1, i8 %b0, i64 %b1)\n  ret void\n}}\n", split_a, split_b);
    // xidxset(b, i, v): set, release b and i; v moves in
    let _ = write!(s, "define internal void @xidxset({{ i8, i64 }} %0, {{ i8, i64 }} %1, {{ i8, i64 }} %2) {{\n{}{}  %c0 = extractvalue {{ i8, i64 }} %2, 0\n  %c1 = extractvalue {{ i8, i64 }} %2, 1\n  call void @nv_index_set(i8 %a0, i64 %a1, i8 %b0, i64 %b1, i8 %c0, i64 %c1)\n{}{}  ret void\n}}\n", split_a, split_b, rel_a, rel_b);
    // xpop(a): pop, release a, return popped
    let _ = write!(s, "define internal {{ i8, i64 }} @xpop({{ i8, i64 }} %0) {{\n{}  %r = call {{ i8, i64 }} @nv_pop(i8 %a0, i64 %a1)\n{}  ret {{ i8, i64 }} %r\n}}\n", split_a, rel_a);
    // xslice(b, lo, hl, hi, hh, inc): release b only
    let _ = write!(s, "define internal {{ i8, i64 }} @xslice({{ i8, i64 }} %0, {{ i8, i64 }} %1, i32 %2, {{ i8, i64 }} %3, i32 %4, i32 %5) {{\n{}  %lo = call i64 @nv_as_int(i8 %b0, i64 %b1)\n  %h0 = extractvalue {{ i8, i64 }} %3, 0\n  %h1 = extractvalue {{ i8, i64 }} %3, 1\n  %hi = call i64 @nv_as_int(i8 %h0, i64 %h1)\n  %r = call {{ i8, i64 }} @nv_slice(i8 %a0, i64 %a1, i64 %lo, i32 %2, i64 %hi, i32 %4, i32 %5)\n{}  ret {{ i8, i64 }} %r\n}}\n", format!("{}{}", split_a, split_b), rel_a);
    // xrange(lo, hi, inc): no release (int bounds)
    let _ = write!(s, "define internal {{ i8, i64 }} @xrange({{ i8, i64 }} %0, {{ i8, i64 }} %1, i32 %2) {{\n{}{}  %lo = call i64 @nv_as_int(i8 %a0, i64 %a1)\n  %hi = call i64 @nv_as_int(i8 %b0, i64 %b1)\n  %r = call {{ i8, i64 }} @nv_range(i64 %lo, i64 %hi, i32 %2)\n  ret {{ i8, i64 }} %r\n}}\n", split_a, split_b);
    s.push('\n');
    s
}

