// Nova JIT — a Cranelift native-code compiler for the integer compute core.
//
// Functions that provably compute and return i64 integers, with no side effects
// and no non-integer values, are JIT-compiled. Calls to other eligible functions
// are compiled as native calls (so recursion runs entirely in machine code); the
// eligible set is a fixpoint over the call graph. Every operation that could
// leave the integer world — arithmetic overflow (Nova promotes to BigInt there),
// division/modulo by zero, `**`, out-of-range shifts, negating i64::MIN, or a
// callee that deopts — branches to a single *deopt* path that sets a shared flag
// and returns. The bytecode VM then re-runs the whole top-level call, which is
// observationally identical because eligible functions are pure. So the JIT can
// only ever be faster, never wrong; the VM/interpreter stays the oracle.

use std::collections::{HashMap, HashSet};
use cranelift::prelude::*;
use cranelift::prelude::types::{I64, I128};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, Linkage, FuncId};
use crate::ast::*;

// functions with more parameters than this stay on the VM (keeps the call ABI
// dispatch in `raw_call` finite)
const MAX_ARITY: usize = 8;

// ---------------------------------------------------------------------------
// Eligibility (a fixpoint over the call graph)
// ---------------------------------------------------------------------------

// The set of function names that can be JIT-compiled: each is structurally
// integer-pure, returns an integer, has arity <= MAX_ARITY, and every function
// it calls is itself eligible.
pub fn eligible_set(prog: &Program) -> HashSet<String> {
    let mut funcs: HashMap<&str, &Func> = HashMap::new();
    for item in &prog.items {
        if let Item::Func(f) = item { funcs.insert(&f.name, f); }
    }
    // start from everything structurally OK (calls allowed to anything for now)
    let mut set: HashSet<String> = funcs.values()
        .filter(|f| f.params.len() <= MAX_ARITY && locally_ok(f))
        .map(|f| f.name.clone())
        .collect();
    // remove any function that calls a name outside the set, until stable
    loop {
        let mut remove = None;
        for name in &set {
            let f = funcs[name.as_str()];
            if collect_calls(&f.body).iter().any(|c| !set.contains(c)) {
                remove = Some(name.clone());
                break;
            }
        }
        match remove {
            Some(n) => { set.remove(&n); }
            None => break,
        }
    }
    set
}

// structural integer-purity, allowing calls (validated by the fixpoint)
fn locally_ok(f: &Func) -> bool {
    !f.body.is_empty()
        && f.body.iter().all(stmt_pure)
        && always_returns(&f.body)
}

// the body always yields an integer value (never falls through to an implicit
// null): it ends in a value return, or in an if/else whose branches both do.
fn always_returns(body: &[Stmt]) -> bool {
    match body.last() {
        Some(Stmt::Return(Some(_))) => true,
        Some(Stmt::If { then, els: Some(els), .. }) =>
            always_returns(then) && always_returns(els),
        _ => false,
    }
}

fn stmt_pure(s: &Stmt) -> bool {
    match s {
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } => expr_pure(value),
        Stmt::Expr(e) => expr_pure(e),
        Stmt::Return(Some(e)) => expr_pure(e),
        Stmt::Return(None) => false,
        Stmt::If { cond, then, els } =>
            expr_pure(cond) && then.iter().all(stmt_pure)
            && els.as_ref().map_or(true, |e| e.iter().all(stmt_pure)),
        Stmt::While { cond, body } => expr_pure(cond) && body.iter().all(stmt_pure),
        Stmt::ForRange { start, end, body, .. } =>
            expr_pure(start) && expr_pure(end) && body.iter().all(stmt_pure),
        Stmt::Break(None) | Stmt::Continue => true,
        _ => false,
    }
}

fn expr_pure(e: &Expr) -> bool {
    match e {
        Expr::At { expr, .. } => expr_pure(expr),
        Expr::Int(_) | Expr::Ident(_) => true,
        Expr::Unary { op, expr } =>
            matches!(op, UnOp::Neg | UnOp::Not | UnOp::BitNot) && expr_pure(expr),
        Expr::Binary { op, lhs, rhs } => binop_pure(*op) && expr_pure(lhs) && expr_pure(rhs),
        Expr::If { cond, then, els } => expr_pure(cond) && expr_pure(then) && expr_pure(els),
        Expr::Block { stmts, tail } =>
            stmts.iter().all(stmt_pure) && tail.as_ref().map_or(false, |t| expr_pure(t)),
        // a call with <= MAX_ARITY integer args; callee eligibility is enforced
        // by the fixpoint via `collect_calls`
        Expr::Call { args, .. } => args.len() <= MAX_ARITY && args.iter().all(expr_pure),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Float (f64) eligibility — a second, disjoint track (Phase 5C.2). A function
// qualifies iff everything is f64-only: Float literals, float arithmetic
// (+ - * /; `%`/`**` have no Cranelift instruction and stay on the VM),
// comparisons/logic only in condition position, calls only to float-eligible
// functions. Floats never deopt: /0.0 -> inf and NaN behave identically to the
// interpreter's `as_f(l) op as_f(r)` arms, and there is no BigInt promotion.
// ---------------------------------------------------------------------------

pub fn float_eligible_set(prog: &Program, int_set: &HashSet<String>) -> HashSet<String> {
    let mut funcs: HashMap<&str, &Func> = HashMap::new();
    for item in &prog.items {
        if let Item::Func(f) = item { funcs.insert(&f.name, f); }
    }
    let mut set: HashSet<String> = funcs.values()
        .filter(|f| f.params.len() <= MAX_ARITY
            && !int_set.contains(&f.name)         // int track takes precedence
            && !f.body.is_empty()
            && f.body.iter().all(f_stmt_pure)
            && always_returns(&f.body))
        .map(|f| f.name.clone())
        .collect();
    loop {
        let mut remove = None;
        for name in &set {
            let f = funcs[name.as_str()];
            if collect_calls(&f.body).iter().any(|c| !set.contains(c)) {
                remove = Some(name.clone());
                break;
            }
        }
        match remove { Some(n) => { set.remove(&n); } None => break }
    }
    set
}

fn f_stmt_pure(s: &Stmt) -> bool {
    match s {
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } => f_expr(value),
        Stmt::Expr(e) => f_expr(e),
        Stmt::Return(Some(e)) => f_expr(e),
        Stmt::If { cond, then, els } =>
            f_cond(cond) && then.iter().all(f_stmt_pure)
            && els.as_ref().map_or(true, |e| e.iter().all(f_stmt_pure)),
        Stmt::While { cond, body } => f_cond(cond) && body.iter().all(f_stmt_pure),
        Stmt::Break(None) | Stmt::Continue => true,
        _ => false, // ForRange is integer-based; everything else as in the i64 track
    }
}

fn f_expr(e: &Expr) -> bool {
    match e {
        Expr::At { expr, .. } => f_expr(expr),
        Expr::Float(_) | Expr::Ident(_) => true,
        Expr::Unary { op: UnOp::Neg, expr } => f_expr(expr),
        Expr::Binary { op, lhs, rhs } =>
            matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
            && f_expr(lhs) && f_expr(rhs),
        Expr::If { cond, then, els } => f_cond(cond) && f_expr(then) && f_expr(els),
        Expr::Block { stmts, tail } =>
            stmts.iter().all(f_stmt_pure) && tail.as_ref().map_or(false, |t| f_expr(t)),
        Expr::Call { args, .. } => args.len() <= MAX_ARITY && args.iter().all(f_expr),
        _ => false,
    }
}

// boolean condition over floats: comparisons of float exprs, combined with && || !
fn f_cond(e: &Expr) -> bool {
    match e {
        Expr::At { expr, .. } => f_cond(expr),
        Expr::Unary { op: UnOp::Not, expr } => f_cond(expr),
        Expr::Binary { op: BinOp::And | BinOp::Or, lhs, rhs } => f_cond(lhs) && f_cond(rhs),
        Expr::Binary { op, lhs, rhs } =>
            matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
            && f_expr(lhs) && f_expr(rhs),
        _ => false,
    }
}

fn binop_pure(op: BinOp) -> bool {
    use BinOp::*;
    matches!(op, Add | Sub | Mul | Div | Rem | Pow
        | Eq | Ne | Lt | Le | Gt | Ge | And | Or
        | BitOr | BitXor | BitAnd | Shl | Shr)
}

// every function name called anywhere in a body
fn collect_calls(body: &[Stmt]) -> Vec<String> {
    let mut out = Vec::new();
    for s in body { calls_stmt(s, &mut out); }
    out
}
fn calls_stmt(s: &Stmt, out: &mut Vec<String>) {
    match s {
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Expr(value)
        | Stmt::Return(Some(value)) => calls_expr(value, out),
        Stmt::If { cond, then, els } => {
            calls_expr(cond, out);
            for s in then { calls_stmt(s, out); }
            if let Some(e) = els { for s in e { calls_stmt(s, out); } }
        }
        Stmt::While { cond, body } => { calls_expr(cond, out); for s in body { calls_stmt(s, out); } }
        Stmt::ForRange { start, end, body, .. } => {
            calls_expr(start, out); calls_expr(end, out);
            for s in body { calls_stmt(s, out); }
        }
        _ => {}
    }
}
fn calls_expr(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::At { expr, .. } | Expr::Unary { expr, .. } => calls_expr(expr, out),
        Expr::Binary { lhs, rhs, .. } => { calls_expr(lhs, out); calls_expr(rhs, out); }
        Expr::If { cond, then, els } => { calls_expr(cond, out); calls_expr(then, out); calls_expr(els, out); }
        Expr::Block { stmts, tail } => {
            for s in stmts { calls_stmt(s, out); }
            if let Some(t) = tail { calls_expr(t, out); }
        }
        Expr::Call { callee, args } => {
            out.push(callee.clone());
            for a in args { calls_expr(a, out); }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// JIT module
// ---------------------------------------------------------------------------

pub struct Jit {
    _module: JITModule,
    // name -> (entry pointer, arity); i64 track
    code: HashMap<String, (*const u8, usize)>,
    // name -> (entry pointer, arity); f64 track (disjoint names)
    fcode: HashMap<String, (*const u8, usize)>,
    // human-readable Cranelift IR per function, for `nova jit --dump`
    ir: HashMap<String, String>,
}

impl Jit {
    pub fn compile(prog: &Program) -> Option<Jit> {
        Self::compile_filtered(prog, None)
    }

    // Compile only the functions named in `only` (which must be closed under
    // calls within the eligible set — a callee closure — so every direct call
    // target is available). `None` compiles the whole eligible set.
    pub fn compile_filtered(prog: &Program, only: Option<&HashSet<String>>) -> Option<Jit> {
        let mut eligible = eligible_set(prog);
        let mut feligible = float_eligible_set(prog, &eligible_set(prog));
        if let Some(filter) = only {
            eligible.retain(|n| filter.contains(n));
            feligible.retain(|n| filter.contains(n));
        }
        if eligible.is_empty() && feligible.is_empty() { return None; }
        let mut funcs: HashMap<&str, &Func> = HashMap::new();
        for item in &prog.items {
            if let Item::Func(f) = item { funcs.insert(&f.name, f); }
        }

        let mut flags = settings::builder();
        flags.set("opt_level", "speed").ok()?;
        let isa = cranelift_native::builder().ok()?
            .finish(settings::Flags::new(flags)).ok()?;
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        // 1) declare every eligible function so calls can reference them
        let mut names: Vec<&str> = eligible.iter().map(|s| s.as_str()).collect();
        names.sort();
        let mut ids: HashMap<String, (FuncId, usize)> = HashMap::new();
        for name in &names {
            let f = funcs[*name];
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(I64)); // deopt flag pointer
            for _ in 0..f.params.len() { sig.params.push(AbiParam::new(I64)); }
            sig.returns.push(AbiParam::new(I64));
            let id = module.declare_function(name, Linkage::Export, &sig).ok()?;
            ids.insert(name.to_string(), (id, f.params.len()));
        }

        // 2) define each function body
        let mut ctx = module.make_context();
        let mut fctx = FunctionBuilderContext::new();
        let mut ir = HashMap::new();
        for name in &names {
            let f = funcs[*name];
            let (id, _) = ids[*name];
            ctx.func.signature.params.push(AbiParam::new(I64));
            for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(I64)); }
            ctx.func.signature.returns.push(AbiParam::new(I64));
            {
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
                let mut g = FnGen::new(&mut b, &mut module, &ids, &f.params);
                g.lower(&f.body)?;
                b.finalize();
            }
            ir.insert(name.to_string(), format!("{}", ctx.func));
            // a compile failure disables the JIT entirely; the VM still runs the
            // program correctly, so this only ever costs speed, never correctness
            module.define_function(id, &mut ctx).ok()?;
            module.clear_context(&mut ctx);
        }

        // 3) the f64 track: declare, then define via FloatGen (no deopt path —
        //    IEEE inf/NaN match the interpreter's float arms exactly)
        let mut fnames: Vec<&str> = feligible.iter().map(|s| s.as_str()).collect();
        fnames.sort();
        let mut fids: HashMap<String, (FuncId, usize)> = HashMap::new();
        for name in &fnames {
            let f = funcs[*name];
            let mut sig = module.make_signature();
            for _ in 0..f.params.len() { sig.params.push(AbiParam::new(types::F64)); }
            sig.returns.push(AbiParam::new(types::F64));
            let id = module.declare_function(name, Linkage::Export, &sig).ok()?;
            fids.insert(name.to_string(), (id, f.params.len()));
        }
        for name in &fnames {
            let f = funcs[*name];
            let (id, _) = fids[*name];
            for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(types::F64)); }
            ctx.func.signature.returns.push(AbiParam::new(types::F64));
            {
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
                let mut g = FloatGen::new(&mut b, &mut module, &fids, &f.params);
                g.lower(&f.body)?;
                b.finalize();
            }
            ir.insert(name.to_string(), format!("{}", ctx.func));
            module.define_function(id, &mut ctx).ok()?;
            module.clear_context(&mut ctx);
        }

        module.finalize_definitions().ok()?;
        let mut code = HashMap::new();
        for (name, (id, arity)) in &ids {
            code.insert(name.clone(), (module.get_finalized_function(*id), *arity));
        }
        let mut fcode = HashMap::new();
        for (name, (id, arity)) in &fids {
            fcode.insert(name.clone(), (module.get_finalized_function(*id), *arity));
        }
        Some(Jit { _module: module, code, fcode, ir })
    }

    pub fn is_compiled(&self, name: &str) -> bool { self.code.contains_key(name) }
    pub fn is_compiled_f64(&self, name: &str) -> bool { self.fcode.contains_key(name) }
    // any track — used by tiering to record what a batch produced
    pub fn has(&self, name: &str) -> bool {
        self.code.contains_key(name) || self.fcode.contains_key(name)
    }

    pub fn dump(&self) -> String {
        let mut names: Vec<&String> = self.ir.keys().collect();
        names.sort();
        let mut out = String::new();
        for n in names {
            out.push_str(&format!("; fn {}\n{}\n", n, self.ir[n]));
        }
        out
    }

    // Invoke a compiled function with integer arguments. Returns (result, deopt):
    // when `deopt` is true the result is meaningless and the caller must re-run
    // the call on the VM.
    pub fn raw_call(&self, name: &str, args: &[i64]) -> (i64, bool) {
        let (ptr, arity) = self.code[name];
        let mut d: i64 = 0;
        let dp = &mut d as *mut i64;
        // ABI: extern "C" fn(deopt_ptr, a0, a1, ...) -> i64
        let raw = unsafe {
            use std::mem::transmute as t;
            match arity {
                0 => (t::<_, extern "C" fn(*mut i64) -> i64>(ptr))(dp),
                1 => (t::<_, extern "C" fn(*mut i64, i64) -> i64>(ptr))(dp, args[0]),
                2 => (t::<_, extern "C" fn(*mut i64, i64, i64) -> i64>(ptr))(dp, args[0], args[1]),
                3 => (t::<_, extern "C" fn(*mut i64, i64, i64, i64) -> i64>(ptr))(dp, args[0], args[1], args[2]),
                4 => (t::<_, extern "C" fn(*mut i64, i64, i64, i64, i64) -> i64>(ptr))(dp, args[0], args[1], args[2], args[3]),
                5 => (t::<_, extern "C" fn(*mut i64, i64, i64, i64, i64, i64) -> i64>(ptr))(dp, args[0], args[1], args[2], args[3], args[4]),
                6 => (t::<_, extern "C" fn(*mut i64, i64, i64, i64, i64, i64, i64) -> i64>(ptr))(dp, args[0], args[1], args[2], args[3], args[4], args[5]),
                7 => (t::<_, extern "C" fn(*mut i64, i64, i64, i64, i64, i64, i64, i64) -> i64>(ptr))(dp, args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
                8 => (t::<_, extern "C" fn(*mut i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64>(ptr))(dp, args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                _ => unreachable!("arity > MAX_ARITY is not compiled"),
            }
        };
        (raw, d != 0)
    }
}

// raw code pointers are only ever read on the thread that built them
unsafe impl Send for Jit {}

impl Jit {
    // Invoke a compiled f64 function. Floats never deopt (no BigInt promotion;
    // /0.0 -> inf and NaN semantics are IEEE, identical to the interpreter).
    pub fn raw_call_f64(&self, name: &str, args: &[f64]) -> f64 {
        let (ptr, arity) = self.fcode[name];
        unsafe {
            use std::mem::transmute as t;
            match arity {
                0 => (t::<_, extern "C" fn() -> f64>(ptr))(),
                1 => (t::<_, extern "C" fn(f64) -> f64>(ptr))(args[0]),
                2 => (t::<_, extern "C" fn(f64, f64) -> f64>(ptr))(args[0], args[1]),
                3 => (t::<_, extern "C" fn(f64, f64, f64) -> f64>(ptr))(args[0], args[1], args[2]),
                4 => (t::<_, extern "C" fn(f64, f64, f64, f64) -> f64>(ptr))(args[0], args[1], args[2], args[3]),
                5 => (t::<_, extern "C" fn(f64, f64, f64, f64, f64) -> f64>(ptr))(args[0], args[1], args[2], args[3], args[4]),
                6 => (t::<_, extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64>(ptr))(args[0], args[1], args[2], args[3], args[4], args[5]),
                7 => (t::<_, extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> f64>(ptr))(args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
                8 => (t::<_, extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64>(ptr))(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
                _ => unreachable!("arity > MAX_ARITY is not compiled"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// f64 code generation — mirrors the interpreter's float arms exactly. No deopt
// machinery: IEEE semantics (inf/NaN) are already identical, and floats never
// promote to BigInt.
// ---------------------------------------------------------------------------

struct FloatGen<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    module: &'a mut JITModule,
    ids: &'a HashMap<String, (FuncId, usize)>,
    vars: HashMap<String, Variable>,
    n_vars: usize,
    loops: Vec<LoopCtx>,
    returned: bool,
}

impl<'a, 'b> FloatGen<'a, 'b> {
    fn new(b: &'a mut FunctionBuilder<'b>, module: &'a mut JITModule,
           ids: &'a HashMap<String, (FuncId, usize)>, params: &[String]) -> Self {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let param_vals: Vec<Value> = b.block_params(entry).to_vec();
        let mut g = FloatGen { b, module, ids, vars: HashMap::new(), n_vars: 0, loops: Vec::new(), returned: false };
        for (i, p) in params.iter().enumerate() {
            let v = g.declare(p);
            g.b.def_var(v, param_vals[i]);
        }
        g
    }

    fn declare(&mut self, name: &str) -> Variable {
        if let Some(v) = self.vars.get(name) { return *v; }
        let v = Variable::new(self.n_vars);
        self.n_vars += 1;
        self.b.declare_var(v, types::F64);
        self.vars.insert(name.to_string(), v);
        v
    }

    fn lower(&mut self, body: &[Stmt]) -> Option<()> {
        for s in body { self.stmt(s)?; }
        if !self.returned {
            let zero = self.b.ins().f64const(0.0);
            self.b.ins().return_(&[zero]);
        }
        Some(())
    }

    fn stmt(&mut self, s: &Stmt) -> Option<()> {
        if self.returned { return Some(()); }
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let v = self.expr(value)?;
                let var = self.declare(name);
                self.b.def_var(var, v);
            }
            Stmt::Expr(e) => { self.expr(e)?; }
            Stmt::Return(Some(e)) => {
                let v = self.expr(e)?;
                self.b.ins().return_(&[v]);
                self.returned = true;
            }
            Stmt::If { cond, then, els } => {
                let c = self.cond(cond)?;
                let then_b = self.b.create_block();
                let else_b = self.b.create_block();
                let merge = self.b.create_block();
                self.b.ins().brif(c, then_b, &[], else_b, &[]);
                self.b.switch_to_block(then_b);
                self.b.seal_block(then_b);
                self.returned = false;
                for s in then { self.stmt(s)?; }
                if !self.returned { self.b.ins().jump(merge, &[]); }
                self.b.switch_to_block(else_b);
                self.b.seal_block(else_b);
                self.returned = false;
                if let Some(els) = els { for s in els { self.stmt(s)?; } }
                if !self.returned { self.b.ins().jump(merge, &[]); }
                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
                self.returned = false;
            }
            Stmt::While { cond, body } => {
                let header = self.b.create_block();
                let body_b = self.b.create_block();
                let exit = self.b.create_block();
                self.b.ins().jump(header, &[]);
                self.b.switch_to_block(header);
                let c = self.cond(cond)?;
                self.b.ins().brif(c, body_b, &[], exit, &[]);
                self.b.switch_to_block(body_b);
                self.b.seal_block(body_b);
                self.loops.push(LoopCtx { header, exit });
                self.returned = false;
                for s in body { self.stmt(s)?; }
                if !self.returned { self.b.ins().jump(header, &[]); }
                self.loops.pop();
                self.b.seal_block(header);
                self.b.switch_to_block(exit);
                self.b.seal_block(exit);
                self.returned = false;
            }
            Stmt::Break(None) => {
                let exit = self.loops.last()?.exit;
                self.b.ins().jump(exit, &[]);
                self.returned = true;
            }
            Stmt::Continue => {
                let header = self.loops.last()?.header;
                self.b.ins().jump(header, &[]);
                self.returned = true;
            }
            _ => return None,
        }
        Some(())
    }

    // boolean condition (i64 0/1): float comparisons + short-circuit && || !
    fn cond(&mut self, e: &Expr) -> Option<Value> {
        match e {
            Expr::At { expr, .. } => self.cond(expr),
            Expr::Unary { op: UnOp::Not, expr } => {
                let v = self.cond(expr)?;
                let one = self.b.ins().iconst(I64, 1);
                Some(self.b.ins().bxor(v, one))
            }
            Expr::Binary { op: op @ (BinOp::And | BinOp::Or), lhs, rhs } => {
                let a = self.cond(lhs)?;
                let then_b = self.b.create_block();
                let else_b = self.b.create_block();
                let merge = self.b.create_block();
                self.b.append_block_param(merge, I64);
                self.b.ins().brif(a, then_b, &[], else_b, &[]);
                self.b.switch_to_block(then_b);
                self.b.seal_block(then_b);
                if matches!(op, BinOp::And) {
                    let bv = self.cond(rhs)?;
                    self.b.ins().jump(merge, &[bv]);
                } else {
                    let one = self.b.ins().iconst(I64, 1);
                    self.b.ins().jump(merge, &[one]);
                }
                self.b.switch_to_block(else_b);
                self.b.seal_block(else_b);
                if matches!(op, BinOp::And) {
                    let zero = self.b.ins().iconst(I64, 0);
                    self.b.ins().jump(merge, &[zero]);
                } else {
                    let bv = self.cond(rhs)?;
                    self.b.ins().jump(merge, &[bv]);
                }
                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
                Some(self.b.block_params(merge)[0])
            }
            Expr::Binary { op, lhs, rhs } => {
                let a = self.expr(lhs)?;
                let bv = self.expr(rhs)?;
                let cc = match op {
                    BinOp::Eq => FloatCC::Equal,
                    BinOp::Ne => FloatCC::NotEqual,
                    BinOp::Lt => FloatCC::LessThan,
                    BinOp::Le => FloatCC::LessThanOrEqual,
                    BinOp::Gt => FloatCC::GreaterThan,
                    BinOp::Ge => FloatCC::GreaterThanOrEqual,
                    _ => return None,
                };
                let c = self.b.ins().fcmp(cc, a, bv);
                Some(self.b.ins().uextend(I64, c))
            }
            _ => None,
        }
    }

    fn expr(&mut self, e: &Expr) -> Option<Value> {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Float(x) => Some(self.b.ins().f64const(*x)),
            Expr::Ident(name) => {
                let v = *self.vars.get(name)?;
                Some(self.b.use_var(v))
            }
            Expr::Unary { op: UnOp::Neg, expr } => {
                let v = self.expr(expr)?;
                Some(self.b.ins().fneg(v))
            }
            Expr::Binary { op, lhs, rhs } => {
                let a = self.expr(lhs)?;
                let bv = self.expr(rhs)?;
                Some(match op {
                    BinOp::Add => self.b.ins().fadd(a, bv),
                    BinOp::Sub => self.b.ins().fsub(a, bv),
                    BinOp::Mul => self.b.ins().fmul(a, bv),
                    BinOp::Div => self.b.ins().fdiv(a, bv), // /0.0 -> inf, as the interp
                    _ => return None,
                })
            }
            Expr::Call { callee, args } => {
                let (id, arity) = *self.ids.get(callee.as_str())?;
                if args.len() != arity { return None; }
                let fref = self.module.declare_func_in_func(id, self.b.func);
                let mut argv = Vec::with_capacity(arity);
                for a in args { argv.push(self.expr(a)?); }
                let inst = self.b.ins().call(fref, &argv);
                Some(self.b.inst_results(inst)[0])
            }
            Expr::If { cond, then, els } => {
                let c = self.cond(cond)?;
                let then_b = self.b.create_block();
                let else_b = self.b.create_block();
                let merge = self.b.create_block();
                self.b.append_block_param(merge, types::F64);
                self.b.ins().brif(c, then_b, &[], else_b, &[]);
                self.b.switch_to_block(then_b);
                self.b.seal_block(then_b);
                let tv = self.expr(then)?;
                self.b.ins().jump(merge, &[tv]);
                self.b.switch_to_block(else_b);
                self.b.seal_block(else_b);
                let ev = self.expr(els)?;
                self.b.ins().jump(merge, &[ev]);
                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
                Some(self.b.block_params(merge)[0])
            }
            Expr::Block { stmts, tail } => {
                for s in stmts { self.stmt(s)?; }
                self.expr(tail.as_ref()?)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tiering (Phase 5C): compile a function only after it has been called
// `threshold` times. When a function turns hot, its whole callee closure
// (within the eligible set) is compiled in one batch so JIT->JIT direct calls
// stay valid; cold functions unreachable from any hot root are NEVER compiled.
// ---------------------------------------------------------------------------

pub struct TieredJit<'p> {
    prog: &'p Program,
    eligible: HashSet<String>,   // i64 track
    feligible: HashSet<String>,  // f64 track (disjoint)
    // call edges within the eligible sets, for closure computation
    callees: HashMap<String, Vec<String>>,
    pub threshold: u64,
    // each hot root compiles into its own self-contained module batch
    jits: std::cell::RefCell<Vec<Jit>>,
    location: std::cell::RefCell<HashMap<String, usize>>, // name -> batch index
    compiled_order: std::cell::RefCell<Vec<String>>,      // observable for --jit-stats
    backend_failed: std::cell::Cell<bool>,
}

impl<'p> TieredJit<'p> {
    pub fn new(prog: &'p Program, threshold: u64) -> TieredJit<'p> {
        let eligible = eligible_set(prog);
        let feligible = float_eligible_set(prog, &eligible);
        let mut callees: HashMap<String, Vec<String>> = HashMap::new();
        for item in &prog.items {
            if let Item::Func(f) = item {
                if eligible.contains(&f.name) || feligible.contains(&f.name) {
                    let cs: Vec<String> = collect_calls(&f.body).into_iter()
                        .filter(|c| eligible.contains(c) || feligible.contains(c)).collect();
                    callees.insert(f.name.clone(), cs);
                }
            }
        }
        TieredJit {
            prog, eligible, feligible, callees,
            threshold: threshold.max(1),
            jits: std::cell::RefCell::new(Vec::new()),
            location: std::cell::RefCell::new(HashMap::new()),
            compiled_order: std::cell::RefCell::new(Vec::new()),
            backend_failed: std::cell::Cell::new(false),
        }
    }

    pub fn is_eligible(&self, name: &str) -> bool {
        self.eligible.contains(name) || self.feligible.contains(name)
    }
    pub fn is_compiled(&self, name: &str) -> bool {
        match self.location.borrow().get(name) {
            Some(idx) => self.jits.borrow()[*idx].is_compiled(name),
            None => false,
        }
    }
    pub fn is_compiled_f64(&self, name: &str) -> bool {
        match self.location.borrow().get(name) {
            Some(idx) => self.jits.borrow()[*idx].is_compiled_f64(name),
            None => false,
        }
    }
    // names in the order their batches were compiled — proof of what got compiled
    pub fn compiled_functions(&self) -> Vec<String> { self.compiled_order.borrow().clone() }

    // Compile `root` plus every eligible function reachable from it.
    pub fn compile_closure(&self, root: &str) {
        if self.backend_failed.get() || !self.is_eligible(root)
            || self.location.borrow().contains_key(root) {
            return;
        }
        let mut set: HashSet<String> = HashSet::new();
        let mut work = vec![root.to_string()];
        while let Some(n) = work.pop() {
            if set.insert(n.clone()) {
                if let Some(cs) = self.callees.get(&n) {
                    for c in cs { if !set.contains(c) { work.push(c.clone()); } }
                }
            }
        }
        match Jit::compile_filtered(self.prog, Some(&set)) {
            Some(j) => {
                let mut names: Vec<String> = set.into_iter().filter(|n| j.has(n)).collect();
                names.sort();
                let mut jits = self.jits.borrow_mut();
                let idx = jits.len();
                jits.push(j);
                let mut loc = self.location.borrow_mut();
                let mut order = self.compiled_order.borrow_mut();
                for n in names {
                    // later batches may re-include an already-compiled callee;
                    // keep the first pointer (both are correct)
                    if !loc.contains_key(&n) {
                        loc.insert(n.clone(), idx);
                        order.push(n);
                    }
                }
            }
            // a backend failure only ever costs speed: stop trying, run on the VM
            None => self.backend_failed.set(true),
        }
    }

    // caller must check `is_compiled` first
    pub fn raw_call(&self, name: &str, args: &[i64]) -> (i64, bool) {
        let idx = self.location.borrow()[name];
        self.jits.borrow()[idx].raw_call(name, args)
    }

    // caller must check `is_compiled_f64` first
    pub fn raw_call_f64(&self, name: &str, args: &[f64]) -> f64 {
        let idx = self.location.borrow()[name];
        self.jits.borrow()[idx].raw_call_f64(name, args)
    }
}

// ---------------------------------------------------------------------------
// Per-function code generation
// ---------------------------------------------------------------------------

struct LoopCtx { header: Block, exit: Block }

struct FnGen<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    module: &'a mut JITModule,
    ids: &'a HashMap<String, (FuncId, usize)>,
    vars: HashMap<String, Variable>,
    n_vars: usize,
    deopt_ptr: Value,
    deopt_block: Block,
    loops: Vec<LoopCtx>,
    returned: bool,
}

impl<'a, 'b> FnGen<'a, 'b> {
    fn new(b: &'a mut FunctionBuilder<'b>, module: &'a mut JITModule,
           ids: &'a HashMap<String, (FuncId, usize)>, params: &[String]) -> Self {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let deopt_ptr = b.block_params(entry)[0];
        let param_vals: Vec<Value> = b.block_params(entry)[1..].to_vec();
        let deopt_block = b.create_block();
        let mut g = FnGen {
            b, module, ids, vars: HashMap::new(), n_vars: 0,
            deopt_ptr, deopt_block, loops: Vec::new(), returned: false,
        };
        for (i, p) in params.iter().enumerate() {
            let v = g.declare(p);
            g.b.def_var(v, param_vals[i]);
        }
        g
    }

    fn declare(&mut self, name: &str) -> Variable {
        if let Some(v) = self.vars.get(name) { return *v; }
        let v = Variable::new(self.n_vars);
        self.n_vars += 1;
        self.b.declare_var(v, I64);
        self.vars.insert(name.to_string(), v);
        v
    }

    fn fresh_var(&mut self) -> Variable {
        let v = Variable::new(self.n_vars);
        self.n_vars += 1;
        self.b.declare_var(v, I64);
        v
    }

    fn lower(&mut self, body: &[Stmt]) -> Option<()> {
        for s in body { self.stmt(s)?; }
        if !self.returned {
            let zero = self.b.ins().iconst(I64, 0);
            self.b.ins().return_(&[zero]);
        }
        self.b.switch_to_block(self.deopt_block);
        self.b.seal_block(self.deopt_block);
        let one = self.b.ins().iconst(I64, 1);
        self.b.ins().store(MemFlags::trusted(), one, self.deopt_ptr, 0);
        let zero = self.b.ins().iconst(I64, 0);
        self.b.ins().return_(&[zero]);
        Some(())
    }

    fn guard_deopt(&mut self, cond: Value) {
        let cont = self.b.create_block();
        self.b.ins().brif(cond, self.deopt_block, &[], cont, &[]);
        self.b.switch_to_block(cont);
        self.b.seal_block(cont);
    }

    fn stmt(&mut self, s: &Stmt) -> Option<()> {
        if self.returned { return Some(()); }
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let v = self.expr(value)?;
                let var = self.declare(name);
                self.b.def_var(var, v);
            }
            Stmt::Expr(e) => { self.expr(e)?; }
            Stmt::Return(Some(e)) => {
                let v = self.expr(e)?;
                self.b.ins().return_(&[v]);
                self.returned = true;
            }
            Stmt::If { cond, then, els } => self.if_stmt(cond, then, els.as_deref())?,
            Stmt::While { cond, body } => self.while_stmt(cond, body)?,
            Stmt::ForRange { var, start, end, inclusive, body } =>
                self.for_range(var, start, end, *inclusive, body)?,
            Stmt::Break(None) => {
                let exit = self.loops.last()?.exit;
                self.b.ins().jump(exit, &[]);
                self.returned = true;
            }
            Stmt::Continue => {
                let header = self.loops.last()?.header;
                self.b.ins().jump(header, &[]);
                self.returned = true;
            }
            _ => return None,
        }
        Some(())
    }

    fn if_stmt(&mut self, cond: &Expr, then: &[Stmt], els: Option<&[Stmt]>) -> Option<()> {
        let c = self.truthy(cond)?;
        let then_b = self.b.create_block();
        let else_b = self.b.create_block();
        let merge = self.b.create_block();
        self.b.ins().brif(c, then_b, &[], else_b, &[]);

        self.b.switch_to_block(then_b);
        self.b.seal_block(then_b);
        self.returned = false;
        for s in then { self.stmt(s)?; }
        if !self.returned { self.b.ins().jump(merge, &[]); }

        self.b.switch_to_block(else_b);
        self.b.seal_block(else_b);
        self.returned = false;
        if let Some(els) = els { for s in els { self.stmt(s)?; } }
        if !self.returned { self.b.ins().jump(merge, &[]); }

        self.b.switch_to_block(merge);
        self.b.seal_block(merge);
        self.returned = false;
        Some(())
    }

    fn while_stmt(&mut self, cond: &Expr, body: &[Stmt]) -> Option<()> {
        let header = self.b.create_block();
        let body_b = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);

        self.b.switch_to_block(header);
        let c = self.truthy(cond)?;
        self.b.ins().brif(c, body_b, &[], exit, &[]);

        self.b.switch_to_block(body_b);
        self.b.seal_block(body_b);
        self.loops.push(LoopCtx { header, exit });
        self.returned = false;
        for s in body { self.stmt(s)?; }
        if !self.returned { self.b.ins().jump(header, &[]); }
        self.loops.pop();

        self.b.seal_block(header);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        self.returned = false;
        Some(())
    }

    fn for_range(&mut self, var: &str, start: &Expr, end: &Expr, inclusive: bool, body: &[Stmt]) -> Option<()> {
        let s = self.expr(start)?;
        let e = self.expr(end)?;
        let iv = self.declare(var);
        self.b.def_var(iv, s);
        let limit = self.fresh_var();
        self.b.def_var(limit, e);

        let header = self.b.create_block();
        let body_b = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);

        self.b.switch_to_block(header);
        let i = self.b.use_var(iv);
        let lim = self.b.use_var(limit);
        let cc = if inclusive { IntCC::SignedLessThanOrEqual } else { IntCC::SignedLessThan };
        let cont = self.b.ins().icmp(cc, i, lim);
        self.b.ins().brif(cont, body_b, &[], exit, &[]);

        self.b.switch_to_block(body_b);
        self.b.seal_block(body_b);
        self.loops.push(LoopCtx { header, exit });
        self.returned = false;
        for st in body { self.stmt(st)?; }
        if !self.returned {
            let i = self.b.use_var(iv);
            let one = self.b.ins().iconst(I64, 1);
            let next = self.add_checked(i, one);
            self.b.def_var(iv, next);
            self.b.ins().jump(header, &[]);
        }
        self.loops.pop();

        self.b.seal_block(header);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        self.returned = false;
        Some(())
    }

    fn truthy(&mut self, e: &Expr) -> Option<Value> {
        let v = self.expr(e)?;
        let zero = self.b.ins().iconst(I64, 0);
        let nz = self.b.ins().icmp(IntCC::NotEqual, v, zero);
        Some(self.b.ins().uextend(I64, nz))
    }

    fn expr(&mut self, e: &Expr) -> Option<Value> {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Int(n) => Some(self.b.ins().iconst(I64, *n)),
            Expr::Ident(name) => {
                let v = *self.vars.get(name)?;
                Some(self.b.use_var(v))
            }
            Expr::Unary { op, expr } => {
                let v = self.expr(expr)?;
                match op {
                    UnOp::Neg => {
                        let min = self.b.ins().iconst(I64, i64::MIN);
                        let is_min = self.b.ins().icmp(IntCC::Equal, v, min);
                        let is_min = self.b.ins().uextend(I64, is_min);
                        self.guard_deopt(is_min);
                        Some(self.b.ins().ineg(v))
                    }
                    UnOp::BitNot => Some(self.b.ins().bnot(v)),
                    UnOp::Not => {
                        let zero = self.b.ins().iconst(I64, 0);
                        let isz = self.b.ins().icmp(IntCC::Equal, v, zero);
                        Some(self.b.ins().uextend(I64, isz))
                    }
                }
            }
            Expr::Binary { op, lhs, rhs } => self.binop(*op, lhs, rhs),
            Expr::Call { callee, args } => self.call(callee, args),
            Expr::If { cond, then, els } => {
                let c = self.truthy(cond)?;
                let then_b = self.b.create_block();
                let else_b = self.b.create_block();
                let merge = self.b.create_block();
                self.b.append_block_param(merge, I64);
                self.b.ins().brif(c, then_b, &[], else_b, &[]);

                self.b.switch_to_block(then_b);
                self.b.seal_block(then_b);
                let tv = self.expr(then)?;
                self.b.ins().jump(merge, &[tv]);

                self.b.switch_to_block(else_b);
                self.b.seal_block(else_b);
                let ev = self.expr(els)?;
                self.b.ins().jump(merge, &[ev]);

                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
                Some(self.b.block_params(merge)[0])
            }
            Expr::Block { stmts, tail } => {
                for s in stmts { self.stmt(s)?; }
                self.expr(tail.as_ref()?)
            }
            _ => None,
        }
    }

    // native call to another eligible function, propagating deopt
    fn call(&mut self, callee: &str, args: &[Expr]) -> Option<Value> {
        let (id, arity) = *self.ids.get(callee)?;
        if args.len() != arity { return None; }
        let fref = self.module.declare_func_in_func(id, self.b.func);
        let mut argv: Vec<Value> = Vec::with_capacity(arity + 1);
        argv.push(self.deopt_ptr);
        for a in args { argv.push(self.expr(a)?); }
        let inst = self.b.ins().call(fref, &argv);
        let res = self.b.inst_results(inst)[0];
        // if the callee asked to deopt, bubble it up
        let flag = self.b.ins().load(I64, MemFlags::trusted(), self.deopt_ptr, 0);
        self.guard_deopt(flag);
        Some(res)
    }

    fn binop(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Option<Value> {
        use BinOp::*;
        if matches!(op, And | Or) {
            return self.logical(op, lhs, rhs);
        }
        let a = self.expr(lhs)?;
        let b = self.expr(rhs)?;
        let v = match op {
            Add => self.add_checked(a, b),
            Sub => self.sub_checked(a, b),
            Mul => self.mul_checked(a, b),
            Div | Rem => {
                let zero = self.b.ins().iconst(I64, 0);
                let is_zero = self.b.ins().icmp(IntCC::Equal, b, zero);
                let is_zero = self.b.ins().uextend(I64, is_zero);
                self.guard_deopt(is_zero);
                let min = self.b.ins().iconst(I64, i64::MIN);
                let neg1 = self.b.ins().iconst(I64, -1);
                let amin = self.b.ins().icmp(IntCC::Equal, a, min);
                let bn1 = self.b.ins().icmp(IntCC::Equal, b, neg1);
                let both = self.b.ins().band(amin, bn1);
                let both = self.b.ins().uextend(I64, both);
                self.guard_deopt(both);
                if matches!(op, Div) { self.b.ins().sdiv(a, b) } else { self.b.ins().srem(a, b) }
            }
            Pow => {
                let one = self.b.ins().iconst(I64, 1);
                self.guard_deopt(one);
                self.b.ins().iconst(I64, 0)
            }
            BitOr => self.b.ins().bor(a, b),
            BitXor => self.b.ins().bxor(a, b),
            BitAnd => self.b.ins().band(a, b),
            Shl | Shr => {
                let zero = self.b.ins().iconst(I64, 0);
                let sixtyfour = self.b.ins().iconst(I64, 64);
                let lt0 = self.b.ins().icmp(IntCC::SignedLessThan, b, zero);
                let ge64 = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, b, sixtyfour);
                let bad = self.b.ins().bor(lt0, ge64);
                let bad = self.b.ins().uextend(I64, bad);
                self.guard_deopt(bad);
                if matches!(op, Shl) { self.b.ins().ishl(a, b) } else { self.b.ins().sshr(a, b) }
            }
            Eq => { let c = self.b.ins().icmp(IntCC::Equal, a, b); self.b.ins().uextend(I64, c) }
            Ne => { let c = self.b.ins().icmp(IntCC::NotEqual, a, b); self.b.ins().uextend(I64, c) }
            Lt | Le | Gt | Ge => self.cmp_as_float(op, a, b),
            And | Or => unreachable!(),
        };
        Some(v)
    }

    fn logical(&mut self, op: BinOp, lhs: &Expr, rhs: &Expr) -> Option<Value> {
        let a = self.truthy(lhs)?;
        let then_b = self.b.create_block();
        let else_b = self.b.create_block();
        let merge = self.b.create_block();
        self.b.append_block_param(merge, I64);
        self.b.ins().brif(a, then_b, &[], else_b, &[]);

        self.b.switch_to_block(then_b);
        self.b.seal_block(then_b);
        if matches!(op, BinOp::And) {
            let bv = self.truthy(rhs)?;
            self.b.ins().jump(merge, &[bv]);
        } else {
            let one = self.b.ins().iconst(I64, 1);
            self.b.ins().jump(merge, &[one]);
        }

        self.b.switch_to_block(else_b);
        self.b.seal_block(else_b);
        if matches!(op, BinOp::And) {
            let zero = self.b.ins().iconst(I64, 0);
            self.b.ins().jump(merge, &[zero]);
        } else {
            let bv = self.truthy(rhs)?;
            self.b.ins().jump(merge, &[bv]);
        }

        self.b.switch_to_block(merge);
        self.b.seal_block(merge);
        Some(self.b.block_params(merge)[0])
    }

    fn cmp_as_float(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        let af = self.b.ins().fcvt_from_sint(types::F64, a);
        let bf = self.b.ins().fcvt_from_sint(types::F64, b);
        let cc = match op {
            BinOp::Lt => FloatCC::LessThan,
            BinOp::Le => FloatCC::LessThanOrEqual,
            BinOp::Gt => FloatCC::GreaterThan,
            BinOp::Ge => FloatCC::GreaterThanOrEqual,
            _ => unreachable!(),
        };
        let c = self.b.ins().fcmp(cc, af, bf);
        self.b.ins().uextend(I64, c)
    }

    fn add_checked(&mut self, a: Value, b: Value) -> Value {
        let r = self.b.ins().iadd(a, b);
        let t1 = self.b.ins().bxor(a, r);
        let t2 = self.b.ins().bxor(b, r);
        let t3 = self.b.ins().band(t1, t2);
        let zero = self.b.ins().iconst(I64, 0);
        let ovf = self.b.ins().icmp(IntCC::SignedLessThan, t3, zero);
        let ovf = self.b.ins().uextend(I64, ovf);
        self.guard_deopt(ovf);
        r
    }

    fn sub_checked(&mut self, a: Value, b: Value) -> Value {
        let r = self.b.ins().isub(a, b);
        let t1 = self.b.ins().bxor(a, b);
        let t2 = self.b.ins().bxor(a, r);
        let t3 = self.b.ins().band(t1, t2);
        let zero = self.b.ins().iconst(I64, 0);
        let ovf = self.b.ins().icmp(IntCC::SignedLessThan, t3, zero);
        let ovf = self.b.ins().uextend(I64, ovf);
        self.guard_deopt(ovf);
        r
    }

    fn mul_checked(&mut self, a: Value, b: Value) -> Value {
        let a128 = self.b.ins().sextend(I128, a);
        let b128 = self.b.ins().sextend(I128, b);
        let m = self.b.ins().imul(a128, b128);
        let m64 = self.b.ins().ireduce(I64, m);
        let back = self.b.ins().sextend(I128, m64);
        let ok = self.b.ins().icmp(IntCC::Equal, m, back);
        let bad = self.b.ins().bnot(ok);
        let bad = self.b.ins().uextend(I64, bad);
        let one = self.b.ins().iconst(I64, 1);
        let bad = self.b.ins().band(bad, one);
        self.guard_deopt(bad);
        m64
    }
}

#[cfg(test)]
mod jit_tests {
    use super::Jit;
    use crate::parser::parse_program;
    use crate::interp::{Interp, fold_program};
    use crate::bytecode::{compile_program, eval_main_jit};

    // The JIT must match the plain VM exactly — including deopt cases (overflow,
    // div-by-zero) — since the VM is itself verified byte-identical to the
    // interpreter elsewhere.
    fn same_jit(src: &str) -> String {
        let mut prog = parse_program(src).expect("parse");
        fold_program(&mut prog);
        let compiled = compile_program(&prog).expect("compile");
        let jit = Jit::compile(&prog);
        // every test program here has an eligible function — guard against the
        // helper passing trivially because nothing was actually JIT-compiled
        assert!(jit.is_some(), "expected something to JIT-compile for: {}", src);
        let i1 = Interp::new(&prog).expect("i1");
        let with_jit = eval_main_jit(&compiled, &i1, jit.as_ref());
        let i2 = Interp::new(&prog).expect("i2");
        let no_jit = eval_main_jit(&compiled, &i2, None);
        let a = match &with_jit { Ok(v) => format!("OK:{}", v), Err(e) => format!("ERR:{}", e) };
        let b = match &no_jit { Ok(v) => format!("OK:{}", v), Err(e) => format!("ERR:{}", e) };
        assert_eq!(a, b, "JIT != VM for: {}", src);
        a
    }

    #[test] fn jit_gcd() {
        same_jit("fn gcd(a,b){ while b != 0 { let t=b; b=a%b; a=t; } a } fn main(){ gcd(1071,462) }");
    }
    #[test] fn jit_sum_loop() {
        same_jit("fn s(n){ let t=0; for i in 1..=n { t=t+i*i; } t } fn main(){ s(1000) }");
    }
    #[test] fn jit_collatz() {
        same_jit("fn c(n){ let k=0; while n!=1 { if n%2==0 {n=n/2} else {n=3*n+1}; k=k+1 } k } fn main(){ c(27) }");
    }
    #[test] fn jit_branches_and_bits() {
        same_jit("fn f(n){ if n>0 { (n & 6) | 1 } else { 0 - n } } fn main(){ f(13) + f(0-4) }");
    }
    #[test] fn jit_comparisons() {
        same_jit("fn f(a,b){ if a<b { 1 } else if a==b { 0 } else { 0-1 } } fn main(){ f(3,5)+f(5,5)+f(9,2) }");
    }
    #[test] fn jit_shifts() {
        same_jit("fn f(n){ (n << 3) + (n >> 1) } fn main(){ f(40) }");
    }
    #[test] fn jit_overflow_promotes_to_bigint() {
        same_jit("fn fac(n){ let p=1; for i in 1..=n { p=p*i; } p } fn main(){ fac(25) }");
    }
    #[test] fn jit_div_by_zero_errors() {
        same_jit("fn f(a,b){ a / b } fn main(){ f(10, 0) }");
    }
    #[test] fn jit_short_circuit() {
        same_jit("fn f(a,b){ if (a>0) && (b>0) { 1 } else { 0 } } fn main(){ f(1,0)+f(2,3)+f(0,9) }");
    }
    #[test] fn jit_logic_or() {
        same_jit("fn f(a,b){ if (a>0) || (b>0) { 1 } else { 0 } } fn main(){ f(0,0)+f(1,0)+f(0,1) }");
    }
    // --- 5B: native recursion + deopt propagation through calls ---
    #[test] fn jit_recursion_fib() {
        same_jit("fn fib(n){ if n<2 { n } else { fib(n-1)+fib(n-2) } } fn main(){ fib(20) }");
    }
    #[test] fn jit_mutual_recursion() {
        same_jit("fn iseven(n){ if n==0 { 1 } else { isodd(n-1) } }\nfn isodd(n){ if n==0 { 0 } else { iseven(n-1) } }\nfn main(){ iseven(10) + isodd(7) }");
    }
    #[test] fn jit_call_chain() {
        same_jit("fn inc(x){ x+1 }\nfn dbl(x){ inc(x)+inc(x) }\nfn main(){ dbl(20) }");
    }
    #[test] fn jit_deopt_through_call() {
        // factorial via recursion overflows i64 at n=25 -> deopt must bubble up
        same_jit("fn fac(n){ if n<=1 { 1 } else { n * fac(n-1) } } fn main(){ fac(30) }");
    }

    // --- 5C.1: tiering ---
    use super::TieredJit;
    use crate::bytecode::eval_main_tiered;

    // run tiered vs plain VM, assert identical, and return compiled fn names
    fn same_tiered(src: &str, threshold: u64) -> Vec<String> {
        let mut prog = parse_program(src).expect("parse");
        fold_program(&mut prog);
        let compiled = compile_program(&prog).expect("compile");
        let t = TieredJit::new(&prog, threshold);
        let i1 = Interp::new(&prog).expect("i1");
        let tiered = eval_main_tiered(&compiled, &i1, &t);
        let i2 = Interp::new(&prog).expect("i2");
        let plain = eval_main_jit(&compiled, &i2, None);
        let a = match &tiered { Ok(v) => format!("OK:{}", v), Err(e) => format!("ERR:{}", e) };
        let b = match &plain { Ok(v) => format!("OK:{}", v), Err(e) => format!("ERR:{}", e) };
        assert_eq!(a, b, "tiered != VM for: {}", src);
        t.compiled_functions()
    }

    #[test] fn tiering_cold_never_compiles_hot_does() {
        let names = same_tiered(
            "fn cold1(a,b){ a*b+1 }\nfn cold2(x){ x-7 }\n\
             fn fib(n){ if n<2 { n } else { fib(n-1)+fib(n-2) } }\n\
             fn main(){ cold1(3,4) + cold2(100) + fib(22) }", 50);
        assert_eq!(names, vec!["fib".to_string()], "only the hot function may compile");
    }
    #[test] fn tiering_crosses_threshold_mid_recursion() {
        // fib crosses the threshold deep inside its own recursion
        let names = same_tiered(
            "fn fib(n){ if n<2 { n } else { fib(n-1)+fib(n-2) } } fn main(){ fib(24) }", 100);
        assert!(names.contains(&"fib".to_string()));
    }
    #[test] fn tiering_hot_closure_includes_callees() {
        // hot `outer` must pull its callee `helper` into the same batch
        let names = same_tiered(
            "fn helper(x){ x*2+1 }\nfn outer(n){ let s=0; for i in 0..n { s = s + helper(i) } s }\n\
             fn main(){ let t=0; for k in 0..300 { t = t + outer(5) } t }", 50);
        assert!(names.contains(&"outer".to_string()) && names.contains(&"helper".to_string()));
    }
    #[test] fn tiering_deopt_after_hot() {
        // becomes hot on small args, then overflows -> deopt path re-runs on VM
        same_tiered(
            "fn fac(n){ let p=1; for i in 1..=n { p=p*i } p }\n\
             fn main(){ let t=0; for k in 0..200 { t = t + fac(10) } t + fac(25) }", 50);
    }
    #[test] fn tiering_below_threshold_never_compiles() {
        let names = same_tiered(
            "fn f(x){ x+1 } fn main(){ f(1)+f(2)+f(3) }", 100);
        assert!(names.is_empty(), "cold-only program must compile nothing");
    }

    // --- 5C.2: float (f64) specialization — must be bit-identical ---
    #[test] fn float_arith_and_compare() {
        same_jit("fn f(a,b){ if a < b { a*2.5 + b } else { a/b - 0.5 } } fn main(){ f(1.5, 4.0) + f(9.0, 2.0) }");
    }
    #[test] fn float_div_by_zero_is_inf() {
        // /0.0 -> inf natively; must match the interpreter exactly (no error)
        same_jit("fn f(a,b){ a / b } fn main(){ f(3.5, 0.0) }");
    }
    #[test] fn float_neg_zero_and_nan_compares() {
        same_jit("fn f(a,b){ if a == b { 1.0 } else { 0.0 } } fn main(){ f(0.0, -0.0) + f(0.0/0.0, 0.0/0.0) }");
    }
    #[test] fn float_while_loop() {
        same_jit("fn s(n){ let t = 0.0; let i = 0.0; while i < n { t = t + i*0.5; i = i + 1.0 } t } fn main(){ s(100.0) }");
    }
    #[test] fn float_recursion() {
        same_jit("fn geo(x){ if x < 0.001 { 0.0 } else { x + geo(x * 0.5) } } fn main(){ geo(8.0) }");
    }
    #[test] fn float_short_circuit() {
        same_jit("fn f(a,b){ if (a > 0.0) && (b > 0.0) { 1.0 } else { 0.0 } } fn main(){ f(1.0,0.0)+f(2.0,3.0) }");
    }
    #[test] fn float_tiered() {
        let names = same_tiered(
            "fn hot(x){ x * 1.0001 + 0.5 }\n\
             fn main(){ let t = 0.0; let i = 0.0; while i < 500.0 { t = hot(t); i = i + 1.0 } t }", 50);
        assert!(names.contains(&"hot".to_string()), "hot float fn must compile");
    }
    #[test] fn float_mixed_stays_on_vm() {
        // int literal inside float math -> ineligible; must still be identical (VM path)
        same_jit("fn f(a){ a * 2 } fn main(){ f(1.5) }");
    }
}
