// A gradual static type checker for Nova. It runs one pass over the AST before
// execution and reports real errors: undefined variables, wrong arity, operators
// applied to incompatible types, non-boolean conditions, and unknown fields.
//
// The checker is deliberately gradual: when a type cannot be determined it
// becomes `Unknown`, which is compatible with everything. This catches genuine
// mistakes without rejecting valid dynamic code that the interpreter accepts.

use crate::ast::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Int,
    Float,
    Str,
    Bool,
    Null,
    // `Array(elem)` — parametric element type; `elem` is `Unknown` when the
    // element type isn't statically known (a bare `Array`/`[]`).
    Array(Box<Ty>),
    Map,
    Struct(String),
    Func,
    Unknown,
}

impl Ty {
    // a bare array whose element type is unknown
    fn arr_unknown() -> Ty { Ty::Array(Box::new(Ty::Unknown)) }
    fn arr(elem: Ty) -> Ty { Ty::Array(Box::new(elem)) }

    fn name(&self) -> String {
        match self {
            Ty::Int => "Int".into(),
            Ty::Float => "Float".into(),
            Ty::Str => "Str".into(),
            Ty::Bool => "Bool".into(),
            Ty::Null => "Null".into(),
            Ty::Array(e) => if matches!(**e, Ty::Unknown) { "Array".into() } else { format!("[{}]", e.name()) },
            Ty::Map => "Map".into(),
            Ty::Struct(n) => n.clone(),
            Ty::Func => "Func".into(),
            Ty::Unknown => "Unknown".into(),
        }
    }
    fn is_num(&self) -> bool {
        matches!(self, Ty::Int | Ty::Float | Ty::Unknown)
    }
    // gradual compatibility: Unknown unifies with anything; arrays are compatible
    // when their element types are (an unknown element makes it match any array).
    fn compatible(&self, other: &Ty) -> bool {
        match (self, other) {
            (Ty::Unknown, _) | (_, Ty::Unknown) => true,
            (Ty::Array(a), Ty::Array(b)) => a.compatible(b),
            _ => self == other || (self.is_num() && other.is_num()),
        }
    }
}

#[derive(Debug, Clone)]
struct FnSig {
    param_types: Vec<Option<String>>,
    ret_type: Option<String>,
    type_params: Vec<String>,
    // generic param -> required trait names, from `[T: Trait]` and `where` clauses
    where_bounds: Vec<(String, Vec<String>)>,
    // declared effects from `![..]`: None = unannotated (not effect-checked)
    effects: Option<Vec<String>>,
}

pub struct Checker {
    // function name -> parameter count (for arity checks)
    funcs: HashMap<String, usize>,
    // function name -> its signature info (param type heads, return, generics)
    sigs: HashMap<String, FnSig>,
    // function name -> inferred return type
    func_returns: HashMap<String, Ty>,
    // struct/data name -> field names
    structs: HashMap<String, Vec<String>>,
    // known stdlib + builtin function names (arity not enforced, dynamic)
    builtins: Vec<&'static str>,
    // known state-machine names (constructors take no args)
    machines: Vec<String>,
    // global constants -> type
    consts: HashMap<String, Ty>,
    // type alias name -> its target type name (e.g. "Id" -> "Int")
    aliases: HashMap<String, String>,
    // declared trait names, for verifying `where` bounds against real traits
    traits: std::collections::HashSet<String>,
    // (type_name, trait_name) pairs from `impl Trait for Type`
    impls: std::collections::HashSet<(String, String)>,
    // source position of the expression currently being checked, for error locations
    cur_pos: (u32, u32),
    errors: Vec<String>,
    warnings: Vec<String>,
    // identifiers referenced in the current function (for unused-var warnings)
    used: std::collections::HashSet<String>,
}

impl Checker {
    pub fn new(program: &Program) -> Self {
        let mut funcs = HashMap::new();
        let mut sigs = HashMap::new();
        let mut structs = HashMap::new();
        let mut machines = Vec::new();
        let consts = HashMap::new();
        let mut aliases = HashMap::new();
        let mut traits = std::collections::HashSet::new();
        let mut impls = std::collections::HashSet::new();
        for item in &program.items {
            match item {
                Item::TypeAlias { name, target, .. } => {
                    if !target.is_empty() { aliases.insert(name.clone(), target.clone()); }
                }
                Item::Func(f) => {
                    funcs.insert(f.name.clone(), f.params.len());
                    sigs.insert(f.name.clone(), FnSig {
                        param_types: f.param_types.clone(),
                        ret_type: f.ret_type.clone(),
                        type_params: f.type_params.clone(),
                        where_bounds: f.where_bounds.clone(),
                        effects: f.effects.clone(),
                    });
                }
                Item::Struct(s) => { structs.insert(s.name.clone(), s.fields.clone()); }
                Item::Machine(m) => machines.push(m.name.clone()),
                Item::Trait(t) => { traits.insert(t.name.clone()); }
                Item::Impl(imp) => {
                    if let Some(tr) = &imp.trait_name {
                        impls.insert((imp.type_name.clone(), tr.clone()));
                    }
                }
                Item::Extern(fns) => {
                    // foreign functions are known to the checker; arity-check the
                    // fixed-arity ones (variadic signatures stay dynamic)
                    for ef in fns {
                        if !ef.variadic { funcs.insert(ef.name.clone(), ef.arity); }
                    }
                }
                _ => {}
            }
        }
        Checker {
            funcs,
            sigs,
            func_returns: HashMap::new(),
            structs,
            builtins: builtin_names(),
            machines,
            consts,
            aliases,
            traits,
            impls,
            cur_pos: (0, 0),
            errors: Vec::new(),
            warnings: Vec::new(),
            used: std::collections::HashSet::new(),
        }
    }

    // Check the whole program; returns (type errors, warnings). Empty errors = OK.
    pub fn check(mut self, program: &Program) -> (Vec<String>, Vec<String>) {
        // seed constant types first so functions can reference them
        for item in &program.items {
            if let Item::Const { name, value } = item {
                let mut empty = HashMap::new();
                let t = self.infer(value, &mut empty);
                self.consts.insert(name.clone(), t);
            }
        }
        // PASS 1: infer each function's return type so call sites can use it.
        // Iterate to a fixed point (bounded) so mutually-recursive functions settle.
        for _ in 0..4 {
            let mut changed = false;
            let funcs: Vec<&Func> = program.items.iter().filter_map(|i| match i {
                Item::Func(f) => Some(f),
                _ => None,
            }).collect();
            for f in funcs {
                let rt = self.infer_func_return(f);
                let prev = self.func_returns.get(&f.name).cloned();
                if prev.as_ref() != Some(&rt) {
                    self.func_returns.insert(f.name.clone(), rt);
                    changed = true;
                }
            }
            if !changed { break; }
        }
        // PASS 2: type-check every function/method/test body for real.
        for item in &program.items {
            match item {
                Item::Func(f) => { self.check_func(f); self.check_effects(f); self.check_moves(f); self.check_zero_alloc(f); }
                Item::Impl(imp) => {
                    for m in &imp.methods {
                        self.check_func(m);
                        self.check_effects(m);
                        self.check_moves(m);
                    }
                }
                Item::Test(t) => {
                    let mut scope = HashMap::new();
                    self.check_block(&t.body, &mut scope);
                }
                _ => {}
            }
        }
        (self.errors, self.warnings)
    }


    // Resolve a written type name to a Ty, following type aliases first.
    // `type Id = Int` lets a parameter typed `Id` check as `Int`.
    fn resolve_ty(&self, name: &str) -> Option<Ty> {
        let mut cur = name;
        // follow the alias chain, bounded to avoid cycles (type A = B; type B = A)
        for _ in 0..64 {
            match self.aliases.get(cur) {
                Some(target) => cur = target,
                None => break,
            }
        }
        ty_from_name(cur)
    }

    // Infer a function's return type from its `return` statements and tail value,
    // without emitting errors (that happens in pass 2). Unifies all exit points.
    fn infer_func_return(&mut self, f: &Func) -> Ty {
        // if a concrete return type is declared, trust it
        if let Some(rt) = &f.ret_type {
            if !f.type_params.contains(rt) {
                if let Some(t) = self.resolve_ty(rt) { return t; }
            }
        }
        let mut scope: HashMap<String, Ty> = HashMap::new();
        scope.insert("self".into(), Ty::Unknown);
        for (i, p) in f.params.iter().enumerate() {
            let ty = f.param_types.get(i).and_then(|o| o.as_ref())
                .and_then(|name| if f.type_params.contains(name) { None } else { self.resolve_ty(name) })
                .unwrap_or(Ty::Unknown);
            scope.insert(p.clone(), ty);
        }
        let saved_err = self.errors.len();
        let saved_warn = self.warnings.len();
        let mut ret = self.block_return_type(&f.body, &mut scope);
        // an implicit trailing expression is the return value
        if let Some(Stmt::Expr(e)) = f.body.last() {
            let t = self.infer(e, &mut scope);
            ret = unify(ret, t);
        }
        // discard any errors/warnings produced during inference; pass 2 owns those
        self.errors.truncate(saved_err);
        self.warnings.truncate(saved_warn);
        ret
    }

    // Collect the unified type of all `return` statements in a block.
    fn block_return_type(&mut self, stmts: &[Stmt], scope: &mut HashMap<String, Ty>) -> Ty {
        let mut acc = Ty::Unknown;
        let mut seen = false;
        for stmt in stmts {
            match stmt {
                Stmt::Return(Some(e)) => {
                    let t = self.infer(e, scope);
                    acc = if seen { unify(acc, t) } else { t };
                    seen = true;
                }
                Stmt::Return(None) => { acc = if seen { unify(acc, Ty::Null) } else { Ty::Null }; seen = true; }
                Stmt::If { then, els, .. } => {
                    let t1 = self.block_return_type(then, scope);
                    if t1 != Ty::Unknown { acc = if seen { unify(acc, t1.clone()) } else { t1 }; seen = true; }
                    if let Some(e) = els {
                        let t2 = self.block_return_type(e, scope);
                        if t2 != Ty::Unknown { acc = if seen { unify(acc, t2.clone()) } else { t2 }; seen = true; }
                    }
                }
                Stmt::While { body, .. } | Stmt::ForRange { body, .. } | Stmt::ForEach { body, .. } => {
                    let t = self.block_return_type(body, scope);
                    if t != Ty::Unknown { acc = if seen { unify(acc, t.clone()) } else { t }; seen = true; }
                }
                _ => {}
            }
        }
        acc
    }

    fn check_func(&mut self, f: &Func) {
        let mut scope: HashMap<String, Ty> = HashMap::new();
        // `self` is available inside methods; params start Unknown (gradual)
        scope.insert("self".into(), Ty::Unknown);
        for (i, p) in f.params.iter().enumerate() {
            // use the declared type when concrete; generic params stay Unknown
            let ty = f.param_types.get(i).and_then(|o| o.as_ref())
                .and_then(|name| {
                    if f.type_params.contains(name) { None } else { self.resolve_ty(name) }
                })
                .unwrap_or(Ty::Unknown);
            scope.insert(p.clone(), ty);
        }
        self.used.clear();
        let mut declared: Vec<String> = Vec::new();
        self.collect_declared(&f.body, &mut declared);
        self.check_block(&f.body, &mut scope);
        // warn about let-bound names that are never read (params excluded — they
        // are part of the signature and often intentionally unused)
        for name in &declared {
            if !self.used.contains(name) && !name.starts_with('_') {
                self.warnings.push(format!(
                    "in `{}`: variable `{}` is assigned but never used", f.name, name
                ));
            }
        }
    }

    // Gather names introduced by `let` in a body (descending into nested blocks).
    fn collect_declared(&self, stmts: &[Stmt], out: &mut Vec<String>) {
        for stmt in stmts {
            match stmt {
                Stmt::Let { name, .. } => { if !out.contains(name) { out.push(name.clone()); } }
                Stmt::If { then, els, .. } => {
                    self.collect_declared(then, out);
                    if let Some(e) = els { self.collect_declared(e, out); }
                }
                Stmt::While { body, .. } | Stmt::ForRange { body, .. } | Stmt::ForEach { body, .. } => {
                    self.collect_declared(body, out);
                }
                Stmt::TryCatch { body, catch_body, finally_body, .. } => {
                    self.collect_declared(body, out);
                    if let Some(c) = catch_body { self.collect_declared(c, out); }
                    if let Some(fb) = finally_body { self.collect_declared(fb, out); }
                }
                _ => {}
            }
        }
    }

    fn check_block(&mut self, stmts: &[Stmt], scope: &mut HashMap<String, Ty>) {
        for stmt in stmts {
            self.check_stmt(stmt, scope);
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt, scope: &mut HashMap<String, Ty>) {
        match stmt {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let t = self.infer(value, scope);
                scope.insert(name.clone(), t);
            }
            Stmt::IndexAssign { base, index, value } => {
                self.infer(base, scope);
                self.infer(index, scope);
                self.infer(value, scope);
            }
            Stmt::FieldAssign { base, value, .. } => {
                self.infer(base, scope);
                self.infer(value, scope);
            }
            Stmt::Expr(e) => { self.infer(e, scope); }
            Stmt::Return(Some(e)) => { self.infer(e, scope); }
            Stmt::Return(None) => {}
            Stmt::Throw(e) => { self.infer(e, scope); }
            Stmt::Yield(Some(e)) => { self.infer(e, scope); }
            Stmt::Yield(None) => {}
            Stmt::Break(Some(e)) => { self.infer(e, scope); }
            Stmt::Break(None) | Stmt::Continue => {}
            Stmt::If { cond, then, els } => {
                let ct = self.infer(cond, scope);
                self.expect_bool(&ct, "if condition");
                let mut s1 = scope.clone();
                self.check_block(then, &mut s1);
                if let Some(e) = els {
                    let mut s2 = scope.clone();
                    self.check_block(e, &mut s2);
                }
            }
            Stmt::While { cond, body } => {
                let ct = self.infer(cond, scope);
                self.expect_bool(&ct, "while condition");
                let mut s = scope.clone();
                self.check_block(body, &mut s);
            }
            Stmt::ForRange { var, start, end, body, .. } => {
                let st = self.infer(start, scope);
                let et = self.infer(end, scope);
                if !st.is_num() { self.err(format!("for-range start must be a number, found {}", st.name())); }
                if !et.is_num() { self.err(format!("for-range end must be a number, found {}", et.name())); }
                let mut s = scope.clone();
                s.insert(var.clone(), Ty::Int);
                self.check_block(body, &mut s);
            }
            Stmt::ForEach { var, iter, body } => {
                self.infer(iter, scope);
                let mut s = scope.clone();
                s.insert(var.clone(), Ty::Unknown);
                self.check_block(body, &mut s);
            }
            Stmt::Defer(body) => {
                let mut s = scope.clone();
                self.check_block(body, &mut s);
            }
            Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
                let mut s1 = scope.clone();
                self.check_block(body, &mut s1);
                if let Some(cb) = catch_body {
                    let mut s2 = scope.clone();
                    if let Some(v) = catch_var { s2.insert(v.clone(), Ty::Unknown); }
                    self.check_block(cb, &mut s2);
                }
                if let Some(fb) = finally_body {
                    let mut s3 = scope.clone();
                    self.check_block(fb, &mut s3);
                }
            }
        }
    }

    fn infer(&mut self, e: &Expr, scope: &mut HashMap<String, Ty>) -> Ty {
        match e {
            Expr::At { pos, expr } => {
                // remember the position so any error in this expression is located
                self.cur_pos = *pos;
                self.infer(expr, scope)
            }
            Expr::Int(_) => Ty::Int,
            Expr::BigIntLit(_) => Ty::Int,
            Expr::Float(_) => Ty::Float,
            Expr::Str(_) => Ty::Str,
            Expr::Bool(_) => Ty::Bool,
            Expr::Null => Ty::Null,
            Expr::FmtStr(parts) => {
                for p in parts {
                    if let FmtPart::Expr(ex) = p { self.infer(ex, scope); }
                }
                Ty::Str
            }
            Expr::Array(items) => {
                let mut elem = Ty::Unknown;
                let mut seen = false;
                for it in items {
                    let t = self.infer(it, scope);
                    elem = if seen { unify(elem, t) } else { t };
                    seen = true;
                }
                Ty::arr(elem)
            }
            Expr::MapLit(entries) => {
                for (k, v) in entries { self.infer(k, scope); self.infer(v, scope); }
                Ty::Map
            }
            Expr::SetLit(items) => {
                for it in items { self.infer(it, scope); }
                Ty::Map
            }
            Expr::Comprehension { body, var, iter, cond } => {
                self.infer(iter, scope);
                let mut s = scope.clone();
                s.insert(var.clone(), Ty::Unknown);
                if let Some(c) = cond { self.infer(c, &mut s); }
                let bt = self.infer(body, &mut s);
                Ty::arr(bt)
            }
            Expr::Ident(name) => {
                self.used.insert(name.clone());
                if let Some(t) = scope.get(name) { return t.clone(); }
                if let Some(t) = self.consts.get(name) { return t.clone(); }
                // a bare function name used as a value, or a unit enum variant
                if self.funcs.contains_key(name) || self.builtins.contains(&name.as_str()) {
                    return Ty::Func;
                }
                // unit enum variants (None, Nil, etc.) start with uppercase — be lenient
                if name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    return Ty::Unknown;
                }
                // names starting with `_` are context-injected or intentionally
                // special (e.g. `_recv` inside a select arm) — don't flag them
                if name.starts_with('_') {
                    return Ty::Unknown;
                }
                self.err(format!("undefined variable: {}", name));
                Ty::Unknown
            }
            Expr::Index { base, index } => {
                let bt = self.infer(base, scope);
                // a[lo..hi] slices: result type matches the base collection
                if let Expr::RangeLit { lo, hi, .. } = &**index {
                    if let Some(e) = lo { self.infer(e, scope); }
                    if let Some(e) = hi { self.infer(e, scope); }
                    // slicing preserves the collection type (element type kept)
                    return match bt { Ty::Array(_) => bt, Ty::Str => Ty::Str, _ => Ty::Unknown };
                }
                self.infer(index, scope);
                // element access: `xs[i]` on `[T]` yields `T`; on a string yields Str
                match bt {
                    Ty::Array(e) => *e,
                    Ty::Str => Ty::Str,
                    _ => Ty::Unknown,
                }
            }
            Expr::RangeLit { lo, hi, .. } => {
                if let Some(e) = lo { self.infer(e, scope); }
                if let Some(e) = hi { self.infer(e, scope); }
                Ty::arr(Ty::Int)
            }
            Expr::StructLit { name, fields } => {
                if let Some(decl_fields) = self.structs.get(name).cloned() {
                    for (fname, fexpr) in fields {
                        self.infer(fexpr, scope);
                        if !decl_fields.contains(fname) {
                            self.err(format!("struct {} has no field `{}`", name, fname));
                        }
                    }
                } else {
                    for (_, fexpr) in fields { self.infer(fexpr, scope); }
                }
                Ty::Struct(name.clone())
            }
            Expr::Field { base, field } => {
                let bt = self.infer(base, scope);
                if let Ty::Struct(sname) = &bt {
                    if let Some(decl) = self.structs.get(sname) {
                        if !decl.contains(field) && field != "state" {
                            self.err(format!("struct {} has no field `{}`", sname, field));
                        }
                    }
                }
                Ty::Unknown
            }
            Expr::SafeField { base, .. } => {
                self.infer(base, scope);
                Ty::Unknown
            }
            Expr::MethodCall { base, args, .. } => {
                self.infer(base, scope);
                for a in args { self.infer(a, scope); }
                Ty::Unknown
            }
            Expr::Lambda { params, body } => {
                let mut s = scope.clone();
                for p in params { s.insert(p.clone(), Ty::Unknown); }
                match &**body {
                    LambdaBody::Expr(e) => { self.infer(e, &mut s); }
                    LambdaBody::Block(stmts) => { self.check_block(stmts, &mut s); }
                }
                Ty::Func
            }
            Expr::CallValue { callee, args } => {
                self.infer(callee, scope);
                for a in args { self.infer(a, scope); }
                Ty::Unknown
            }
            Expr::Unary { op, expr } => {
                let t = self.infer(expr, scope);
                match op {
                    UnOp::Neg => {
                        if !t.is_num() { self.err(format!("cannot negate {}", t.name())); }
                        t
                    }
                    UnOp::Not => Ty::Bool,
                    UnOp::BitNot => {
                        if !t.is_num() { self.err(format!("cannot apply `~` to {}", t.name())); }
                        Ty::Int
                    }
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let lt = self.infer(lhs, scope);
                let rt = self.infer(rhs, scope);
                self.check_binop(*op, &lt, &rt)
            }
            Expr::Call { callee, args } => {
                // infer each argument once (re-used for bounds and generic return)
                let arg_tys: Vec<Ty> = args.iter().map(|a| self.infer(a, scope)).collect();
                // arity check for user functions (builtins are dynamic)
                if let Some(arity) = self.funcs.get(callee).copied() {
                    if arity != args.len() {
                        self.err(format!(
                            "function `{}` expects {} argument(s), got {}",
                            callee, arity, args.len()
                        ));
                    }
                    if let Some(sig) = self.sigs.get(callee).cloned() {
                        // argument types: a declared, concrete (non-generic) parameter
                        // type must accept the argument. Stays gradual — an Unknown on
                        // either side (dynamic value, unannotated param) is allowed, as
                        // are numeric widenings (Int↔Float), so only genuine mismatches
                        // like passing a Str where an Int is required are reported.
                        for (i, pt) in sig.param_types.iter().enumerate() {
                            let Some(decl) = pt else { continue };
                            if sig.type_params.contains(decl) { continue; }
                            let Some(expected) = self.resolve_ty(decl) else { continue };
                            if let Some(actual) = arg_tys.get(i) {
                                if !actual.compatible(&expected) {
                                    self.err(format!(
                                        "argument {} to `{}` expects {}, found {}",
                                        i + 1, callee, expected.name(), actual.name()
                                    ));
                                }
                            }
                        }
                        // trait bounds: when an argument bound to a generic param has a
                        // known concrete type, that type must implement the required
                        // traits (`where T: Trait` / `[T: Trait]`). Unknown args are
                        // left alone — the checker stays gradual.
                        for (gen, req_traits) in &sig.where_bounds {
                            for (i, pt) in sig.param_types.iter().enumerate() {
                                if pt.as_deref() != Some(gen.as_str()) { continue; }
                                if let Some(Ty::Struct(tn)) = arg_tys.get(i) {
                                    for tr in req_traits {
                                        if self.traits.contains(tr)
                                            && !self.impls.contains(&(tn.clone(), tr.clone()))
                                        {
                                            self.err(format!(
                                                "type `{}` does not satisfy bound `{}: {}` (no `impl {} for {}`)",
                                                tn, gen, tr, tr, tn
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        // generic substitution: if the return type names a generic
                        // parameter (e.g. `-> T`), resolve it from the argument bound
                        // to that same parameter. e.g. id[T](x: T) -> T applied to Int
                        // yields Int.
                        if let Some(ret) = &sig.ret_type {
                            if sig.type_params.contains(ret) {
                                for (i, pt) in sig.param_types.iter().enumerate() {
                                    if pt.as_deref() == Some(ret.as_str()) {
                                        if let Some(t) = arg_tys.get(i) {
                                            return t.clone();
                                        }
                                    }
                                }
                            } else if let Some(concrete) = self.resolve_ty(ret) {
                                // a concrete declared return type
                                return concrete;
                            }
                        }
                    }
                    // otherwise fall back to the inferred return type
                    return self.func_returns.get(callee).cloned().unwrap_or(Ty::Unknown);
                }
                if self.machines.contains(callee) {
                    return Ty::Struct(callee.clone());
                }
                if self.structs.contains_key(callee) {
                    return Ty::Struct(callee.clone());
                }
                // enum variant constructor or stdlib: lenient
                Ty::Unknown
            }
            Expr::Block { stmts, tail } => {
                let mut s = scope.clone();
                self.check_block(stmts, &mut s);
                match tail {
                    Some(e) => self.infer(e, &mut s),
                    None => Ty::Null,
                }
            }
            Expr::If { cond, then, els } => {
                let ct = self.infer(cond, scope);
                self.expect_bool(&ct, "if expression condition");
                let tt = self.infer(then, scope);
                let et = self.infer(els, scope);
                if tt.compatible(&et) { tt } else { Ty::Unknown }
            }
            Expr::Match { scrutinee, arms } => {
                self.infer(scrutinee, scope);
                for arm in arms {
                    let mut s = scope.clone();
                    bind_pattern(&arm.pattern, &mut s);
                    if let Some(g) = &arm.guard { self.infer(g, &mut s); }
                    self.infer(&arm.body, &mut s);
                }
                Ty::Unknown
            }
            // --- async / concurrency ---
            // The gradual checker doesn't model Future/Task/Channel types yet,
            // so it walks their sub-expressions (to catch ordinary errors inside)
            // and yields Unknown, which unifies with everything.
            Expr::Await(inner) => {
                self.infer(inner, scope);
                Ty::Unknown
            }
            Expr::Spawn(stmts) => {
                let mut s = scope.clone();
                self.check_block(stmts, &mut s);
                Ty::Unknown
            }
            Expr::Recv(chan) => {
                self.infer(chan, scope);
                Ty::Unknown
            }
            Expr::Send { chan, value } => {
                self.infer(chan, scope);
                self.infer(value, scope);
                Ty::Null
            }
            Expr::Select(arms) => {
                for arm in arms {
                    self.infer(&arm.chan, scope);
                    let mut s = scope.clone();
                    if let Some(name) = &arm.binding {
                        s.insert(name.clone(), Ty::Unknown);
                    }
                    self.infer(&arm.body, &mut s);
                }
                Ty::Unknown
            }
        }
    }

    fn check_binop(&mut self, op: BinOp, lt: &Ty, rt: &Ty) -> Ty {
        use BinOp::*;
        match op {
            Add => {
                // + works on numbers and on strings (concat); arrays too in Nova
                if matches!(lt, Ty::Str) || matches!(rt, Ty::Str) { return Ty::Str; }
                if lt.is_num() && rt.is_num() {
                    return if *lt == Ty::Float || *rt == Ty::Float { Ty::Float } else { Ty::Int };
                }
                if matches!(lt, Ty::Unknown) || matches!(rt, Ty::Unknown) { return Ty::Unknown; }
                if let (Ty::Array(a), Ty::Array(b)) = (lt, rt) {
                    return Ty::arr(unify((**a).clone(), (**b).clone()));
                }
                self.err(format!("cannot apply `+` to {} and {}", lt.name(), rt.name()));
                Ty::Unknown
            }
            Sub | Mul | Div | Rem | Pow => {
                if !lt.is_num() || !rt.is_num() {
                    self.err(format!("operator `{}` requires numbers, found {} and {}",
                        binop_sym(op), lt.name(), rt.name()));
                    return Ty::Unknown;
                }
                if *lt == Ty::Float || *rt == Ty::Float || matches!(op, Div | Pow) { Ty::Float } else { Ty::Int }
            }
            Eq | Ne => Ty::Bool,
            Lt | Le | Gt | Ge => {
                if !lt.compatible(rt) {
                    self.err(format!("cannot compare {} with {}", lt.name(), rt.name()));
                }
                Ty::Bool
            }
            And | Or => Ty::Bool,
            BitOr | BitXor | BitAnd | Shl | Shr => {
                if !lt.is_num() || !rt.is_num() {
                    self.err(format!("bitwise operator requires integers, found {} and {}", lt.name(), rt.name()));
                }
                Ty::Int
            }
        }
    }

    fn expect_bool(&mut self, t: &Ty, ctx: &str) {
        // gradual: Unknown is fine; only flag a definitely-wrong type
        if !matches!(t, Ty::Bool | Ty::Unknown) {
            self.err(format!("{} must be Bool, found {}", ctx, t.name()));
        }
    }

    // Verify a function's declared effects. Unannotated functions are skipped
    // (gradual). A function that performs an effect outside its `![..]` set — via
    // an effectful builtin or a call to a function that declares that effect — is
    // a located error. Each missing effect is reported once, at its first site.
    fn check_effects(&mut self, f: &Func) {
        let declared = match &f.effects {
            Some(d) => d.clone(),
            None => return, // unannotated: not effect-checked
        };
        let mut found: Vec<(String, (u32, u32))> = Vec::new();
        for s in &f.body {
            self.collect_stmt_effects(s, (0, 0), &mut found);
        }
        let mut reported: Vec<String> = Vec::new();
        for (eff, pos) in found {
            if !declared.iter().any(|d| d == &eff) && !reported.contains(&eff) {
                reported.push(eff.clone());
                let saved = self.cur_pos;
                self.cur_pos = pos;
                self.err(format!(
                    "function `{}` performs effect `{}` not in its declared effects ![{}]",
                    f.name, eff, declared.join(", ")
                ));
                self.cur_pos = saved;
            }
        }
    }

    fn collect_stmt_effects(&self, s: &Stmt, pos: (u32, u32), out: &mut Vec<(String, (u32, u32))>) {
        match s {
            Stmt::Let { value, .. } | Stmt::Assign { value, .. } => self.collect_expr_effects(value, pos, out),
            Stmt::IndexAssign { base, index, value } => {
                self.collect_expr_effects(base, pos, out);
                self.collect_expr_effects(index, pos, out);
                self.collect_expr_effects(value, pos, out);
            }
            Stmt::FieldAssign { base, value, .. } => {
                self.collect_expr_effects(base, pos, out);
                self.collect_expr_effects(value, pos, out);
            }
            Stmt::Expr(e) | Stmt::Throw(e) => self.collect_expr_effects(e, pos, out),
            Stmt::Return(opt) | Stmt::Break(opt) | Stmt::Yield(opt) => {
                if let Some(e) = opt { self.collect_expr_effects(e, pos, out); }
            }
            Stmt::If { cond, then, els } => {
                self.collect_expr_effects(cond, pos, out);
                for s in then { self.collect_stmt_effects(s, pos, out); }
                if let Some(e) = els { for s in e { self.collect_stmt_effects(s, pos, out); } }
            }
            Stmt::While { cond, body } => {
                self.collect_expr_effects(cond, pos, out);
                for s in body { self.collect_stmt_effects(s, pos, out); }
            }
            Stmt::ForRange { start, end, body, .. } => {
                self.collect_expr_effects(start, pos, out);
                self.collect_expr_effects(end, pos, out);
                for s in body { self.collect_stmt_effects(s, pos, out); }
            }
            Stmt::ForEach { iter, body, .. } => {
                self.collect_expr_effects(iter, pos, out);
                for s in body { self.collect_stmt_effects(s, pos, out); }
            }
            Stmt::Defer(b) => { for s in b { self.collect_stmt_effects(s, pos, out); } }
            Stmt::TryCatch { body, catch_body, finally_body, .. } => {
                for s in body { self.collect_stmt_effects(s, pos, out); }
                if let Some(b) = catch_body { for s in b { self.collect_stmt_effects(s, pos, out); } }
                if let Some(b) = finally_body { for s in b { self.collect_stmt_effects(s, pos, out); } }
            }
            Stmt::Continue => {}
        }
    }

    // Walk an expression, recording the effects it performs. Does not descend into
    // lambda/spawn bodies (their effects occur where they are eventually called).
    fn collect_expr_effects(&self, e: &Expr, pos: (u32, u32), out: &mut Vec<(String, (u32, u32))>) {
        match e {
            Expr::At { pos: p, expr } => self.collect_expr_effects(expr, *p, out),
            Expr::Call { callee, args } => {
                if let Some(eff) = builtin_effect(callee) { out.push((eff.to_string(), pos)); }
                if let Some(sig) = self.sigs.get(callee) {
                    if let Some(effs) = &sig.effects {
                        for ef in effs { out.push((ef.clone(), pos)); }
                    }
                }
                for a in args { self.collect_expr_effects(a, pos, out); }
            }
            Expr::CallValue { callee, args } => {
                self.collect_expr_effects(callee, pos, out);
                for a in args { self.collect_expr_effects(a, pos, out); }
            }
            Expr::MethodCall { base, args, .. } => {
                self.collect_expr_effects(base, pos, out);
                for a in args { self.collect_expr_effects(a, pos, out); }
            }
            Expr::Unary { expr, .. } => self.collect_expr_effects(expr, pos, out),
            Expr::Binary { lhs, rhs, .. } => {
                self.collect_expr_effects(lhs, pos, out);
                self.collect_expr_effects(rhs, pos, out);
            }
            Expr::Index { base, index } => {
                self.collect_expr_effects(base, pos, out);
                self.collect_expr_effects(index, pos, out);
            }
            Expr::RangeLit { lo, hi, .. } => {
                if let Some(l) = lo { self.collect_expr_effects(l, pos, out); }
                if let Some(h) = hi { self.collect_expr_effects(h, pos, out); }
            }
            Expr::Field { base, .. } | Expr::SafeField { base, .. } => self.collect_expr_effects(base, pos, out),
            Expr::Array(xs) | Expr::SetLit(xs) => { for x in xs { self.collect_expr_effects(x, pos, out); } }
            Expr::MapLit(pairs) => {
                for (k, v) in pairs { self.collect_expr_effects(k, pos, out); self.collect_expr_effects(v, pos, out); }
            }
            Expr::StructLit { fields, .. } => { for (_, v) in fields { self.collect_expr_effects(v, pos, out); } }
            Expr::FmtStr(parts) => {
                for p in parts { if let FmtPart::Expr(ex) = p { self.collect_expr_effects(ex, pos, out); } }
            }
            Expr::Comprehension { body, iter, cond, .. } => {
                self.collect_expr_effects(body, pos, out);
                self.collect_expr_effects(iter, pos, out);
                if let Some(c) = cond { self.collect_expr_effects(c, pos, out); }
            }
            Expr::Block { stmts, tail } => {
                for s in stmts { self.collect_stmt_effects(s, pos, out); }
                if let Some(t) = tail { self.collect_expr_effects(t, pos, out); }
            }
            Expr::If { cond, then, els } => {
                self.collect_expr_effects(cond, pos, out);
                self.collect_expr_effects(then, pos, out);
                self.collect_expr_effects(els, pos, out);
            }
            Expr::Match { scrutinee, arms } => {
                self.collect_expr_effects(scrutinee, pos, out);
                for a in arms {
                    if let Some(g) = &a.guard { self.collect_expr_effects(g, pos, out); }
                    self.collect_expr_effects(&a.body, pos, out);
                }
            }
            Expr::Await(x) | Expr::Recv(x) => self.collect_expr_effects(x, pos, out),
            Expr::Send { chan, value } => {
                self.collect_expr_effects(chan, pos, out);
                self.collect_expr_effects(value, pos, out);
            }
            // leaves and scope-introducing forms (lambda/spawn) contribute nothing here
            _ => {}
        }
    }

    // Ownership / move checking for `linear` and `affine` parameters. A linear
    // value must be consumed exactly once; an affine value at most once. Using a
    // value after it is moved — or inside a loop, where it could run more than
    // once — is a located error. Only functions with such params are checked.
    // `#[zero_alloc]`: a static guarantee that the function performs no heap
    // allocation. Any array/map/set/struct literal, comprehension, f-string,
    // closure, or string concatenation is a violation, reported by `nova check`.
    fn check_zero_alloc(&mut self, f: &Func) {
        if !f.attrs.iter().any(|a| a.name == "zero_alloc") { return; }
        let mut bad: Vec<&'static str> = Vec::new();
        for s in &f.body { zero_alloc_stmt(s, &mut bad); }
        for what in bad {
            self.errors.push(format!(
                "#[zero_alloc] function `{}` allocates: {}", f.name, what));
        }
    }

    fn check_moves(&mut self, f: &Func) {
        use std::collections::HashSet;
        let mut linear: HashSet<String> = HashSet::new();
        let mut tracked: HashSet<String> = HashSet::new();
        for (i, p) in f.params.iter().enumerate() {
            match f.param_modes.get(i).and_then(|m| m.as_deref()) {
                Some("linear") => { linear.insert(p.clone()); tracked.insert(p.clone()); }
                Some("affine") => { tracked.insert(p.clone()); }
                _ => {}
            }
        }
        if tracked.is_empty() { return; }
        let mut moved: HashSet<String> = HashSet::new();
        self.moves_block(&f.body, &tracked, &mut moved, false);
        for v in &linear {
            if !moved.contains(v) {
                self.cur_pos = (0, 0);
                self.err(format!("linear value `{}` is never used (it must be consumed once)", v));
            }
        }
    }

    fn moves_block(&mut self, stmts: &[Stmt], tracked: &std::collections::HashSet<String>,
                   moved: &mut std::collections::HashSet<String>, in_loop: bool) {
        for s in stmts { self.moves_stmt(s, tracked, moved, in_loop); }
    }

    fn moves_stmt(&mut self, s: &Stmt, tracked: &std::collections::HashSet<String>,
                  moved: &mut std::collections::HashSet<String>, in_loop: bool) {
        match s {
            Stmt::Let { value, .. } | Stmt::Assign { value, .. } => self.moves_expr(value, tracked, moved, in_loop),
            Stmt::IndexAssign { base, index, value } => {
                self.moves_expr(base, tracked, moved, in_loop);
                self.moves_expr(index, tracked, moved, in_loop);
                self.moves_expr(value, tracked, moved, in_loop);
            }
            Stmt::FieldAssign { base, value, .. } => {
                self.moves_expr(base, tracked, moved, in_loop);
                self.moves_expr(value, tracked, moved, in_loop);
            }
            Stmt::Expr(e) | Stmt::Throw(e) => self.moves_expr(e, tracked, moved, in_loop),
            Stmt::Return(o) | Stmt::Break(o) | Stmt::Yield(o) => {
                if let Some(e) = o { self.moves_expr(e, tracked, moved, in_loop); }
            }
            Stmt::If { cond, then, els } => {
                self.moves_expr(cond, tracked, moved, in_loop);
                let mut a = moved.clone();
                self.moves_block(then, tracked, &mut a, in_loop);
                let mut b = moved.clone();
                if let Some(e) = els { self.moves_block(e, tracked, &mut b, in_loop); }
                for v in a { moved.insert(v); }   // a value moved on either branch
                for v in b { moved.insert(v); }   // is conservatively moved after
            }
            Stmt::While { cond, body } => {
                self.moves_expr(cond, tracked, moved, in_loop);
                self.moves_block(body, tracked, moved, true);
            }
            Stmt::ForRange { start, end, body, .. } => {
                self.moves_expr(start, tracked, moved, in_loop);
                self.moves_expr(end, tracked, moved, in_loop);
                self.moves_block(body, tracked, moved, true);
            }
            Stmt::ForEach { iter, body, .. } => {
                self.moves_expr(iter, tracked, moved, in_loop);
                self.moves_block(body, tracked, moved, true);
            }
            Stmt::Defer(b) => self.moves_block(b, tracked, moved, in_loop),
            Stmt::TryCatch { body, catch_body, finally_body, .. } => {
                self.moves_block(body, tracked, moved, in_loop);
                if let Some(b) = catch_body { self.moves_block(b, tracked, moved, in_loop); }
                if let Some(b) = finally_body { self.moves_block(b, tracked, moved, in_loop); }
            }
            Stmt::Continue => {}
        }
    }

    fn moves_expr(&mut self, e: &Expr, tracked: &std::collections::HashSet<String>,
                  moved: &mut std::collections::HashSet<String>, in_loop: bool) {
        match e {
            Expr::At { pos, expr } => { self.cur_pos = *pos; self.moves_expr(expr, tracked, moved, in_loop); }
            Expr::Ident(name) => {
                if tracked.contains(name) {
                    if in_loop {
                        self.err(format!("linear/affine value `{}` used inside a loop (may run more than once)", name));
                        moved.insert(name.clone()); // counts as consumed: avoid a redundant never-used error
                    } else if moved.contains(name) {
                        self.err(format!("use of moved value `{}`", name));
                    } else {
                        moved.insert(name.clone());
                    }
                }
            }
            Expr::Unary { expr, .. } => self.moves_expr(expr, tracked, moved, in_loop),
            Expr::Binary { lhs, rhs, .. } => {
                self.moves_expr(lhs, tracked, moved, in_loop);
                self.moves_expr(rhs, tracked, moved, in_loop);
            }
            Expr::Index { base, index } => {
                self.moves_expr(base, tracked, moved, in_loop);
                self.moves_expr(index, tracked, moved, in_loop);
            }
            Expr::RangeLit { lo, hi, .. } => {
                if let Some(l) = lo { self.moves_expr(l, tracked, moved, in_loop); }
                if let Some(h) = hi { self.moves_expr(h, tracked, moved, in_loop); }
            }
            Expr::Field { base, .. } | Expr::SafeField { base, .. } => self.moves_expr(base, tracked, moved, in_loop),
            Expr::MethodCall { base, args, .. } => {
                self.moves_expr(base, tracked, moved, in_loop);
                for a in args { self.moves_expr(a, tracked, moved, in_loop); }
            }
            Expr::Call { args, .. } => { for a in args { self.moves_expr(a, tracked, moved, in_loop); } }
            Expr::CallValue { callee, args } => {
                self.moves_expr(callee, tracked, moved, in_loop);
                for a in args { self.moves_expr(a, tracked, moved, in_loop); }
            }
            Expr::Array(xs) | Expr::SetLit(xs) => { for x in xs { self.moves_expr(x, tracked, moved, in_loop); } }
            Expr::MapLit(pairs) => {
                for (k, v) in pairs { self.moves_expr(k, tracked, moved, in_loop); self.moves_expr(v, tracked, moved, in_loop); }
            }
            Expr::StructLit { fields, .. } => { for (_, v) in fields { self.moves_expr(v, tracked, moved, in_loop); } }
            Expr::FmtStr(parts) => { for p in parts { if let FmtPart::Expr(ex) = p { self.moves_expr(ex, tracked, moved, in_loop); } } }
            Expr::Comprehension { body, iter, cond, .. } => {
                // a comprehension iterates, so its body is loop-like
                self.moves_expr(iter, tracked, moved, in_loop);
                self.moves_expr(body, tracked, moved, true);
                if let Some(c) = cond { self.moves_expr(c, tracked, moved, true); }
            }
            Expr::Block { stmts, tail } => {
                self.moves_block(stmts, tracked, moved, in_loop);
                if let Some(t) = tail { self.moves_expr(t, tracked, moved, in_loop); }
            }
            Expr::If { cond, then, els } => {
                self.moves_expr(cond, tracked, moved, in_loop);
                let mut a = moved.clone();
                self.moves_expr(then, tracked, &mut a, in_loop);
                let mut b = moved.clone();
                self.moves_expr(els, tracked, &mut b, in_loop);
                for v in a { moved.insert(v); }
                for v in b { moved.insert(v); }
            }
            Expr::Match { scrutinee, arms } => {
                self.moves_expr(scrutinee, tracked, moved, in_loop);
                let base = moved.clone();
                for arm in arms {
                    let mut m = base.clone();
                    if let Some(g) = &arm.guard { self.moves_expr(g, tracked, &mut m, in_loop); }
                    self.moves_expr(&arm.body, tracked, &mut m, in_loop);
                    for v in m { moved.insert(v); }
                }
            }
            Expr::Await(x) | Expr::Recv(x) => self.moves_expr(x, tracked, moved, in_loop),
            Expr::Send { chan, value } => {
                self.moves_expr(chan, tracked, moved, in_loop);
                self.moves_expr(value, tracked, moved, in_loop);
            }
            // leaves and scope-introducing forms (lambda/spawn) are not walked here
            _ => {}
        }
    }

    fn err(&mut self, msg: String) {
        let (line, col) = self.cur_pos;
        if line == 0 {
            self.errors.push(msg);
        } else {
            self.errors.push(format!("line {}, col {}: {}", line, col, msg));
        }
    }
}

// Merge two types at a control-flow join point. Identical types stay; numbers
// widen to a common numeric type; anything else (or Unknown) yields Unknown.
// Map a declared type name to a checker Ty. Unknown for type variables / unknown.
// Canonical effect performed by an effectful builtin, if any. Pure builtins
// (math, string, array helpers, ...) return None.
fn builtin_effect(name: &str) -> Option<&'static str> {
    match name {
        "print" | "println" | "input" | "read_line" | "read" | "write" => Some("IO"),
        "read_file" | "write_file" | "append_file" | "remove_file" | "file_exists"
        | "list_dir" | "mkdir" | "cwd" | "chdir" | "exec" | "eprint" | "setenv" => Some("IO"),
        "rand" | "random" | "rand_int" | "rand_float" | "rand_range" | "rand_bool" => Some("Rand"),
        "now" | "time" | "time_now" | "clock" | "sleep" | "now_ms" | "sleep_ms" => Some("Time"),
        _ => None,
    }
}

fn ty_from_name(name: &str) -> Option<Ty> {
    let n = name.trim();
    // `[Elem]` array type: resolve the element head (Unknown if unrecognised),
    // so `[Int]` -> Array(Int), `[T]` -> Array(Unknown).
    if let Some(inner) = n.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        return Some(Ty::arr(ty_from_name(inner).unwrap_or(Ty::Unknown)));
    }
    Some(match n {
        "Int" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => Ty::Int,
        "Float" | "f32" | "f64" => Ty::Float,
        "Str" | "String" => Ty::Str,
        "Bool" => Ty::Bool,
        "Null" => Ty::Null,
        "Array" | "List" | "Vec" => Ty::arr_unknown(),
        "Map" | "Dict" => Ty::Map,
        _ => return None,
    })
}

fn unify(a: Ty, b: Ty) -> Ty {
    if a == b { return a; }
    match (&a, &b) {
        (Ty::Int, Ty::Float) | (Ty::Float, Ty::Int) => Ty::Float,
        (Ty::Array(x), Ty::Array(y)) => Ty::arr(unify((**x).clone(), (**y).clone())),
        (Ty::Unknown, _) | (_, Ty::Unknown) => Ty::Unknown,
        _ => Ty::Unknown,
    }
}

fn binop_sym(op: BinOp) -> &'static str {
    use BinOp::*;
    match op {
        Add => "+", Sub => "-", Mul => "*", Div => "/", Rem => "%", Pow => "**",
        Eq => "==", Ne => "!=", Lt => "<", Le => "<=", Gt => ">", Ge => ">=",
        And => "&&", Or => "||",
        BitOr => "|", BitXor => "^", BitAnd => "&", Shl => "<<", Shr => ">>",
    }
}

// Bind the variables a pattern introduces (as Unknown) so the arm body can use them.
fn bind_pattern(p: &Pattern, scope: &mut HashMap<String, Ty>) {
    match p {
        Pattern::Binding(name) => { scope.insert(name.clone(), Ty::Unknown); }
        Pattern::EnumVariant { sub, .. } => {
            for b in sub { bind_pattern(b, scope); }
        }
        Pattern::Or(ps) => { for x in ps { bind_pattern(x, scope); } }
        Pattern::Tuple(ps) => { for x in ps { bind_pattern(x, scope); } }
        Pattern::Struct { fields, .. } => {
            for (_, sub) in fields { bind_pattern(sub, scope); }
        }
        Pattern::Slice { prefix, rest, suffix } => {
            for x in prefix { bind_pattern(x, scope); }
            for x in suffix { bind_pattern(x, scope); }
            if let Some(Some(name)) = rest { scope.insert(name.clone(), Ty::arr_unknown()); }
        }
        _ => {}
    }
}

// Walk a statement/expression looking for heap-allocating constructs, for the
// `#[zero_alloc]` guarantee. Reports each distinct kind found.
fn zero_alloc_stmt(s: &Stmt, bad: &mut Vec<&'static str>) {
    match s {
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Expr(value)
        | Stmt::Return(Some(value)) | Stmt::Throw(value) => zero_alloc_expr(value, bad),
        Stmt::IndexAssign { base, index, value } => {
            zero_alloc_expr(base, bad); zero_alloc_expr(index, bad); zero_alloc_expr(value, bad);
        }
        Stmt::FieldAssign { base, value, .. } => { zero_alloc_expr(base, bad); zero_alloc_expr(value, bad); }
        Stmt::If { cond, then, els } => {
            zero_alloc_expr(cond, bad);
            for s in then { zero_alloc_stmt(s, bad); }
            if let Some(e) = els { for s in e { zero_alloc_stmt(s, bad); } }
        }
        Stmt::While { cond, body } => { zero_alloc_expr(cond, bad); for s in body { zero_alloc_stmt(s, bad); } }
        Stmt::ForRange { start, end, body, .. } => {
            zero_alloc_expr(start, bad); zero_alloc_expr(end, bad);
            for s in body { zero_alloc_stmt(s, bad); }
        }
        Stmt::ForEach { iter, body, .. } => { zero_alloc_expr(iter, bad); for s in body { zero_alloc_stmt(s, bad); } }
        Stmt::TryCatch { body, catch_body, finally_body, .. } => {
            for s in body { zero_alloc_stmt(s, bad); }
            if let Some(b) = catch_body { for s in b { zero_alloc_stmt(s, bad); } }
            if let Some(b) = finally_body { for s in b { zero_alloc_stmt(s, bad); } }
        }
        Stmt::Defer(b) => for s in b { zero_alloc_stmt(s, bad); },
        _ => {}
    }
}

fn note(bad: &mut Vec<&'static str>, what: &'static str) {
    if !bad.contains(&what) { bad.push(what); }
}

fn zero_alloc_expr(e: &Expr, bad: &mut Vec<&'static str>) {
    match e {
        Expr::At { expr, .. } => zero_alloc_expr(expr, bad),
        Expr::Array(xs) | Expr::SetLit(xs) => { note(bad, "array/set literal"); for x in xs { zero_alloc_expr(x, bad); } }
        Expr::MapLit(kv) => { note(bad, "map literal"); for (k, v) in kv { zero_alloc_expr(k, bad); zero_alloc_expr(v, bad); } }
        Expr::StructLit { fields, .. } => { note(bad, "struct literal"); for (_, v) in fields { zero_alloc_expr(v, bad); } }
        Expr::Comprehension { .. } => note(bad, "comprehension"),
        Expr::FmtStr(_) => note(bad, "f-string"),
        Expr::Lambda { .. } => note(bad, "closure"),
        Expr::Binary { op: BinOp::Add, lhs, rhs } => {
            // string concatenation allocates; flag it only when an operand is
            // statically a string (literal / f-string) to avoid false positives
            // on integer addition (the gradual checker can't always know).
            let is_str = |x: &Expr| {
                let mut x = x;
                while let Expr::At { expr, .. } = x { x = expr; }
                matches!(x, Expr::Str(_) | Expr::FmtStr(_))
            };
            if is_str(lhs) || is_str(rhs) { note(bad, "string concat"); }
            zero_alloc_expr(lhs, bad); zero_alloc_expr(rhs, bad);
        }
        Expr::Binary { lhs, rhs, .. } => { zero_alloc_expr(lhs, bad); zero_alloc_expr(rhs, bad); }
        Expr::Unary { expr, .. } => zero_alloc_expr(expr, bad),
        Expr::If { cond, then, els } => { zero_alloc_expr(cond, bad); zero_alloc_expr(then, bad); zero_alloc_expr(els, bad); }
        Expr::Call { args, .. } | Expr::CallValue { args, .. } => for a in args { zero_alloc_expr(a, bad); },
        Expr::Index { base, index } => { zero_alloc_expr(base, bad); zero_alloc_expr(index, bad); }
        Expr::Field { base, .. } | Expr::SafeField { base, .. } => zero_alloc_expr(base, bad),
        Expr::Block { stmts, tail } => {
            for s in stmts { zero_alloc_stmt(s, bad); }
            if let Some(t) = tail { zero_alloc_expr(t, bad); }
        }
        _ => {}
    }
}

fn builtin_names() -> Vec<&'static str> {
    vec![
        "print", "len", "push", "pop", "array", "str", "int", "float", "abs", "sqrt",
        "map", "filter", "reduce", "range", "dict", "map_set", "map_get", "map_has",
        "map_del", "map_len", "map_keys", "map_values", "assert", "assert_eq",
        "assert_ne", "assert_true", "assert_false", "assert_gt", "assert_lt", "assert_contains",
        "send", "state_of", "type_of",
        "args", "env", "read_file", "write_file", "append_file", "file_exists",
        "remove_file", "read_line", "input", "eprint", "exit", "to_int",
        "to_float", "chr", "ord",
        "exec", "list_dir", "mkdir", "cwd", "chdir", "now_ms", "sleep_ms", "setenv",
        "tcp_listen", "tcp_accept", "tcp_connect", "tcp_read", "tcp_write", "tcp_close",
    ]
}

#[cfg(test)]
mod generics_tests {
    use super::*;
    use crate::parser::parse_program;

    fn errors(src: &str) -> Vec<String> {
        let prog = parse_program(src).expect("parse");
        Checker::new(&prog).check(&prog).0
    }

    #[test]
    fn array_element_param_not_false_flagged() {
        // `[Int]` param used to resolve to head `Int`, wrongly rejecting an array
        // argument. Now `[Int]` -> Array(Int), so this is clean.
        let errs = errors("fn first(xs: [Int]) -> Int { xs[0] }\nfn main(){ print(first([1,2,3])) }");
        assert!(errs.is_empty(), "no false positive expected, got: {:?}", errs);
    }

    #[test]
    fn element_type_flows_through_index() {
        // element type of `[Str]` is Str; `names[0] * 2` must be a type error.
        let errs = errors("fn main(){ let names = [\"a\",\"b\"]; let x = names[0] * 2; print(x) }");
        assert!(errs.iter().any(|e| e.contains("requires numbers") && e.contains("Str")),
            "element-typed error expected, got: {:?}", errs);
    }

    #[test]
    fn homogeneous_array_keeps_element_gradual() {
        // a mixed/empty array degrades to Unknown element (no false errors).
        assert!(errors("fn main(){ let xs = []; let y = xs[0] + 1; print(y) }").is_empty());
        assert!(errors("fn main(){ let xs = [1, 2.0]; print(xs[0]) }").is_empty());
    }
}
