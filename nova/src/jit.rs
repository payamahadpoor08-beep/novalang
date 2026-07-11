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
use crate::ast::*;

// The cranelift native-code half lives in the `cl` submodule, gated behind the
// `jit` feature. Everything above this point (the eligibility/kind analysis:
// `eligible_set`, `float_eligible_set`, `numeric_kinds`, `FKind`, `array_vars`,
// …) is pure Rust with no cranelift dependency and is also consumed by `aot.rs`,
// so it stays compiled on every arch. When the feature is off (e.g. a 32-bit-ARM
// build where cranelift can't target the host) the whole `cl` module — `Jit`,
// `TieredJit`, and their codegen — simply isn't compiled, and the VM runs pure
// bytecode. `Jit` and `TieredJit` are re-exported so callers see `jit::Jit` etc.
#[cfg(feature = "jit")]
pub use cl::{Jit, TieredJit, compile_object, NativeTarget};

// functions with more parameters than this stay on the VM (keeps the call ABI
// dispatch in `raw_call` finite)
const MAX_ARITY: usize = 8;

// The JIT's runtime helpers (the `nova_arr_*` local-array arena, `nova_fmod`,
// `nova_fpow`) are cranelift-linked and live inside the `cl` module below, so
// they aren't compiled when the `jit` feature is off.

// ---------------------------------------------------------------------------
// Eligibility (a fixpoint over the call graph)
// ---------------------------------------------------------------------------

// The set of function names that can be JIT-compiled: each is structurally
// integer-pure, returns an integer, has arity <= MAX_ARITY, and every function
// it calls is itself eligible.
pub fn eligible_set(prog: &Program) -> HashSet<String> {
    let mut funcs: HashMap<&str, &Func> = HashMap::new();
    let mut defs: HashMap<String, Vec<String>> = HashMap::new();
    for item in &prog.items {
        match item {
            Item::Func(f) => { funcs.insert(&f.name, f); }
            Item::Struct(sd) => { defs.insert(sd.name.clone(), sd.fields.clone()); }
            _ => {}
        }
    }
    // start from everything structurally OK (calls allowed to anything for now)
    let mut set: HashSet<String> = funcs.values()
        .filter(|f| f.params.len() <= MAX_ARITY && locally_ok(f, &defs))
        .map(|f| f.name.clone())
        .collect();
    // remove any function that calls a name outside the set, until stable
    loop {
        let mut remove = None;
        for name in &set {
            let f = funcs[name.as_str()];
            let arrays = array_vars(f);
            let structs = struct_vars(f, &defs);
            if collect_real_calls(&f.body, &arrays, &structs).iter().any(|c| !set.contains(c)) {
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
fn locally_ok(f: &Func, defs: &HashMap<String, Vec<String>>) -> bool {
    let arrays = array_vars(f);
    let structs = struct_vars(f, defs);
    !f.body.is_empty()
        && f.body.iter().all(|s| stmt_pure(s, &arrays, &structs))
        && always_returns(&f.body)
}

// ---------------------------------------------------------------------------
// Local-array support for the i64 track. A variable is an "array var" when it
// is only ever assigned an integer array literal (or an alias of another array
// var — aliases share a handle exactly like the interpreter's shared Rc). Such
// arrays may be indexed, index-assigned, len()'d, push()'d and pop()'d, and
// must never escape (returned, passed to a call, or used as a scalar); then
// they can live in the JIT arena instead of the heap.
// ---------------------------------------------------------------------------

fn as_ident(e: &Expr) -> Option<&str> {
    match e {
        Expr::At { expr, .. } => as_ident(expr),
        Expr::Ident(n) => Some(n),
        _ => None,
    }
}

fn strip_at(e: &Expr) -> &Expr {
    match e { Expr::At { expr, .. } => strip_at(expr), other => other }
}

// Names assigned from an array literal or from another array var, to a fixpoint.
pub(crate) fn array_vars(f: &Func) -> HashSet<String> {
    let mut set: HashSet<String> = HashSet::new();
    loop {
        let before = set.len();
        collect_array_assigns(&f.body, &mut set);
        if set.len() == before { break; }
    }
    // a name also assigned any non-array value is not a stable array var
    let mut bad: HashSet<String> = HashSet::new();
    check_array_assign_conflicts(&f.body, &set, &mut bad);
    for b in bad { set.remove(&b); }
    set
}

fn collect_array_assigns(body: &[Stmt], set: &mut HashSet<String>) {
    for s in body {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                match strip_at(value) {
                    Expr::Array(_) => { set.insert(name.clone()); }
                    Expr::Ident(src) if set.contains(src) => { set.insert(name.clone()); }
                    _ => {}
                }
            }
            Stmt::If { then, els, .. } => {
                collect_array_assigns(then, set);
                if let Some(e) = els { collect_array_assigns(e, set); }
            }
            Stmt::While { body, .. } | Stmt::ForRange { body, .. } =>
                collect_array_assigns(body, set),
            _ => {}
        }
    }
}

fn check_array_assign_conflicts(body: &[Stmt], set: &HashSet<String>, bad: &mut HashSet<String>) {
    for s in body {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } if set.contains(name) => {
                match strip_at(value) {
                    Expr::Array(_) => {}
                    Expr::Ident(src) if set.contains(src) => {}
                    _ => { bad.insert(name.clone()); }
                }
            }
            Stmt::If { then, els, .. } => {
                check_array_assign_conflicts(then, set, bad);
                if let Some(e) = els { check_array_assign_conflicts(e, set, bad); }
            }
            Stmt::While { body, .. } | Stmt::ForRange { body, .. } =>
                check_array_assign_conflicts(body, set, bad),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Local-struct support for the i64 track. A variable is a "struct var" when it
// is only ever assigned a struct literal with one consistent field order (or an
// alias of another struct var). Fields hold scalars; reads/writes lower to the
// same arena as arrays (a struct is a fixed block of slots, field name -> slot
// index). Struct vars must never escape — same rule as array vars.
// ---------------------------------------------------------------------------

pub(crate) fn struct_vars(f: &Func, defs: &HashMap<String, Vec<String>>)
    -> HashMap<String, Vec<String>>
{
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    loop {
        let before = map.len();
        collect_struct_assigns(&f.body, defs, &mut map);
        if map.len() == before { break; }
    }
    // drop vars that are also assigned non-struct values or a different shape
    let mut bad: HashSet<String> = HashSet::new();
    check_struct_assign_conflicts(&f.body, &map, &mut bad);
    for b in bad { map.remove(&b); }
    map
}

fn lit_shape(fields: &[(String, Expr)]) -> Vec<String> {
    fields.iter().map(|(n, _)| n.clone()).collect()
}

// a literal is JIT-safe only when its struct is declared and its field-name
// set exactly matches the declaration (the interpreter errors on unknown
// structs, missing fields, and unknown fields — those must stay interp-run)
fn lit_matches_def(name: &str, fields: &[(String, Expr)],
                   defs: &HashMap<String, Vec<String>>) -> bool {
    match defs.get(name) {
        Some(decl) => decl.len() == fields.len()
            && decl.iter().all(|d| fields.iter().any(|(fname, _)| fname == d)),
        None => false,
    }
}

fn collect_struct_assigns(body: &[Stmt], defs: &HashMap<String, Vec<String>>,
                          map: &mut HashMap<String, Vec<String>>) {
    for s in body {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                match strip_at(value) {
                    Expr::StructLit { name: sname, fields }
                        if !map.contains_key(name) && lit_matches_def(sname, fields, defs) =>
                    {
                        map.insert(name.clone(), lit_shape(fields));
                    }
                    Expr::Ident(src) => {
                        if let Some(shape) = map.get(src).cloned() {
                            map.entry(name.clone()).or_insert(shape);
                        }
                    }
                    _ => {}
                }
            }
            Stmt::If { then, els, .. } => {
                collect_struct_assigns(then, defs, map);
                if let Some(e) = els { collect_struct_assigns(e, defs, map); }
            }
            Stmt::While { body, .. } | Stmt::ForRange { body, .. } =>
                collect_struct_assigns(body, defs, map),
            _ => {}
        }
    }
}

fn check_struct_assign_conflicts(body: &[Stmt], map: &HashMap<String, Vec<String>>,
                                 bad: &mut HashSet<String>) {
    for s in body {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value }
                if map.contains_key(name) =>
            {
                match strip_at(value) {
                    Expr::StructLit { fields, .. } if lit_shape(fields) == map[name] => {}
                    Expr::Ident(src) if map.get(src) == map.get(name) && map.contains_key(src) => {}
                    _ => { bad.insert(name.clone()); }
                }
            }
            Stmt::If { then, els, .. } => {
                check_struct_assign_conflicts(then, map, bad);
                if let Some(e) = els { check_struct_assign_conflicts(e, map, bad); }
            }
            Stmt::While { body, .. } | Stmt::ForRange { body, .. } =>
                check_struct_assign_conflicts(body, map, bad),
            _ => {}
        }
    }
}

// is this call one of the array builtins applied to a local array var?
fn array_builtin_call(callee: &str, args: &[Expr], arrays: &HashSet<String>) -> bool {
    match callee {
        "len" | "pop" => args.len() == 1
            && as_ident(&args[0]).map_or(false, |n| arrays.contains(n)),
        "push" => args.len() == 2
            && as_ident(&args[0]).map_or(false, |n| arrays.contains(n)),
        _ => false,
    }
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

fn stmt_pure(s: &Stmt, arrays: &HashSet<String>, structs: &HashMap<String, Vec<String>>) -> bool {
    match s {
        Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
            if arrays.contains(name) {
                // an array var may only be (re)assigned an array literal of
                // scalars or an alias of another array var
                match strip_at(value) {
                    Expr::Array(elems) => elems.iter().all(|e| expr_pure(e, arrays, structs)),
                    Expr::Ident(src) => arrays.contains(src),
                    _ => false,
                }
            } else if structs.contains_key(name) {
                // a struct var: a same-shape literal with pure field values, or
                // an alias of a same-shape struct var
                match strip_at(value) {
                    Expr::StructLit { fields, .. } =>
                        lit_shape(fields) == structs[name]
                        && fields.iter().all(|(_, e)| expr_pure(e, arrays, structs)),
                    Expr::Ident(src) => structs.get(src) == structs.get(name)
                        && structs.contains_key(src),
                    _ => false,
                }
            } else {
                expr_pure(value, arrays, structs)
            }
        }
        // p.field = value on a local struct var with a known field
        Stmt::FieldAssign { base, field, value } =>
            as_ident(base).and_then(|n| structs.get(n))
                .map_or(false, |shape| shape.iter().any(|f| f == field))
            && expr_pure(value, arrays, structs),
        Stmt::Expr(e) => {
            // allow `push(arr, v)` as a statement (it returns null in the interp)
            if let Expr::Call { callee, args } = strip_at(e) {
                if callee == "push" && array_builtin_call(callee, args, arrays) {
                    return expr_pure(&args[1], arrays, structs);
                }
            }
            expr_pure(e, arrays, structs)
        }
        Stmt::Return(Some(e)) => expr_pure(e, arrays, structs),
        Stmt::Return(None) => false,
        Stmt::IndexAssign { base, index, value } =>
            as_ident(base).map_or(false, |n| arrays.contains(n))
            && expr_pure(index, arrays, structs) && expr_pure(value, arrays, structs),
        Stmt::If { cond, then, els } =>
            expr_pure(cond, arrays, structs) && then.iter().all(|s| stmt_pure(s, arrays, structs))
            && els.as_ref().map_or(true, |e| e.iter().all(|s| stmt_pure(s, arrays, structs))),
        Stmt::While { cond, body } =>
            expr_pure(cond, arrays, structs) && body.iter().all(|s| stmt_pure(s, arrays, structs)),
        Stmt::ForRange { start, end, body, .. } =>
            expr_pure(start, arrays, structs) && expr_pure(end, arrays, structs)
            && body.iter().all(|s| stmt_pure(s, arrays, structs)),
        Stmt::Break(None) | Stmt::Continue => true,
        _ => false,
    }
}

fn expr_pure(e: &Expr, arrays: &HashSet<String>, structs: &HashMap<String, Vec<String>>) -> bool {
    match e {
        Expr::At { expr, .. } => expr_pure(expr, arrays, structs),
        Expr::Int(_) => true,
        // an array/struct var is not a scalar (escape = ineligible)
        Expr::Ident(n) => !arrays.contains(n) && !structs.contains_key(n),
        // p.field read on a local struct var with a known field is a scalar
        Expr::Field { base, field } =>
            as_ident(base).and_then(|n| structs.get(n))
                .map_or(false, |shape| shape.iter().any(|f| f == field)),
        Expr::Unary { op, expr } =>
            matches!(op, UnOp::Neg | UnOp::Not | UnOp::BitNot) && expr_pure(expr, arrays, structs),
        Expr::Binary { op, lhs, rhs } =>
            binop_pure(*op) && expr_pure(lhs, arrays, structs) && expr_pure(rhs, arrays, structs),
        Expr::If { cond, then, els } =>
            expr_pure(cond, arrays, structs) && expr_pure(then, arrays, structs) && expr_pure(els, arrays, structs),
        Expr::Block { stmts, tail } =>
            stmts.iter().all(|s| stmt_pure(s, arrays, structs))
            && tail.as_ref().map_or(false, |t| expr_pure(t, arrays, structs)),
        Expr::Index { base, index } =>
            as_ident(base).map_or(false, |n| arrays.contains(n))
            && expr_pure(index, arrays, structs),
        // len/pop on a local array var yield scalars; other calls must have
        // <= MAX_ARITY scalar args (callee eligibility enforced by the fixpoint)
        Expr::Call { callee, args } => {
            if array_builtin_call(callee, args, arrays) {
                callee != "push" // push yields null; only allowed as a statement
            } else {
                args.len() <= MAX_ARITY && args.iter().all(|a| expr_pure(a, arrays, structs))
            }
        }
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
            && FloatCheck::check_fn(f))
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

// Static kind of a numeric expression in the float track: definitely-Float or
// definitely-Int. Int values (literals, for-range counters) may mix freely
// with floats — the interpreter promotes via as_f, and we mirror that with an
// exact i64->f64 convert — but Int∘Int arithmetic must stay off this track
// (integer division/remainder truncate; overflow promotes to BigInt).
#[derive(Clone, Copy, PartialEq)]
pub enum FKind { F, I }

struct FloatCheck { kinds: HashMap<String, FKind> }

impl FloatCheck {
    fn check_fn(f: &Func) -> bool {
        let mut c = FloatCheck { kinds: HashMap::new() };
        for p in &f.params { c.kinds.insert(p.clone(), FKind::F); }
        !f.body.is_empty() && c.stmts(&f.body) && always_returns(&f.body)
    }

    fn stmts(&mut self, body: &[Stmt]) -> bool { body.iter().all(|s| self.stmt(s)) }

    // a variable's kind must stay stable across every assignment
    fn bind(&mut self, name: &str, k: FKind) -> bool {
        match self.kinds.get(name) {
            Some(prev) => *prev == k,
            None => { self.kinds.insert(name.to_string(), k); true }
        }
    }

    fn stmt(&mut self, s: &Stmt) -> bool {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                match self.expr(value) { Some(k) => self.bind(name, k), None => false }
            }
            Stmt::Expr(e) => self.expr(e).is_some(),
            Stmt::Return(Some(e)) => self.expr(e) == Some(FKind::F),
            Stmt::If { cond, then, els } =>
                self.cond(cond) && self.stmts(then)
                && els.as_ref().map_or(true, |e| self.stmts(e)),
            Stmt::While { cond, body } => self.cond(cond) && self.stmts(body),
            Stmt::ForRange { var, start, end, body, .. } =>
                self.expr(start) == Some(FKind::I)
                && self.expr(end) == Some(FKind::I)
                && self.bind(var, FKind::I)
                && self.stmts(body),
            Stmt::Break(None) | Stmt::Continue => true,
            _ => false,
        }
    }

    fn expr(&mut self, e: &Expr) -> Option<FKind> {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Float(_) => Some(FKind::F),
            Expr::Int(_) => Some(FKind::I),
            Expr::Ident(n) => self.kinds.get(n).copied(),
            Expr::Unary { op: UnOp::Neg, expr } => {
                match self.expr(expr) { Some(FKind::F) => Some(FKind::F), _ => None }
            }
            Expr::Binary { op, lhs, rhs }
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div
                                | BinOp::Rem | BinOp::Pow) =>
            {
                let l = self.expr(lhs)?;
                let r = self.expr(rhs)?;
                if l == FKind::I && r == FKind::I { None } else { Some(FKind::F) }
            }
            Expr::If { cond, then, els } => {
                if !self.cond(cond) { return None; }
                let t = self.expr(then)?;
                let e2 = self.expr(els)?;
                if t == e2 { Some(t) } else { None }
            }
            Expr::Block { stmts, tail } => {
                if !self.stmts(stmts) { return None; }
                self.expr(tail.as_ref()?)
            }
            Expr::Call { args, .. } => {
                if args.len() > MAX_ARITY { return None; }
                // callees receive f64 through the ABI, so every argument must
                // be definitely-Float (an Int arg would take the callee's
                // integer arms in the interpreter)
                for a in args { if self.expr(a) != Some(FKind::F) { return None; } }
                Some(FKind::F)
            }
            _ => None,
        }
    }

    // boolean condition: comparisons over any numeric mix (the interpreter
    // compares every numeric pair through as_f), combined with && || !.
    // Eq/Ne of two Ints stays off the track: values_eq compares those exactly.
    fn cond(&mut self, e: &Expr) -> bool {
        match e {
            Expr::At { expr, .. } => self.cond(expr),
            Expr::Unary { op: UnOp::Not, expr } => self.cond(expr),
            Expr::Binary { op: BinOp::And | BinOp::Or, lhs, rhs } =>
                self.cond(lhs) && self.cond(rhs),
            Expr::Binary { op, lhs, rhs }
                if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) =>
                self.expr(lhs).is_some() && self.expr(rhs).is_some(),
            Expr::Binary { op: BinOp::Eq | BinOp::Ne, lhs, rhs } => {
                match (self.expr(lhs), self.expr(rhs)) {
                    (Some(FKind::I), Some(FKind::I)) => false,
                    (Some(_), Some(_)) => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Unified numeric track (Phase 11). Handles functions that mix i64 and f64 —
// the shape the two scalar tracks each reject: integer loop counters and
// accumulators driving float math, with an int OR float result (e.g. a
// mandelbrot escape-counter). All parameters are integers (the defining trait:
// these are kernels driven by integer dimensions/counts), so the VM dispatches
// them exactly like the i64 track — all-Int args, deopt-guarded — and reads the
// result back as Int or Float per the function's return kind. Everything the
// pure i64/f64 tracks already claim is left to them; this only adds coverage.
// ---------------------------------------------------------------------------

// A function is numeric-eligible iff it is neither i64- nor f64-eligible, all
// its params behave as integers, every statement/expression type-checks under
// NumCheck (int∘int allowed — it deopts on overflow like the i64 track — plus
// float and mixed math, to_float/to_int, and calls to other numeric functions),
// and it always returns a definite kind. The returned map also records each
// function's result kind (I or F).
#[cfg_attr(not(feature = "jit"), allow(dead_code))] // consumed only by the JIT
pub fn numeric_eligible_set(prog: &Program, int_set: &HashSet<String>, float_set: &HashSet<String>)
    -> (HashSet<String>, HashMap<String, FKind>)
{
    let mut funcs: HashMap<&str, &Func> = HashMap::new();
    for item in &prog.items {
        if let Item::Func(f) = item { funcs.insert(&f.name, f); }
    }
    let mut rets: HashMap<String, FKind> = HashMap::new();
    let mut set: HashSet<String> = funcs.values()
        .filter(|f| f.params.len() <= MAX_ARITY
            && !int_set.contains(&f.name) && !float_set.contains(&f.name))
        .filter_map(|f| NumCheck::check_fn(f).map(|rk| { rets.insert(f.name.clone(), rk); f.name.clone() }))
        .collect();
    // fixpoint: drop any function calling a name outside the numeric set (calls
    // to i64/f64 functions are not allowed here — the ABI/kinds differ)
    loop {
        let mut remove = None;
        for name in &set {
            let f = funcs[name.as_str()];
            if collect_calls(&f.body).iter().any(|c| !set.contains(c) && !is_num_intrinsic(c)) {
                remove = Some(name.clone());
                break;
            }
        }
        match remove { Some(n) => { set.remove(&n); rets.remove(&n); } None => break }
    }
    (set, rets)
}

#[cfg_attr(not(feature = "jit"), allow(dead_code))] // consumed only by the JIT
fn is_num_intrinsic(name: &str) -> bool { matches!(name, "to_float" | "to_int") }

// Type-checker for the numeric track: like FloatCheck but int∘int is allowed
// (kind I, deopts on overflow) and the return may be I or F.
struct NumCheck { kinds: HashMap<String, FKind> }

// Public for the AOT backend: if `f` is numeric-eligible (mixed int/float, all
// params int), return (result kind, per-local FKind map). None if ineligible.
// Rejects functions containing `**` (the AOT mixed path can't emit its
// deopt-on-overflow semantics) so those fall back to embed honestly.
pub fn numeric_kinds(f: &Func) -> Option<(FKind, HashMap<String, FKind>)> {
    if body_has_pow(&f.body) { return None; }
    let mut c = NumCheck { kinds: HashMap::new() };
    for p in &f.params { c.kinds.insert(p.clone(), FKind::I); }
    if f.body.is_empty() || !always_returns(&f.body) { return None; }
    let mut ret: Option<FKind> = None;
    if !c.stmts(&f.body, &mut ret) { return None; }
    ret.map(|rk| (rk, c.kinds))
}

fn body_has_pow(body: &[Stmt]) -> bool {
    fn se(e: &Expr) -> bool {
        match e {
            Expr::At { expr, .. } | Expr::Unary { expr, .. } => se(expr),
            Expr::Binary { op, lhs, rhs } => *op == BinOp::Pow || se(lhs) || se(rhs),
            Expr::If { cond, then, els } => se(cond) || se(then) || se(els),
            Expr::Call { args, .. } => args.iter().any(se),
            Expr::Block { stmts, tail } => stmts.iter().any(ss) || tail.as_ref().map_or(false, |t| se(t)),
            _ => false,
        }
    }
    fn ss(s: &Stmt) -> bool {
        match s {
            Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Expr(value)
            | Stmt::Return(Some(value)) => se(value),
            Stmt::If { cond, then, els } => se(cond) || then.iter().any(ss)
                || els.as_ref().map_or(false, |e| e.iter().any(ss)),
            Stmt::While { cond, body } => se(cond) || body.iter().any(ss),
            Stmt::ForRange { start, end, body, .. } => se(start) || se(end) || body.iter().any(ss),
            _ => false,
        }
    }
    body.iter().any(ss)
}

impl NumCheck {
    // returns the function's result kind if it is numeric-eligible
    #[cfg_attr(not(feature = "jit"), allow(dead_code))] // consumed only by the JIT
    fn check_fn(f: &Func) -> Option<FKind> {
        let mut c = NumCheck { kinds: HashMap::new() };
        for p in &f.params { c.kinds.insert(p.clone(), FKind::I); } // all params are ints
        if f.body.is_empty() || !always_returns(&f.body) { return None; }
        let mut ret: Option<FKind> = None;
        if !c.stmts(&f.body, &mut ret) { return None; }
        ret
    }

    fn stmts(&mut self, body: &[Stmt], ret: &mut Option<FKind>) -> bool {
        body.iter().all(|s| self.stmt(s, ret))
    }

    fn bind(&mut self, name: &str, k: FKind) -> bool {
        match self.kinds.get(name) {
            Some(prev) => *prev == k,
            None => { self.kinds.insert(name.to_string(), k); true }
        }
    }

    fn set_ret(ret: &mut Option<FKind>, k: FKind) -> bool {
        match ret { Some(prev) => *prev == k, None => { *ret = Some(k); true } }
    }

    fn stmt(&mut self, s: &Stmt, ret: &mut Option<FKind>) -> bool {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } =>
                match self.expr(value) { Some(k) => self.bind(name, k), None => false },
            Stmt::Expr(e) => self.expr(e).is_some(),
            Stmt::Return(Some(e)) => match self.expr(e) { Some(k) => Self::set_ret(ret, k), None => false },
            Stmt::If { cond, then, els } =>
                self.cond(cond) && self.stmts(then, ret)
                && els.as_ref().map_or(true, |e| self.stmts(e, ret)),
            Stmt::While { cond, body } => self.cond(cond) && self.stmts(body, ret),
            Stmt::ForRange { var, start, end, body, .. } =>
                self.expr(start) == Some(FKind::I) && self.expr(end) == Some(FKind::I)
                && self.bind(var, FKind::I) && self.stmts(body, ret),
            Stmt::Break(None) | Stmt::Continue => true,
            _ => false,
        }
    }

    fn expr(&mut self, e: &Expr) -> Option<FKind> {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Float(_) => Some(FKind::F),
            Expr::Int(_) => Some(FKind::I),
            Expr::Ident(n) => self.kinds.get(n).copied(),
            Expr::Unary { op: UnOp::Neg, expr } => self.expr(expr), // I or F
            Expr::Binary { op, lhs, rhs }
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div
                                | BinOp::Rem | BinOp::Pow) =>
            {
                let l = self.expr(lhs)?;
                let r = self.expr(rhs)?;
                // int∘int stays I (deopts on overflow); any float ⇒ F
                if l == FKind::I && r == FKind::I { Some(FKind::I) } else { Some(FKind::F) }
            }
            Expr::If { cond, then, els } => {
                if !self.cond(cond) { return None; }
                let t = self.expr(then)?;
                let e2 = self.expr(els)?;
                if t == e2 { Some(t) } else { None }
            }
            Expr::Block { stmts, tail } => {
                let mut dummy = None;
                if !self.stmts(stmts, &mut dummy) { return None; }
                self.expr(tail.as_ref()?)
            }
            Expr::Call { callee, args } => {
                if callee == "to_float" && args.len() == 1 {
                    return if self.expr(&args[0]).is_some() { Some(FKind::F) } else { None };
                }
                if callee == "to_int" && args.len() == 1 {
                    // to_int only accepts a Float here (int→int is a no-op we skip)
                    return if self.expr(&args[0]) == Some(FKind::F) { Some(FKind::I) } else { None };
                }
                None // user numeric-fn calls handled by the fixpoint; kinds unknown here → reject for now
            }
            _ => None,
        }
    }

    fn cond(&mut self, e: &Expr) -> bool {
        match e {
            Expr::At { expr, .. } => self.cond(expr),
            Expr::Unary { op: UnOp::Not, expr } => self.cond(expr),
            Expr::Binary { op: BinOp::And | BinOp::Or, lhs, rhs } => self.cond(lhs) && self.cond(rhs),
            Expr::Binary { op, lhs, rhs }
                if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                                | BinOp::Eq | BinOp::Ne) =>
                self.expr(lhs).is_some() && self.expr(rhs).is_some(),
            _ => false,
        }
    }
}

fn binop_pure(op: BinOp) -> bool {
    use BinOp::*;
    matches!(op, Add | Sub | Mul | Div | Rem | Pow
        | Eq | Ne | Lt | Le | Gt | Ge | And | Or
        | BitOr | BitXor | BitAnd | Shl | Shr)
}

// does this body contain a loop at any nesting depth?
#[cfg_attr(not(feature = "jit"), allow(dead_code))] // consumed only by the JIT
fn body_has_loop(body: &[Stmt]) -> bool {
    body.iter().any(|s| match s {
        Stmt::While { .. } | Stmt::ForRange { .. } | Stmt::ForEach { .. } => true,
        Stmt::If { then, els, .. } =>
            body_has_loop(then) || els.as_ref().map_or(false, |e| body_has_loop(e)),
        _ => false,
    })
}

// every function name called anywhere in a body
fn collect_calls(body: &[Stmt]) -> Vec<String> {
    let mut out = Vec::new();
    for s in body { calls_stmt(s, &mut out); }
    out
}

// like collect_calls, but array builtins on local array vars (len/push/pop)
// are not real calls — they lower to arena helpers, not Nova functions
fn collect_real_calls(body: &[Stmt], arrays: &HashSet<String>, structs: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut out = Vec::new();
    for s in body { real_calls_stmt(s, arrays, structs, &mut out); }
    out
}
fn real_calls_stmt(s: &Stmt, arrays: &HashSet<String>, structs: &HashMap<String, Vec<String>>, out: &mut Vec<String>) {
    match s {
        Stmt::Let { value, .. } | Stmt::Assign { value, .. } | Stmt::Expr(value)
        | Stmt::Return(Some(value)) => real_calls_expr(value, arrays, structs, out),
        Stmt::IndexAssign { base, index, value } => {
            real_calls_expr(base, arrays, structs, out);
            real_calls_expr(index, arrays, structs, out);
            real_calls_expr(value, arrays, structs, out);
        }
        Stmt::FieldAssign { base, value, .. } => {
            real_calls_expr(base, arrays, structs, out);
            real_calls_expr(value, arrays, structs, out);
        }
        Stmt::If { cond, then, els } => {
            real_calls_expr(cond, arrays, structs, out);
            for s in then { real_calls_stmt(s, arrays, structs, out); }
            if let Some(e) = els { for s in e { real_calls_stmt(s, arrays, structs, out); } }
        }
        Stmt::While { cond, body } => {
            real_calls_expr(cond, arrays, structs, out);
            for s in body { real_calls_stmt(s, arrays, structs, out); }
        }
        Stmt::ForRange { start, end, body, .. } => {
            real_calls_expr(start, arrays, structs, out);
            real_calls_expr(end, arrays, structs, out);
            for s in body { real_calls_stmt(s, arrays, structs, out); }
        }
        _ => {}
    }
}
fn real_calls_expr(e: &Expr, arrays: &HashSet<String>, structs: &HashMap<String, Vec<String>>, out: &mut Vec<String>) {
    match e {
        Expr::At { expr, .. } | Expr::Unary { expr, .. } => real_calls_expr(expr, arrays, structs, out),
        Expr::Binary { lhs, rhs, .. } => {
            real_calls_expr(lhs, arrays, structs, out);
            real_calls_expr(rhs, arrays, structs, out);
        }
        Expr::If { cond, then, els } => {
            real_calls_expr(cond, arrays, structs, out);
            real_calls_expr(then, arrays, structs, out);
            real_calls_expr(els, arrays, structs, out);
        }
        Expr::Block { stmts, tail } => {
            for s in stmts { real_calls_stmt(s, arrays, structs, out); }
            if let Some(t) = tail { real_calls_expr(t, arrays, structs, out); }
        }
        Expr::Index { base, index } => {
            real_calls_expr(base, arrays, structs, out);
            real_calls_expr(index, arrays, structs, out);
        }
        Expr::StructLit { fields, .. } => {
            for (_, v) in fields { real_calls_expr(v, arrays, structs, out); }
        }
        Expr::Field { base, .. } => real_calls_expr(base, arrays, structs, out),
        Expr::Call { callee, args } => {
            if !array_builtin_call(callee, args, arrays) {
                out.push(callee.clone());
            }
            // still scan args (a push value may itself contain a real call)
            for a in args { real_calls_expr(a, arrays, structs, out); }
        }
        _ => {}
    }
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
// JIT module (cranelift-backed; compiled only when the `jit` feature is on)
// ---------------------------------------------------------------------------

#[cfg(feature = "jit")]
mod cl {
use super::*;
use cranelift::prelude::*;
use cranelift::prelude::types::{I64, I128, I32, I8};
use cranelift::codegen::Context;
use cranelift::codegen::control::ControlPlane;
use cranelift::codegen::isa;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, Linkage, FuncId, DataId, DataDescription};
use cranelift_object::{ObjectBuilder, ObjectModule};

// ---------------------------------------------------------------------------
// Runtime helpers callable from JIT code.
//
// Local integer arrays: a JIT-eligible function may build arrays of i64 that
// never escape it (see `array_vars`). They live in a thread-local arena of
// Vec<i64> pools addressed by handle; `raw_call` resets the arena after each
// top-level native call, so re-running on the VM after a deopt starts clean —
// array effects are invisible outside the JIT, preserving purity. Bounds
// violations and pops of an empty array set the deopt flag: the VM re-run then
// raises the exact interpreter error (or returns its null for empty pop).
// ---------------------------------------------------------------------------

thread_local! {
    static JIT_ARENA: std::cell::RefCell<(Vec<Vec<i64>>, usize)> =
        std::cell::RefCell::new((Vec::new(), 0));
}

extern "C" fn nova_arr_new() -> i64 {
    JIT_ARENA.with(|a| {
        let (pool, live) = &mut *a.borrow_mut();
        if *live < pool.len() { pool[*live].clear(); } else { pool.push(Vec::new()); }
        *live += 1;
        (*live - 1) as i64
    })
}

extern "C" fn nova_arr_push(h: i64, v: i64) {
    JIT_ARENA.with(|a| a.borrow_mut().0[h as usize].push(v));
}

// Allocate a fresh arena array of `n` copies of `v` in one shot — the fused form
// of `s = []; for _ in 0..n { push(s, v) }`. `n <= 0` yields an empty array,
// matching a for-range that never iterates. Byte-identical to the push loop.
extern "C" fn nova_arr_fill(n: i64, v: i64) -> i64 {
    JIT_ARENA.with(|a| {
        let (pool, live) = &mut *a.borrow_mut();
        let cnt = if n <= 0 { 0 } else { n as usize };
        if *live < pool.len() {
            let slot = &mut pool[*live];
            slot.clear();
            slot.resize(cnt, v);
        } else {
            pool.push(vec![v; cnt]);
        }
        *live += 1;
        (*live - 1) as i64
    })
}

extern "C" fn nova_arr_len(h: i64) -> i64 {
    JIT_ARENA.with(|a| a.borrow().0[h as usize].len() as i64)
}

extern "C" fn nova_arr_get(dp: *mut i64, h: i64, i: i64) -> i64 {
    JIT_ARENA.with(|a| {
        let arr = &a.borrow().0[h as usize];
        if i < 0 || i as usize >= arr.len() {
            unsafe { *dp = 1; }
            0
        } else {
            arr[i as usize]
        }
    })
}

extern "C" fn nova_arr_set(dp: *mut i64, h: i64, i: i64, v: i64) {
    JIT_ARENA.with(|a| {
        let arr = &mut a.borrow_mut().0[h as usize];
        if i < 0 || i as usize >= arr.len() {
            unsafe { *dp = 1; }
        } else {
            arr[i as usize] = v;
        }
    })
}

extern "C" fn nova_arr_pop(dp: *mut i64, h: i64) -> i64 {
    JIT_ARENA.with(|a| {
        match a.borrow_mut().0[h as usize].pop() {
            Some(v) => v,
            None => { unsafe { *dp = 1; } 0 } // interp returns null: deopt re-runs
        }
    })
}

fn jit_arena_reset() {
    JIT_ARENA.with(|a| a.borrow_mut().1 = 0);
}

// f64 `%` and `**` have no Cranelift instruction; call back into Rust so the
// results are bit-identical to the interpreter's `as_f(l) % as_f(r)` / powf.
extern "C" fn nova_fmod(a: f64, b: f64) -> f64 { a % b }
extern "C" fn nova_fpow(a: f64, b: f64) -> f64 { a.powf(b) }

// Minimum function count before parallel compilation pays for its thread setup.
const PAR_MIN: usize = 6;

// Multicore compilation: lower a batch of already-IR-generated functions to
// machine code across the host's cores, then define them into `module` serially.
//
// This is the honest, buildable form of "compile n× faster with n units": each
// function's back end (legalization, register allocation, encoding — the
// expensive step) is independent, so it fans out across threads. Cranelift
// codegen is deterministic per function (thread-independent), and this mirrors
// exactly what `Module::define_function` does serially — `ctx.compile(isa,
// ControlPlane::default())` then `define_function_bytes(...)` — so the emitted
// machine code, and hence program output, is byte-identical to the serial path.
// The corpus (×4) and AOT oracle gates still verify every tier.
//
// Falls back to serial for a single core or a small batch (thread setup would
// cost more than it saves). Returns None on any compile/define failure — the
// caller then disables that tier and runs the correct VM/interpreter, never
// shipping wrong code.
fn define_parallel<M: Module>(module: &mut M, mut jobs: Vec<(FuncId, Context)>) -> Option<()> {
    // Thread count: the host's core count, overridable via NOVA_COMPILE_THREADS
    // (1 forces the serial path — handy for benchmarking the n× speedup and as an
    // escape hatch). Clamped to at least 1.
    let ncpu = std::env::var("NOVA_COMPILE_THREADS").ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let timing = std::env::var("NOVA_COMPILE_TIMING").is_ok();
    let t0 = std::time::Instant::now();
    if ncpu <= 1 || jobs.len() < PAR_MIN {
        for (id, ctx) in &mut jobs { module.define_function(*id, ctx).ok()?; }
        if timing { eprintln!("nova: compiled {} fns serially in {:?}", jobs.len(), t0.elapsed()); }
        return Some(());
    }
    // 1) parallel: compile each function to machine code across ALL of the host's
    //    logical CPUs (every core of every socket that `available_parallelism`
    //    reports — so a dual-socket 2×96-core workstation runs ~192-wide, a
    //    many-socket rack wider still). A shared work queue makes it dynamic
    //    work-stealing: every worker pulls the next function the instant it
    //    finishes, so no core idles even when per-function compile cost varies
    //    wildly — that's what keeps the speedup near-linear at high core counts.
    //    Only the brief pop/push touch the lock; `ctx.compile` runs unlocked. The
    //    ISA is Send+Sync; each worker uses its own ControlPlane. Deterministic
    //    per function, so output stays byte-identical regardless of thread count.
    let njobs = jobs.len();
    let nthreads = ncpu.min(njobs);
    let compiled: Vec<(FuncId, Context)> = {
        let isa = module.isa();
        let queue = std::sync::Mutex::new(jobs);            // functions still to compile
        let done = std::sync::Mutex::new(Vec::with_capacity(njobs)); // compiled results
        let ok = std::thread::scope(|s| {
            let handles: Vec<_> = (0..nthreads).map(|_| s.spawn(|| {
                let mut cp = ControlPlane::default();
                loop {
                    let job = queue.lock().unwrap().pop();
                    let Some((id, mut ctx)) = job else { return true };
                    if ctx.compile(isa, &mut cp).is_err() { return false; }
                    done.lock().unwrap().push((id, ctx));
                }
            })).collect();
            handles.into_iter().all(|h| h.join().unwrap_or(false))
        });
        if !ok { return None; }
        done.into_inner().unwrap()
    };
    // 2) serial: ingest each function's pre-compiled bytes + relocations. Order is
    //    irrelevant — each function is defined independently by its FuncId.
    for (id, ctx) in &compiled {
        let cc = ctx.compiled_code()?;
        module.define_function_bytes(*id, &ctx.func, cc.buffer.alignment as u64,
            cc.code_buffer(), cc.buffer.relocs()).ok()?;
    }
    if timing {
        eprintln!("nova: compiled {} fns across {} threads in {:?}",
            njobs, nthreads, t0.elapsed());
    }
    Some(())
}

impl Drop for Jit {
    fn drop(&mut self) {
        // Release the executable arena. Safe: `Drop` runs only once execution is
        // finished, so no JIT'd function pointer (in `code`/`fcode`/`ncode`) is
        // ever called after this frees their memory.
        if let Some(m) = self._module.take() {
            unsafe { m.free_memory(); }
        }
    }
}

pub struct Jit {
    // Kept alive so the JIT'd code pages stay mapped. `Option` so `Drop` can take
    // it and call `free_memory` (which consumes the module) — cranelift's own
    // `Drop` does NOT release the executable arena, so without this every process
    // that warms the JIT leaks its code memory until exit (valgrind-dirty).
    _module: Option<JITModule>,
    // name -> (entry pointer, arity); i64 track
    code: HashMap<String, (*const u8, usize)>,
    // name -> (entry pointer, arity); f64 track (disjoint names)
    fcode: HashMap<String, (*const u8, usize)>,
    // name -> (entry pointer, arity); numeric mixed-int/float track (i64 ABI)
    ncode: HashMap<String, (*const u8, usize)>,
    // numeric functions' result kind (F ⇒ the i64 result is f64 bits)
    nret: HashMap<String, FKind>,
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
        let mut feligible = float_eligible_set(prog, &eligible);
        let (mut numeric, mut nret) = numeric_eligible_set(prog, &eligible, &feligible);
        if let Some(filter) = only {
            eligible.retain(|n| filter.contains(n));
            feligible.retain(|n| filter.contains(n));
            numeric.retain(|n| filter.contains(n));
            nret.retain(|n, _| filter.contains(n));
        }
        if eligible.is_empty() && feligible.is_empty() && numeric.is_empty() { return None; }
        let mut funcs: HashMap<&str, &Func> = HashMap::new();
        let mut sdefs: HashMap<String, Vec<String>> = HashMap::new();
        for item in &prog.items {
            match item {
                Item::Func(f) => { funcs.insert(&f.name, f); }
                Item::Struct(sd) => { sdefs.insert(sd.name.clone(), sd.fields.clone()); }
                _ => {}
            }
        }

        let mut flags = settings::builder();
        flags.set("opt_level", "speed").ok()?;
        let isa = cranelift_native::builder().ok()?
            .finish(settings::Flags::new(flags)).ok()?;
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        builder.symbol("nova_arr_new", nova_arr_new as *const u8);
        builder.symbol("nova_arr_fill", nova_arr_fill as *const u8);
        builder.symbol("nova_arr_push", nova_arr_push as *const u8);
        builder.symbol("nova_arr_len", nova_arr_len as *const u8);
        builder.symbol("nova_arr_get", nova_arr_get as *const u8);
        builder.symbol("nova_arr_set", nova_arr_set as *const u8);
        builder.symbol("nova_arr_pop", nova_arr_pop as *const u8);
        builder.symbol("nova_fmod", nova_fmod as *const u8);
        builder.symbol("nova_fpow", nova_fpow as *const u8);
        let mut module = JITModule::new(builder);

        // imported runtime helpers (arena arrays + f64 %/**)
        let mut helpers: HashMap<&'static str, FuncId> = HashMap::new();
        {
            let f64t = types::F64;
            let sigs: [(&'static str, &[Type], &[Type]); 9] = [
                ("nova_arr_new", &[], &[I64]),
                ("nova_arr_fill", &[I64, I64], &[I64]),
                ("nova_arr_push", &[I64, I64], &[]),
                ("nova_arr_len", &[I64], &[I64]),
                ("nova_arr_get", &[I64, I64, I64], &[I64]),
                ("nova_arr_set", &[I64, I64, I64, I64], &[]),
                ("nova_arr_pop", &[I64, I64], &[I64]),
                ("nova_fmod", &[f64t, f64t], &[f64t]),
                ("nova_fpow", &[f64t, f64t], &[f64t]),
            ];
            for (name, params, rets) in sigs {
                let mut sig = module.make_signature();
                for p in params { sig.params.push(AbiParam::new(*p)); }
                for r in rets { sig.returns.push(AbiParam::new(*r)); }
                let id = module.declare_function(name, Linkage::Import, &sig).ok()?;
                helpers.insert(name, id);
            }
        }
        let libm: HashMap<&'static str, FuncId> = [
            ("fmod", helpers["nova_fmod"]), ("fpow", helpers["nova_fpow"]),
        ].into_iter().collect();

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

        // 2) generate each function body's IR (serial), collecting Contexts to
        //    compile across cores below.
        let mut fctx = FunctionBuilderContext::new();
        let mut ir = HashMap::new();
        let mut jobs: Vec<(FuncId, Context)> = Vec::new();
        for name in &names {
            let f = funcs[*name];
            let (id, _) = ids[*name];
            let mut ctx = module.make_context();
            ctx.func.signature.params.push(AbiParam::new(I64));
            for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(I64)); }
            ctx.func.signature.returns.push(AbiParam::new(I64));
            {
                let arrays = array_vars(f);
                let structs = struct_vars(f, &sdefs);
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
                let mut g = FnGen::new(&mut b, &mut module, &ids, &helpers, arrays, structs, &f.params, false);
                g.lower(&f.body)?;
                b.finalize();
            }
            ir.insert(name.to_string(), format!("{}", ctx.func));
            jobs.push((id, ctx));
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
            let mut ctx = module.make_context();
            for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(types::F64)); }
            ctx.func.signature.returns.push(AbiParam::new(types::F64));
            {
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
                let mut g = FloatGen::new(&mut b, &mut module, &fids, &libm, &f.params);
                g.lower(&f.body)?;
                b.finalize();
            }
            ir.insert(name.to_string(), format!("{}", ctx.func));
            jobs.push((id, ctx));
        }

        // 4) the numeric (mixed int/float) track: same all-i64 ABI as track 1
        //    (deopt_ptr + i64 args → i64 bits), defined via NumGen.
        let mut nnames: Vec<&str> = numeric.iter().map(|s| s.as_str()).collect();
        nnames.sort();
        let mut nids: HashMap<String, (FuncId, usize)> = HashMap::new();
        for name in &nnames {
            let f = funcs[*name];
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(I64)); // deopt flag pointer
            for _ in 0..f.params.len() { sig.params.push(AbiParam::new(I64)); }
            sig.returns.push(AbiParam::new(I64));
            let id = module.declare_function(name, Linkage::Export, &sig).ok()?;
            nids.insert(name.to_string(), (id, f.params.len()));
        }
        for name in &nnames {
            let f = funcs[*name];
            let (id, _) = nids[*name];
            let rk = nret[*name];
            let mut ctx = module.make_context();
            ctx.func.signature.params.push(AbiParam::new(I64));
            for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(I64)); }
            ctx.func.signature.returns.push(AbiParam::new(I64));
            {
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
                let mut g = NumGen::new(&mut b, &mut module, &libm, &f.params, rk);
                g.lower(&f.body)?;
                b.finalize();
            }
            ir.insert(name.to_string(), format!("{}", ctx.func));
            jobs.push((id, ctx));
        }

        // compile every collected function across the host's cores, then finalize.
        define_parallel(&mut module, jobs)?;
        module.finalize_definitions().ok()?;
        let mut code = HashMap::new();
        for (name, (id, arity)) in &ids {
            code.insert(name.clone(), (module.get_finalized_function(*id), *arity));
        }
        let mut fcode = HashMap::new();
        for (name, (id, arity)) in &fids {
            fcode.insert(name.clone(), (module.get_finalized_function(*id), *arity));
        }
        let mut ncode = HashMap::new();
        for (name, (id, arity)) in &nids {
            ncode.insert(name.clone(), (module.get_finalized_function(*id), *arity));
        }
        Some(Jit { _module: Some(module), code, fcode, ncode, nret, ir })
    }

    pub fn is_compiled(&self, name: &str) -> bool { self.code.contains_key(name) }
    pub fn is_compiled_f64(&self, name: &str) -> bool { self.fcode.contains_key(name) }
    pub fn is_compiled_num(&self, name: &str) -> bool { self.ncode.contains_key(name) }
    // is the numeric function's result a float (returned as raw bits)?
    pub fn num_ret_is_float(&self, name: &str) -> bool { self.nret.get(name) == Some(&FKind::F) }
    // any track — used by tiering to record what a batch produced
    pub fn has(&self, name: &str) -> bool {
        self.code.contains_key(name) || self.fcode.contains_key(name) || self.ncode.contains_key(name)
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
        Self::call_i64_abi(ptr, arity, args)
    }

    // The numeric track shares the i64 ABI; the raw i64 result is either an
    // integer or f64 bits (the caller reinterprets via `num_ret_is_float`).
    pub fn raw_call_num(&self, name: &str, args: &[i64]) -> (i64, bool) {
        let (ptr, arity) = self.ncode[name];
        Self::call_i64_abi(ptr, arity, args)
    }

    fn call_i64_abi(ptr: *const u8, arity: usize, args: &[i64]) -> (i64, bool) {
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
        // local arrays live only for the duration of one top-level native call;
        // resetting here also makes the deopt re-run on the VM start clean
        jit_arena_reset();
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
    module: &'a mut dyn Module,
    ids: &'a HashMap<String, (FuncId, usize)>,
    libm: &'a HashMap<&'static str, FuncId>,
    // variables carry their static kind: F64 values, or I64 for-range
    // counters / int-literal bindings that promote via as_f at each use
    vars: HashMap<String, (Variable, FKind)>,
    n_vars: usize,
    loops: Vec<LoopCtx>,
    returned: bool,
}

impl<'a, 'b> FloatGen<'a, 'b> {
    fn new(b: &'a mut FunctionBuilder<'b>, module: &'a mut dyn Module,
           ids: &'a HashMap<String, (FuncId, usize)>,
           libm: &'a HashMap<&'static str, FuncId>, params: &[String]) -> Self {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let param_vals: Vec<Value> = b.block_params(entry).to_vec();
        let mut g = FloatGen {
            b, module, ids, libm, vars: HashMap::new(), n_vars: 0,
            loops: Vec::new(), returned: false,
        };
        for (i, p) in params.iter().enumerate() {
            let v = g.declare(p, FKind::F);
            g.b.def_var(v, param_vals[i]);
        }
        g
    }

    fn declare(&mut self, name: &str, kind: FKind) -> Variable {
        if let Some((v, k)) = self.vars.get(name) {
            debug_assert!(*k == kind, "kind changed for {}", name);
            return *v;
        }
        let v = Variable::new(self.n_vars);
        self.n_vars += 1;
        self.b.declare_var(v, if kind == FKind::F { types::F64 } else { I64 });
        self.vars.insert(name.to_string(), (v, kind));
        v
    }

    // promote an (value, kind) pair to f64 — the interpreter's as_f
    fn to_f(&mut self, v: Value, k: FKind) -> Value {
        match k {
            FKind::F => v,
            FKind::I => self.b.ins().fcvt_from_sint(types::F64, v),
        }
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
                let (v, k) = self.expr(value)?;
                let var = self.declare(name, k);
                self.b.def_var(var, v);
            }
            Stmt::Expr(e) => { self.expr(e)?; }
            Stmt::Return(Some(e)) => {
                let (v, k) = self.expr(e)?;
                let v = self.to_f(v, k);
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
            Stmt::ForRange { var, start, end, inclusive, body } => {
                // i64 counter exactly like the interpreter's ForRange; the loop
                // var is I-kind and promotes at float uses inside the body
                let (s0, sk) = self.expr(start)?;
                if sk != FKind::I { return None; }
                let (e0, ek) = self.expr(end)?;
                if ek != FKind::I { return None; }
                let iv = self.declare(var, FKind::I);
                self.b.def_var(iv, s0);
                let limit = {
                    let v = Variable::new(self.n_vars);
                    self.n_vars += 1;
                    self.b.declare_var(v, I64);
                    v
                };
                self.b.def_var(limit, e0);

                let header = self.b.create_block();
                let body_b = self.b.create_block();
                let exit = self.b.create_block();
                self.b.ins().jump(header, &[]);
                self.b.switch_to_block(header);
                let i = self.b.use_var(iv);
                let lim = self.b.use_var(limit);
                let cc = if *inclusive { IntCC::SignedLessThanOrEqual } else { IntCC::SignedLessThan };
                let cont = self.b.ins().icmp(cc, i, lim);
                self.b.ins().brif(cont, body_b, &[], exit, &[]);
                self.b.switch_to_block(body_b);
                self.b.seal_block(body_b);
                self.loops.push(LoopCtx { header, exit });
                self.returned = false;
                for st in body { self.stmt(st)?; }
                if !self.returned {
                    let i = self.b.use_var(iv);
                    let next = self.b.ins().iadd_imm(i, 1);
                    self.b.def_var(iv, next);
                    self.b.ins().jump(header, &[]);
                }
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

    // boolean condition (i64 0/1): numeric comparisons through as_f (exactly
    // the interpreter's is_num arms) + short-circuit && || !
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
                let (a, ak) = self.expr(lhs)?;
                let (bv, bk) = self.expr(rhs)?;
                let a = self.to_f(a, ak);
                let bv = self.to_f(bv, bk);
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

    fn expr(&mut self, e: &Expr) -> Option<(Value, FKind)> {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Float(x) => Some((self.b.ins().f64const(*x), FKind::F)),
            Expr::Int(n) => Some((self.b.ins().iconst(I64, *n), FKind::I)),
            Expr::Ident(name) => {
                let (v, k) = *self.vars.get(name)?;
                Some((self.b.use_var(v), k))
            }
            Expr::Unary { op: UnOp::Neg, expr } => {
                let (v, k) = self.expr(expr)?;
                if k != FKind::F { return None; }
                Some((self.b.ins().fneg(v), FKind::F))
            }
            Expr::Binary { op, lhs, rhs } => {
                let (a, ak) = self.expr(lhs)?;
                let (bv, bk) = self.expr(rhs)?;
                if ak == FKind::I && bk == FKind::I { return None; }
                let a = self.to_f(a, ak);
                let bv = self.to_f(bv, bk);
                let v = match op {
                    BinOp::Add => self.b.ins().fadd(a, bv),
                    BinOp::Sub => self.b.ins().fsub(a, bv),
                    BinOp::Mul => self.b.ins().fmul(a, bv),
                    BinOp::Div => self.b.ins().fdiv(a, bv), // /0.0 -> inf, as the interp
                    BinOp::Rem => self.libcall("fmod", a, bv)?,
                    BinOp::Pow => self.libcall("fpow", a, bv)?,
                    _ => return None,
                };
                Some((v, FKind::F))
            }
            Expr::Call { callee, args } => {
                let (id, arity) = *self.ids.get(callee.as_str())?;
                if args.len() != arity { return None; }
                let fref = self.module.declare_func_in_func(id, self.b.func);
                let mut argv = Vec::with_capacity(arity);
                for a in args {
                    let (v, k) = self.expr(a)?;
                    if k != FKind::F { return None; }
                    argv.push(v);
                }
                let inst = self.b.ins().call(fref, &argv);
                Some((self.b.inst_results(inst)[0], FKind::F))
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
                let (tv, tk) = self.expr(then)?;
                if tk != FKind::F { return None; }
                self.b.ins().jump(merge, &[tv]);
                self.b.switch_to_block(else_b);
                self.b.seal_block(else_b);
                let (ev, ek) = self.expr(els)?;
                if ek != FKind::F { return None; }
                self.b.ins().jump(merge, &[ev]);
                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
                Some((self.b.block_params(merge)[0], FKind::F))
            }
            Expr::Block { stmts, tail } => {
                for s in stmts { self.stmt(s)?; }
                self.expr(tail.as_ref()?)
            }
            _ => None,
        }
    }

    // f64 % and ** through the Rust helpers, bit-identical to the interpreter
    fn libcall(&mut self, which: &str, a: Value, bv: Value) -> Option<Value> {
        let id = *self.libm.get(which)?;
        let fref = self.module.declare_func_in_func(id, self.b.func);
        let inst = self.b.ins().call(fref, &[a, bv]);
        Some(self.b.inst_results(inst)[0])
    }
}

// ---------------------------------------------------------------------------
// Numeric (mixed int/float) code generation. Same all-i64 ABI as the i64 track
// (deopt_ptr + i64 args → i64), so it reuses `raw_call`. All params are ints;
// f64 results are returned as raw bits (the VM reinterprets by the function's
// recorded return kind). Int ops deopt on overflow exactly like the i64 track;
// float and mixed ops mirror the interpreter's `as_f` promotion bit-for-bit.
// ---------------------------------------------------------------------------

struct NumGen<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    module: &'a mut dyn Module,
    libm: &'a HashMap<&'static str, FuncId>,
    vars: HashMap<String, (Variable, FKind)>,
    n_vars: usize,
    deopt_ptr: Value,
    deopt_block: Block,
    loops: Vec<LoopCtx>,
    returned: bool,
    ret_kind: FKind,
}

impl<'a, 'b> NumGen<'a, 'b> {
    fn new(b: &'a mut FunctionBuilder<'b>, module: &'a mut dyn Module,
           libm: &'a HashMap<&'static str, FuncId>, params: &[String], ret_kind: FKind) -> Self {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let deopt_ptr = b.block_params(entry)[0];
        let param_vals: Vec<Value> = b.block_params(entry)[1..].to_vec();
        let deopt_block = b.create_block();
        let mut g = NumGen {
            b, module, libm, vars: HashMap::new(), n_vars: 0,
            deopt_ptr, deopt_block, loops: Vec::new(), returned: false, ret_kind,
        };
        for (i, p) in params.iter().enumerate() {
            let v = g.declare(p, FKind::I); // all params are ints
            g.b.def_var(v, param_vals[i]);
        }
        g
    }

    fn declare(&mut self, name: &str, kind: FKind) -> Variable {
        if let Some((v, _)) = self.vars.get(name) { return *v; }
        let v = Variable::new(self.n_vars);
        self.n_vars += 1;
        self.b.declare_var(v, if kind == FKind::F { types::F64 } else { I64 });
        self.vars.insert(name.to_string(), (v, kind));
        v
    }
    fn fresh(&mut self) -> Variable {
        let v = Variable::new(self.n_vars); self.n_vars += 1;
        self.b.declare_var(v, I64); v
    }
    fn to_f(&mut self, v: Value, k: FKind) -> Value {
        match k { FKind::F => v, FKind::I => self.b.ins().fcvt_from_sint(types::F64, v) }
    }
    fn guard_deopt(&mut self, cond: Value) {
        let cont = self.b.create_block();
        self.b.ins().brif(cond, self.deopt_block, &[], cont, &[]);
        self.b.switch_to_block(cont);
        self.b.seal_block(cont);
    }

    fn lower(&mut self, body: &[Stmt]) -> Option<()> {
        for s in body { self.stmt(s)?; }
        if !self.returned {
            let z = self.b.ins().iconst(I64, 0);
            self.b.ins().return_(&[z]);
        }
        self.b.switch_to_block(self.deopt_block);
        self.b.seal_block(self.deopt_block);
        let one = self.b.ins().iconst(I64, 1);
        self.b.ins().store(MemFlags::trusted(), one, self.deopt_ptr, 0);
        let z = self.b.ins().iconst(I64, 0);
        self.b.ins().return_(&[z]);
        Some(())
    }

    // convert a produced value to the raw i64 the ABI returns
    fn to_ret_bits(&mut self, v: Value, k: FKind) -> Value {
        match self.ret_kind {
            FKind::I => v, // NumCheck guarantees k == I here
            FKind::F => { let f = self.to_f(v, k); self.b.ins().bitcast(I64, MemFlags::new(), f) }
        }
    }

    fn stmt(&mut self, s: &Stmt) -> Option<()> {
        if self.returned { return Some(()); }
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                let (v, k) = self.expr(value)?;
                let var = self.declare(name, k);
                self.b.def_var(var, v);
            }
            Stmt::Expr(e) => { self.expr(e)?; }
            Stmt::Return(Some(e)) => {
                let (v, k) = self.expr(e)?;
                let bits = self.to_ret_bits(v, k);
                self.b.ins().return_(&[bits]);
                self.returned = true;
            }
            Stmt::If { cond, then, els } => {
                let c = self.cond(cond)?;
                let (tb, eb, mb) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                self.b.ins().brif(c, tb, &[], eb, &[]);
                self.b.switch_to_block(tb); self.b.seal_block(tb); self.returned = false;
                for s in then { self.stmt(s)?; }
                if !self.returned { self.b.ins().jump(mb, &[]); }
                self.b.switch_to_block(eb); self.b.seal_block(eb); self.returned = false;
                if let Some(els) = els { for s in els { self.stmt(s)?; } }
                if !self.returned { self.b.ins().jump(mb, &[]); }
                self.b.switch_to_block(mb); self.b.seal_block(mb); self.returned = false;
            }
            Stmt::While { cond, body } => {
                let (h, bb, ex) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                self.b.ins().jump(h, &[]);
                self.b.switch_to_block(h);
                let c = self.cond(cond)?;
                self.b.ins().brif(c, bb, &[], ex, &[]);
                self.b.switch_to_block(bb); self.b.seal_block(bb);
                self.loops.push(LoopCtx { header: h, exit: ex }); self.returned = false;
                for s in body { self.stmt(s)?; }
                if !self.returned { self.b.ins().jump(h, &[]); }
                self.loops.pop();
                self.b.seal_block(h);
                self.b.switch_to_block(ex); self.b.seal_block(ex); self.returned = false;
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                let (s0, sk) = self.expr(start)?; if sk != FKind::I { return None; }
                let (e0, ek) = self.expr(end)?; if ek != FKind::I { return None; }
                let iv = self.declare(var, FKind::I); self.b.def_var(iv, s0);
                let lim = self.fresh(); self.b.def_var(lim, e0);
                let (h, bb, ex) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                self.b.ins().jump(h, &[]);
                self.b.switch_to_block(h);
                let i = self.b.use_var(iv); let l = self.b.use_var(lim);
                let cc = if *inclusive { IntCC::SignedLessThanOrEqual } else { IntCC::SignedLessThan };
                let c = self.b.ins().icmp(cc, i, l);
                self.b.ins().brif(c, bb, &[], ex, &[]);
                self.b.switch_to_block(bb); self.b.seal_block(bb);
                self.loops.push(LoopCtx { header: h, exit: ex }); self.returned = false;
                for s in body { self.stmt(s)?; }
                if !self.returned {
                    let i = self.b.use_var(iv);
                    let n = self.b.ins().iadd_imm(i, 1);
                    self.b.def_var(iv, n);
                    self.b.ins().jump(h, &[]);
                }
                self.loops.pop();
                self.b.seal_block(h);
                self.b.switch_to_block(ex); self.b.seal_block(ex); self.returned = false;
            }
            Stmt::Break(None) => { let ex = self.loops.last()?.exit; self.b.ins().jump(ex, &[]); self.returned = true; }
            Stmt::Continue => { let h = self.loops.last()?.header; self.b.ins().jump(h, &[]); self.returned = true; }
            _ => return None,
        }
        Some(())
    }

    // condition → i64 0/1. Both-int comparisons are exact (icmp), matching the
    // interpreter's Int fast path; any float operand compares via as_f (fcmp).
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
                let (tb, eb, mb) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                self.b.append_block_param(mb, I64);
                self.b.ins().brif(a, tb, &[], eb, &[]);
                self.b.switch_to_block(tb); self.b.seal_block(tb);
                if matches!(op, BinOp::And) { let v = self.cond(rhs)?; self.b.ins().jump(mb, &[v]); }
                else { let o = self.b.ins().iconst(I64, 1); self.b.ins().jump(mb, &[o]); }
                self.b.switch_to_block(eb); self.b.seal_block(eb);
                if matches!(op, BinOp::And) { let z = self.b.ins().iconst(I64, 0); self.b.ins().jump(mb, &[z]); }
                else { let v = self.cond(rhs)?; self.b.ins().jump(mb, &[v]); }
                self.b.switch_to_block(mb); self.b.seal_block(mb);
                Some(self.b.block_params(mb)[0])
            }
            Expr::Binary { op, lhs, rhs } => {
                let (a, ak) = self.expr(lhs)?;
                let (bv, bk) = self.expr(rhs)?;
                let c = if ak == FKind::I && bk == FKind::I {
                    let cc = match op {
                        BinOp::Eq => IntCC::Equal, BinOp::Ne => IntCC::NotEqual,
                        BinOp::Lt => IntCC::SignedLessThan, BinOp::Le => IntCC::SignedLessThanOrEqual,
                        BinOp::Gt => IntCC::SignedGreaterThan, BinOp::Ge => IntCC::SignedGreaterThanOrEqual,
                        _ => return None,
                    };
                    self.b.ins().icmp(cc, a, bv)
                } else {
                    let a = self.to_f(a, ak); let bv = self.to_f(bv, bk);
                    let cc = match op {
                        BinOp::Eq => FloatCC::Equal, BinOp::Ne => FloatCC::NotEqual,
                        BinOp::Lt => FloatCC::LessThan, BinOp::Le => FloatCC::LessThanOrEqual,
                        BinOp::Gt => FloatCC::GreaterThan, BinOp::Ge => FloatCC::GreaterThanOrEqual,
                        _ => return None,
                    };
                    self.b.ins().fcmp(cc, a, bv)
                };
                Some(self.b.ins().uextend(I64, c))
            }
            _ => None,
        }
    }

    fn expr(&mut self, e: &Expr) -> Option<(Value, FKind)> {
        match e {
            Expr::At { expr, .. } => self.expr(expr),
            Expr::Float(x) => Some((self.b.ins().f64const(*x), FKind::F)),
            Expr::Int(n) => Some((self.b.ins().iconst(I64, *n), FKind::I)),
            Expr::Ident(name) => { let (v, k) = *self.vars.get(name)?; Some((self.b.use_var(v), k)) }
            Expr::Unary { op: UnOp::Neg, expr } => {
                let (v, k) = self.expr(expr)?;
                match k {
                    FKind::F => Some((self.b.ins().fneg(v), FKind::F)),
                    FKind::I => {
                        let min = self.b.ins().iconst(I64, i64::MIN);
                        let bad = self.b.ins().icmp(IntCC::Equal, v, min);
                        let bad = self.b.ins().uextend(I64, bad);
                        self.guard_deopt(bad);
                        Some((self.b.ins().ineg(v), FKind::I))
                    }
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let (a, ak) = self.expr(lhs)?;
                let (bv, bk) = self.expr(rhs)?;
                if ak == FKind::I && bk == FKind::I {
                    Some((self.int_binop(*op, a, bv)?, FKind::I))
                } else {
                    let a = self.to_f(a, ak); let bv = self.to_f(bv, bk);
                    let v = match op {
                        BinOp::Add => self.b.ins().fadd(a, bv),
                        BinOp::Sub => self.b.ins().fsub(a, bv),
                        BinOp::Mul => self.b.ins().fmul(a, bv),
                        BinOp::Div => self.b.ins().fdiv(a, bv),
                        BinOp::Rem => self.libcall("fmod", a, bv)?,
                        BinOp::Pow => self.libcall("fpow", a, bv)?,
                        _ => return None,
                    };
                    Some((v, FKind::F))
                }
            }
            Expr::Call { callee, args } => {
                if callee == "to_float" && args.len() == 1 {
                    let (v, k) = self.expr(&args[0])?;
                    return Some((self.to_f(v, k), FKind::F));
                }
                if callee == "to_int" && args.len() == 1 {
                    let (v, k) = self.expr(&args[0])?;
                    if k != FKind::F { return None; }
                    // truncation toward zero, saturating — matches Rust `f64 as i64`
                    return Some((self.b.ins().fcvt_to_sint_sat(I64, v), FKind::I));
                }
                None
            }
            Expr::If { cond, then, els } => {
                let c = self.cond(cond)?;
                let (tb, eb, mb) = (self.b.create_block(), self.b.create_block(), self.b.create_block());
                // evaluate a probe of the then-branch kind by structure is hard; both
                // arms must share a kind (NumCheck guarantees it) — assume via then
                let (tv, tk);
                self.b.append_block_param(mb, I64); // carry as bits; reinterpret after
                self.b.ins().brif(c, tb, &[], eb, &[]);
                self.b.switch_to_block(tb); self.b.seal_block(tb);
                let (v, k) = self.expr(then)?; tk = k;
                let tvb = match k { FKind::I => v, FKind::F => self.b.ins().bitcast(I64, MemFlags::new(), v) };
                tv = tvb; let _ = tv;
                self.b.ins().jump(mb, &[tvb]);
                self.b.switch_to_block(eb); self.b.seal_block(eb);
                let (ev, ek) = self.expr(els)?;
                if ek != tk { return None; }
                let evb = match ek { FKind::I => ev, FKind::F => self.b.ins().bitcast(I64, MemFlags::new(), ev) };
                self.b.ins().jump(mb, &[evb]);
                self.b.switch_to_block(mb); self.b.seal_block(mb);
                let raw = self.b.block_params(mb)[0];
                let out = match tk { FKind::I => raw, FKind::F => self.b.ins().bitcast(types::F64, MemFlags::new(), raw) };
                Some((out, tk))
            }
            Expr::Block { stmts, tail } => {
                for s in stmts { self.stmt(s)?; }
                self.expr(tail.as_ref()?)
            }
            _ => None,
        }
    }

    // integer binary op with overflow/zero deopt guards — identical to the i64
    // track, so a deopt re-run on the VM is observationally the same.
    fn int_binop(&mut self, op: BinOp, a: Value, b: Value) -> Option<Value> {
        use BinOp::*;
        Some(match op {
            Add => self.add_ovf(a, b),
            Sub => self.sub_ovf(a, b),
            Mul => self.mul_ovf(a, b),
            Div | Rem => {
                let z = self.b.ins().iconst(I64, 0);
                let iz = self.b.ins().icmp(IntCC::Equal, b, z);
                let iz = self.b.ins().uextend(I64, iz);
                self.guard_deopt(iz);
                let min = self.b.ins().iconst(I64, i64::MIN);
                let n1 = self.b.ins().iconst(I64, -1);
                let am = self.b.ins().icmp(IntCC::Equal, a, min);
                let bn = self.b.ins().icmp(IntCC::Equal, b, n1);
                let both = self.b.ins().band(am, bn);
                let both = self.b.ins().uextend(I64, both);
                self.guard_deopt(both);
                if matches!(op, Div) { self.b.ins().sdiv(a, b) } else { self.b.ins().srem(a, b) }
            }
            Pow => return self.pow_ovf(a, b),
            _ => return None,
        })
    }
    fn add_ovf(&mut self, a: Value, b: Value) -> Value {
        let r = self.b.ins().iadd(a, b);
        let t1 = self.b.ins().bxor(a, r); let t2 = self.b.ins().bxor(b, r);
        let t3 = self.b.ins().band(t1, t2);
        let z = self.b.ins().iconst(I64, 0);
        let o = self.b.ins().icmp(IntCC::SignedLessThan, t3, z);
        let o = self.b.ins().uextend(I64, o); self.guard_deopt(o); r
    }
    fn sub_ovf(&mut self, a: Value, b: Value) -> Value {
        let r = self.b.ins().isub(a, b);
        let t1 = self.b.ins().bxor(a, b); let t2 = self.b.ins().bxor(a, r);
        let t3 = self.b.ins().band(t1, t2);
        let z = self.b.ins().iconst(I64, 0);
        let o = self.b.ins().icmp(IntCC::SignedLessThan, t3, z);
        let o = self.b.ins().uextend(I64, o); self.guard_deopt(o); r
    }
    fn mul_ovf(&mut self, a: Value, b: Value) -> Value {
        let a128 = self.b.ins().sextend(I128, a); let b128 = self.b.ins().sextend(I128, b);
        let m = self.b.ins().imul(a128, b128);
        let m64 = self.b.ins().ireduce(I64, m);
        let back = self.b.ins().sextend(I128, m64);
        let ok = self.b.ins().icmp(IntCC::Equal, m, back);
        let bad = self.b.ins().bnot(ok);
        let one = self.b.ins().iconst(I64, 1);
        let bad = self.b.ins().uextend(I64, bad);
        let bad = self.b.ins().band(bad, one);
        self.guard_deopt(bad); m64
    }
    fn pow_ovf(&mut self, base0: Value, exp0: Value) -> Option<Value> {
        let z = self.b.ins().iconst(I64, 0);
        let neg = self.b.ins().icmp(IntCC::SignedLessThan, exp0, z);
        let neg = self.b.ins().uextend(I64, neg); self.guard_deopt(neg);
        let umax = self.b.ins().iconst(I64, u32::MAX as i64);
        let big = self.b.ins().icmp(IntCC::SignedGreaterThan, exp0, umax);
        let big = self.b.ins().uextend(I64, big); self.guard_deopt(big);
        let acc = self.fresh(); let base = self.fresh(); let exp = self.fresh();
        let one = self.b.ins().iconst(I64, 1);
        self.b.def_var(acc, one); self.b.def_var(base, base0); self.b.def_var(exp, exp0);
        let (h, odd, sh, mb) = (self.b.create_block(), self.b.create_block(), self.b.create_block(), self.b.create_block());
        self.b.append_block_param(mb, I64);
        let isz = self.b.ins().icmp(IntCC::Equal, exp0, z);
        let one2 = self.b.ins().iconst(I64, 1);
        self.b.ins().brif(isz, mb, &[one2], h, &[]);
        self.b.switch_to_block(h);
        let e = self.b.use_var(exp);
        let bit = self.b.ins().band_imm(e, 1);
        self.b.ins().brif(bit, odd, &[], sh, &[]);
        self.b.switch_to_block(odd); self.b.seal_block(odd);
        let av = self.b.use_var(acc); let bvv = self.b.use_var(base);
        let a2 = self.mul_ovf(av, bvv); self.b.def_var(acc, a2);
        let e = self.b.use_var(exp);
        let is1 = self.b.ins().icmp_imm(IntCC::Equal, e, 1);
        self.b.ins().brif(is1, mb, &[a2], sh, &[]);
        self.b.switch_to_block(sh); self.b.seal_block(sh);
        let e = self.b.use_var(exp);
        let e2 = self.b.ins().ushr_imm(e, 1); self.b.def_var(exp, e2);
        let bvv = self.b.use_var(base);
        let sq = self.mul_ovf(bvv, bvv); self.b.def_var(base, sq);
        self.b.ins().jump(h, &[]);
        self.b.seal_block(h);
        self.b.switch_to_block(mb); self.b.seal_block(mb);
        Some(self.b.block_params(mb)[0])
    }

    fn libcall(&mut self, which: &str, a: Value, bv: Value) -> Option<Value> {
        let id = *self.libm.get(which)?;
        let fref = self.module.declare_func_in_func(id, self.b.func);
        let inst = self.b.ins().call(fref, &[a, bv]);
        Some(self.b.inst_results(inst)[0])
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
    numeric: HashSet<String>,    // mixed int/float track (disjoint)
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
        let (numeric, _nret) = numeric_eligible_set(prog, &eligible, &feligible);
        let mut callees: HashMap<String, Vec<String>> = HashMap::new();
        let claimed = |n: &String| eligible.contains(n) || feligible.contains(n) || numeric.contains(n);
        for item in &prog.items {
            if let Item::Func(f) = item {
                if claimed(&f.name) {
                    let cs: Vec<String> = collect_calls(&f.body).into_iter()
                        .filter(|c| claimed(c)).collect();
                    callees.insert(f.name.clone(), cs);
                }
            }
        }
        TieredJit {
            prog, eligible, feligible, numeric, callees,
            threshold: threshold.max(1),
            jits: std::cell::RefCell::new(Vec::new()),
            location: std::cell::RefCell::new(HashMap::new()),
            compiled_order: std::cell::RefCell::new(Vec::new()),
            backend_failed: std::cell::Cell::new(false),
        }
    }

    pub fn is_eligible(&self, name: &str) -> bool {
        self.eligible.contains(name) || self.feligible.contains(name) || self.numeric.contains(name)
    }

    // Eagerly compile every eligible function whose body contains a loop, before
    // execution starts. A loop means the function does real work per call, so a
    // function called only once from `main` (a compute kernel — a sieve, a
    // mandelbrot counter) would otherwise never cross the call-count threshold
    // and would crawl on the interpreter tier. The one-time compile cost is
    // negligible next to the loop it accelerates; correctness is unchanged
    // (deopt still guards every path).
    pub fn warm_loops(&self) {
        let mut roots: Vec<String> = Vec::new();
        for item in &self.prog.items {
            if let Item::Func(f) = item {
                // `#[simd]` is a JIT hint: it forces eager native compilation of
                // the function's numeric/array kernel, exactly like `#[hot]`. (True
                // Cranelift SIMD-type auto-vectorization is a documented future
                // deepening — this attribute honestly means "compile this kernel
                // up-front", not "it is vectorized".)
                let hinted_hot = f.attrs.iter().any(|a| a.name == "hot" || a.name == "simd");
                let hinted_cold = f.attrs.iter().any(|a| a.name == "cold");
                // #[hot]/#[simd] compile up-front unconditionally; #[cold] never
                // warms; otherwise loop-bearing kernels warm as before.
                if self.is_eligible(&f.name) && !hinted_cold && (hinted_hot || body_has_loop(&f.body)) {
                    roots.push(f.name.clone());
                }
            }
        }
        for r in roots { self.compile_closure(&r); }
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
    pub fn is_compiled_num(&self, name: &str) -> bool {
        match self.location.borrow().get(name) {
            Some(idx) => self.jits.borrow()[*idx].is_compiled_num(name),
            None => false,
        }
    }
    pub fn num_ret_is_float(&self, name: &str) -> bool {
        match self.location.borrow().get(name) {
            Some(idx) => self.jits.borrow()[*idx].num_ret_is_float(name),
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

    // caller must check `is_compiled_num` first
    pub fn raw_call_num(&self, name: &str, args: &[i64]) -> (i64, bool) {
        let idx = self.location.borrow()[name];
        self.jits.borrow()[idx].raw_call_num(name, args)
    }
}

// ---------------------------------------------------------------------------
// Per-function code generation
// ---------------------------------------------------------------------------

// Is `e` a fill value that is the same on every iteration of a loop counted by
// `var`? Conservative: literals and outer variables (≠ var) are invariant; simple
// unary/binary/`@` compositions of invariants are too; anything else returns false
// (so we don't fuse). Never returns true for an expression that reads `var`.
fn fill_value_ok(e: &Expr, var: &str) -> bool {
    match e {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Null => true,
        Expr::Ident(n) => n != var,
        Expr::At { expr, .. } => fill_value_ok(expr, var),
        Expr::Unary { expr, .. } => fill_value_ok(expr, var),
        Expr::Binary { lhs, rhs, .. } => fill_value_ok(lhs, var) && fill_value_ok(rhs, var),
        _ => false,
    }
}

struct LoopCtx { header: Block, exit: Block }

struct FnGen<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    module: &'a mut dyn Module,
    ids: &'a HashMap<String, (FuncId, usize)>,
    helpers: &'a HashMap<&'static str, FuncId>,
    arrays: HashSet<String>, // local array vars: I64 arena handles
    // local struct vars: arena handle + field order (field name -> slot index)
    structs: HashMap<String, Vec<String>>,
    vars: HashMap<String, Variable>,
    n_vars: usize,
    deopt_ptr: Value,
    deopt_block: Block,
    loops: Vec<LoopCtx>,
    returned: bool,
    // emit the overflow check without the `sadd_overflow`/`ssub_overflow`
    // intrinsics (Cranelift's riscv64 backend doesn't lower them) — the portable
    // sign-bit test instead, byte-identical in which inputs it flags.
    manual_ovf: bool,
}

impl<'a, 'b> FnGen<'a, 'b> {
    fn new(b: &'a mut FunctionBuilder<'b>, module: &'a mut dyn Module,
           ids: &'a HashMap<String, (FuncId, usize)>,
           helpers: &'a HashMap<&'static str, FuncId>,
           arrays: HashSet<String>, structs: HashMap<String, Vec<String>>,
           params: &[String], manual_ovf: bool) -> Self {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let deopt_ptr = b.block_params(entry)[0];
        let param_vals: Vec<Value> = b.block_params(entry)[1..].to_vec();
        let deopt_block = b.create_block();
        let mut g = FnGen {
            b, module, ids, helpers, arrays, structs, vars: HashMap::new(), n_vars: 0,
            deopt_ptr, deopt_block, loops: Vec::new(), returned: false, manual_ovf,
        };
        for (i, p) in params.iter().enumerate() {
            let v = g.declare(p);
            g.b.def_var(v, param_vals[i]);
        }
        g
    }

    // call an imported runtime helper; `fallible` loads and guards the deopt flag
    fn helper_call(&mut self, name: &str, args: &[Value], fallible: bool) -> Option<Value> {
        let id = *self.helpers.get(name)?;
        let fref = self.module.declare_func_in_func(id, self.b.func);
        let inst = self.b.ins().call(fref, args);
        let res = self.b.inst_results(inst).first().copied();
        if fallible {
            let flag = self.b.ins().load(I64, MemFlags::trusted(), self.deopt_ptr, 0);
            self.guard_deopt(flag);
        }
        res
    }

    // build a fresh arena array from a literal and bind it to `name`
    fn build_array(&mut self, name: &str, elems: &[Expr]) -> Option<()> {
        let h = self.helper_call("nova_arr_new", &[], false)?;
        let var = self.declare(name);
        self.b.def_var(var, h);
        for el in elems {
            let v = self.expr(el)?;
            let h = self.b.use_var(var);
            self.helper_call("nova_arr_push", &[h, v], false);
        }
        Some(())
    }

    // build a fresh arena block from a struct literal (one slot per field, in
    // shape order) and bind it to `name`
    fn build_struct(&mut self, name: &str, fields: &[(String, Expr)]) -> Option<()> {
        let h = self.helper_call("nova_arr_new", &[], false)?;
        let var = self.declare(name);
        self.b.def_var(var, h);
        for (_, fe) in fields {
            let v = self.expr(fe)?;
            let h = self.b.use_var(var);
            self.helper_call("nova_arr_push", &[h, v], false);
        }
        Some(())
    }

    // slot index for `field` on struct var `name`
    fn struct_slot(&self, name: &str, field: &str) -> Option<i64> {
        self.structs.get(name)?.iter().position(|f| f == field).map(|i| i as i64)
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

    // Recognise `s = []` immediately followed by `for _ in 0..N { push(s, V) }`
    // where V does not depend on the loop counter — the idiomatic way to build a
    // filled array. Returns (array_name, count_expr, value_expr) so the caller can
    // emit a single `nova_arr_fill(N, V)` instead of one `push` call per element.
    fn fill_fusion<'s>(&self, stmts: &'s [Stmt]) -> Option<(&'s str, &'s Expr, &'s Expr)> {
        if stmts.len() < 2 { return None; }
        // stmts[0]: `name = []` (empty array literal) on a known array var
        let name = match &stmts[0] {
            Stmt::Let { name, value, .. } | Stmt::Assign { name, value } => {
                match strip_at(value) { Expr::Array(e) if e.is_empty() => name.as_str(), _ => return None }
            }
            _ => return None,
        };
        if !self.arrays.contains(name) { return None; }
        // stmts[1]: `for VAR in 0..END { push(name, VALUE) }`, exclusive, one stmt
        let (var, end, count_body) = match &stmts[1] {
            Stmt::ForRange { var, start, end, inclusive: false, body } => {
                match strip_at(start) { Expr::Int(0) => (var.as_str(), end, body), _ => return None }
            }
            _ => return None,
        };
        if count_body.len() != 1 { return None; }
        let value = match &count_body[0] {
            Stmt::Expr(e) => match strip_at(e) {
                Expr::Call { callee, args } if callee == "push" && args.len() == 2
                    && as_ident(&args[0]) == Some(name) => &args[1],
                _ => return None,
            },
            _ => return None,
        };
        // the pushed value must be identical every iteration: only fuse when it is
        // provably loop-invariant (a constant/outer variable, conservatively). The
        // fill loop's body is just the push, so an outer variable can't change
        // across iterations; anything we can't prove invariant falls back to the
        // ordinary push-loop lowering (still correct, just not fused).
        if !fill_value_ok(value, var) { return None; }
        Some((name, end, value))
    }

    fn lower(&mut self, body: &[Stmt]) -> Option<()> {
        let mut i = 0;
        while i < body.len() {
            if self.returned { break; }
            if let Some((name, count, value)) = self.fill_fusion(&body[i..]) {
                let n = self.expr(count)?;
                let v = self.expr(value)?;
                let h = self.helper_call("nova_arr_fill", &[n, v], false)?;
                let var = self.declare(name);
                self.b.def_var(var, h);
                i += 2;
                continue;
            }
            self.stmt(&body[i])?;
            i += 1;
        }
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
                if self.arrays.contains(name) {
                    match strip_at(value) {
                        Expr::Array(elems) => { self.build_array(name, elems)?; }
                        Expr::Ident(src) => {
                            // alias: same handle, same shared storage as the interp's Rc
                            let sv = *self.vars.get(src)?;
                            let h = self.b.use_var(sv);
                            let var = self.declare(name);
                            self.b.def_var(var, h);
                        }
                        _ => return None,
                    }
                } else if self.structs.contains_key(name) {
                    match strip_at(value) {
                        Expr::StructLit { fields, .. } => { self.build_struct(name, fields)?; }
                        Expr::Ident(src) => {
                            // alias: same handle (shared, like the interp's Rc struct)
                            let sv = *self.vars.get(src)?;
                            let h = self.b.use_var(sv);
                            let var = self.declare(name);
                            self.b.def_var(var, h);
                        }
                        _ => return None,
                    }
                } else {
                    let v = self.expr(value)?;
                    let var = self.declare(name);
                    self.b.def_var(var, v);
                }
            }
            Stmt::Expr(e) => {
                if let Expr::Call { callee, args } = strip_at(e) {
                    if callee == "push" && array_builtin_call(callee, args, &self.arrays) {
                        let name = as_ident(&args[0])?;
                        let av = *self.vars.get(name)?;
                        let h = self.b.use_var(av);
                        let v = self.expr(&args[1])?;
                        self.helper_call("nova_arr_push", &[h, v], false);
                        return Some(());
                    }
                }
                self.expr(e)?;
            }
            Stmt::Return(Some(e)) => {
                let v = self.expr(e)?;
                self.b.ins().return_(&[v]);
                self.returned = true;
            }
            Stmt::IndexAssign { base, index, value } => {
                let name = as_ident(base)?;
                if !self.arrays.contains(name) { return None; }
                let av = *self.vars.get(name)?;
                let h = self.b.use_var(av);
                let i = self.expr(index)?;
                let v = self.expr(value)?;
                self.helper_call("nova_arr_set", &[self.deopt_ptr, h, i, v], true);
            }
            // p.field = v on a local struct var: a slot store (index known at
            // compile time, always in bounds by construction)
            Stmt::FieldAssign { base, field, value } => {
                let name = as_ident(base)?;
                let slot = self.struct_slot(name, field)?;
                let sv = *self.vars.get(name)?;
                let h = self.b.use_var(sv);
                let i = self.b.ins().iconst(I64, slot);
                let v = self.expr(value)?;
                self.helper_call("nova_arr_set", &[self.deopt_ptr, h, i, v], true);
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
            Expr::Index { base, index } => {
                let name = as_ident(base)?;
                if !self.arrays.contains(name) { return None; }
                let av = *self.vars.get(name)?;
                let h = self.b.use_var(av);
                let i = self.expr(index)?;
                self.helper_call("nova_arr_get", &[self.deopt_ptr, h, i], true)
            }
            // p.field on a local struct var: a slot load (compile-time index,
            // always in bounds by construction)
            Expr::Field { base, field } => {
                let name = as_ident(base)?;
                let slot = self.struct_slot(name, field)?;
                let sv = *self.vars.get(name)?;
                let h = self.b.use_var(sv);
                let i = self.b.ins().iconst(I64, slot);
                self.helper_call("nova_arr_get", &[self.deopt_ptr, h, i], true)
            }
            Expr::Call { callee, args } => {
                if array_builtin_call(callee, args, &self.arrays) {
                    let name = as_ident(&args[0])?;
                    let av = *self.vars.get(name)?;
                    let h = self.b.use_var(av);
                    return match callee.as_str() {
                        "len" => self.helper_call("nova_arr_len", &[h], false),
                        // empty pop -> interp yields null: deopt re-runs on the VM
                        "pop" => self.helper_call("nova_arr_pop", &[self.deopt_ptr, h], true),
                        _ => None, // push is statement-only
                    };
                }
                self.call(callee, args)
            }
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
            Pow => self.pow_checked(a, b),
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
            // Both operands are i64 on this track, so compare as integers —
            // exactly what the interpreter does for `Int op Int` (interp.rs:
            // `a < b`). The earlier float detour (fcvt+fcmp) was slower *and*
            // lossy for |value| > 2^53; `icmp` is faster and byte-identical.
            Eq => { let c = self.b.ins().icmp(IntCC::Equal, a, b); self.b.ins().uextend(I64, c) }
            Ne => { let c = self.b.ins().icmp(IntCC::NotEqual, a, b); self.b.ins().uextend(I64, c) }
            Lt => { let c = self.b.ins().icmp(IntCC::SignedLessThan, a, b); self.b.ins().uextend(I64, c) }
            Le => { let c = self.b.ins().icmp(IntCC::SignedLessThanOrEqual, a, b); self.b.ins().uextend(I64, c) }
            Gt => { let c = self.b.ins().icmp(IntCC::SignedGreaterThan, a, b); self.b.ins().uextend(I64, c) }
            Ge => { let c = self.b.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b); self.b.ins().uextend(I64, c) }
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

    // Signed add/sub with overflow deopt. Cranelift's `sadd_overflow` /
    // `ssub_overflow` emit the result plus a hardware overflow flag in one op —
    // the flag is set on exactly the inputs where `i64::checked_add`/`checked_sub`
    // return None, so this is byte-identical to the interpreter (which promotes to
    // BigInt on overflow) while replacing the 5-instruction sign-bit trick.
    fn add_checked(&mut self, a: Value, b: Value) -> Value {
        if self.manual_ovf {
            // s = a + b; overflow iff a,b share a sign but s differs:
            // ((a ^ s) & (b ^ s)) < 0.
            let s = self.b.ins().iadd(a, b);
            let axs = self.b.ins().bxor(a, s);
            let bxs = self.b.ins().bxor(b, s);
            let both = self.b.ins().band(axs, bxs);
            let ovf = self.b.ins().icmp_imm(IntCC::SignedLessThan, both, 0);
            self.guard_deopt(ovf);
            return s;
        }
        let (r, ovf) = self.b.ins().sadd_overflow(a, b);
        self.guard_deopt(ovf);
        r
    }

    fn sub_checked(&mut self, a: Value, b: Value) -> Value {
        if self.manual_ovf {
            // d = a - b; overflow iff a,b differ in sign and d differs from a:
            // ((a ^ b) & (a ^ d)) < 0.
            let d = self.b.ins().isub(a, b);
            let axb = self.b.ins().bxor(a, b);
            let axd = self.b.ins().bxor(a, d);
            let both = self.b.ins().band(axb, axd);
            let ovf = self.b.ins().icmp_imm(IntCC::SignedLessThan, both, 0);
            self.guard_deopt(ovf);
            return d;
        }
        let (r, ovf) = self.b.ins().ssub_overflow(a, b);
        self.guard_deopt(ovf);
        r
    }

    // Integer `**`, transcribing i64::checked_pow's square-and-multiply exactly:
    // any intermediate overflow deopts (the interpreter promotes to BigInt), a
    // negative exponent deopts (interp returns a Float), and exponents beyond
    // u32::MAX deopt (interp switches to powf).
    fn pow_checked(&mut self, base0: Value, exp0: Value) -> Value {
        let zero = self.b.ins().iconst(I64, 0);
        let neg = self.b.ins().icmp(IntCC::SignedLessThan, exp0, zero);
        let neg = self.b.ins().uextend(I64, neg);
        self.guard_deopt(neg);
        let umax = self.b.ins().iconst(I64, u32::MAX as i64);
        let big = self.b.ins().icmp(IntCC::SignedGreaterThan, exp0, umax);
        let big = self.b.ins().uextend(I64, big);
        self.guard_deopt(big);

        let acc_v = self.fresh_var();
        let base_v = self.fresh_var();
        let exp_v = self.fresh_var();
        let one = self.b.ins().iconst(I64, 1);
        self.b.def_var(acc_v, one);
        self.b.def_var(base_v, base0);
        self.b.def_var(exp_v, exp0);

        let header = self.b.create_block();
        let odd = self.b.create_block();
        let shift = self.b.create_block();
        let merge = self.b.create_block();
        self.b.append_block_param(merge, I64);

        // exp == 0 -> 1
        let is_zero = self.b.ins().icmp(IntCC::Equal, exp0, zero);
        let one2 = self.b.ins().iconst(I64, 1);
        self.b.ins().brif(is_zero, merge, &[one2], header, &[]);

        self.b.switch_to_block(header);
        let e = self.b.use_var(exp_v);
        let bit = self.b.ins().band_imm(e, 1);
        self.b.ins().brif(bit, odd, &[], shift, &[]);

        self.b.switch_to_block(odd);
        self.b.seal_block(odd);
        let acc = self.b.use_var(acc_v);
        let bas = self.b.use_var(base_v);
        let acc2 = self.mul_checked(acc, bas);
        self.b.def_var(acc_v, acc2);
        let e = self.b.use_var(exp_v);
        let is_one = self.b.ins().icmp_imm(IntCC::Equal, e, 1);
        self.b.ins().brif(is_one, merge, &[acc2], shift, &[]);

        self.b.switch_to_block(shift);
        self.b.seal_block(shift);
        let e = self.b.use_var(exp_v);
        let e2 = self.b.ins().ushr_imm(e, 1);
        self.b.def_var(exp_v, e2);
        let bas = self.b.use_var(base_v);
        let sq = self.mul_checked(bas, bas);
        self.b.def_var(base_v, sq);
        self.b.ins().jump(header, &[]);

        self.b.seal_block(header);
        self.b.switch_to_block(merge);
        self.b.seal_block(merge);
        self.b.block_params(merge)[0]
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

// ---------------------------------------------------------------------------
// Native object-code AOT backend: Cranelift IR -> relocatable .o (no C for
// program logic). `ObjectModule` implements the same `cranelift_module::Module`
// trait as the in-memory `JITModule`, so the identical `FnGen` lowering that the
// JIT uses emits a real object file here. The system linker (`cc`) is used
// purely as the libc linker driver — it compiles none of the program.
//
// Covers the full numeric surface the JIT compiles: `main` may be one or more
// `print(<expr>)` statements, each value produced on the integer (`FnGen`, incl.
// local integer arrays/structs), float (`FloatGen`), or numeric-mixed (`NumGen`)
// track. A single-print integer program keeps an in-IR `write(2)` itoa (so it
// links only libc); multi-print and float/array programs print through the
// linked runtime (`runtime/nova_native_rt.c`). Anything not numeric-native (a
// string/bool print, or a `main` with lets/loops) returns `None` so the caller
// falls back to the C/embed AOT — never wrong output.
// ---------------------------------------------------------------------------

// Strip position wrappers so structural matching sees the real expression.
fn unwrap_at(e: &Expr) -> &Expr {
    match e { Expr::At { expr, .. } => unwrap_at(expr), other => other }
}

// If every statement of `main`'s body is a `print(<expr>)` (as an expression
// statement or a trailing implicit/explicit return), return the printed
// arguments in order. `None` if any statement is something else — so a
// multi-print `main` builds natively, but a `main` with lets/loops/other calls
// falls back. Must yield at least one print.
fn main_print_args(f: &Func) -> Option<Vec<Expr>> {
    let mut out: Vec<Expr> = Vec::new();
    for s in &f.body {
        let e = match s {
            Stmt::Expr(e) => e,
            Stmt::Return(Some(e)) => e,
            _ => return None, // any other statement -> not a simple print main
        };
        match unwrap_at(e) {
            Expr::Call { callee, args } if callee == "print" && args.len() == 1 =>
                out.push(args[0].clone()),
            _ => return None,
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

// Which JIT track the value-producing function was compiled on, and how the
// entry must call + print it.
#[derive(Clone, Copy, PartialEq)]
enum Track { Int, Float, NumInt, NumFloat }

// A compile-time-constant argument to the value-producing function.
#[derive(Clone, Copy)]
enum ArgConst { I(i64), F(f64) }

// A resolved printed value, in source order: baked constant-string bytes (a
// read-only data object + its byte length, printed with a `write` syscall), a
// number produced by a compiled function on a numeric track, or a run-time string
// produced by a StrGen function (its FuncId + each constant-string argument boxed
// as a data object + length, printed with nova_str_print).
enum RItem {
    Str(DataId, usize),
    Num(FuncId, Track, Vec<ArgConst>),
    StrProducer(FuncId, Vec<(DataId, usize)>),
}

// Extract a constant literal argument (post-fold): an Int/Float literal or a
// negated one. `None` if the expression isn't a decidable numeric constant.
fn const_arg(e: &Expr) -> Option<ArgConst> {
    match unwrap_at(e) {
        Expr::Int(n) => Some(ArgConst::I(*n)),
        Expr::Float(x) => Some(ArgConst::F(*x)),
        Expr::Unary { op: UnOp::Neg, expr } => match const_arg(expr)? {
            ArgConst::I(n) => Some(ArgConst::I(n.wrapping_neg())),
            ArgConst::F(x) => Some(ArgConst::F(-x)),
        },
        _ => None,
    }
}

// A compile-time-constant string value: string literals, string concatenation
// (`"a" + "b"`), f-strings with constant holes (`f"n={3}"`), and `str(...)` over
// constants. Each piece is formatted exactly as Nova's own `print`/`to_string`
// (decimal ints, `true`/`false`, raw strings), so the emitted bytes are
// byte-identical by construction; the build's oracle gate is the backstop.
// Returns `None` for anything not a decidable constant string — those take the
// numeric path or fall back to the C/embed build (never wrong output). Note the
// entry rejects any `main` that isn't purely `print(...)` statements, so an
// f-string hole referencing a local variable never reaches here.
fn const_str(e: &Expr) -> Option<String> {
    match unwrap_at(e) {
        Expr::Str(s) => Some(s.clone()),
        Expr::Binary { op: BinOp::Add, lhs, rhs } =>
            Some(format!("{}{}", const_str(lhs)?, const_str(rhs)?)),
        Expr::FmtStr(parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    FmtPart::Lit(s) => out.push_str(s),
                    FmtPart::Expr(h) => out.push_str(&const_hole(h)?),
                }
            }
            Some(out)
        }
        Expr::Call { callee, args } if (callee == "str" || callee == "to_str") && args.len() == 1 =>
            const_hole(&args[0]),
        // constant case folding uses Rust's Unicode `to_uppercase`/`to_lowercase`
        // — exactly what the interpreter calls — so it is byte-identical for any
        // input (unlike the ASCII-only runtime used for a dynamic parameter).
        Expr::Call { callee, args } if callee == "upper" && args.len() == 1 =>
            Some(const_str(&args[0])?.to_uppercase()),
        Expr::Call { callee, args } if callee == "lower" && args.len() == 1 =>
            Some(const_str(&args[0])?.to_lowercase()),
        _ => None,
    }
}

// Format a constant interpolation hole / `str()` argument exactly as Nova prints
// it. Ints -> decimal, floats -> `float_str`, bools -> `true`/`false`, strings ->
// raw. Anything not a decidable constant returns `None` (leave that program to the
// C/embed fallback; the oracle gate would catch any mismatch regardless).
fn const_hole(e: &Expr) -> Option<String> {
    match unwrap_at(e) {
        Expr::Str(s) => Some(s.clone()),
        Expr::Bool(b) => Some(if *b { "true".to_string() } else { "false".to_string() }),
        other => match const_arg(other) {
            Some(ArgConst::I(n)) => Some(n.to_string()),
            Some(ArgConst::F(x)) => Some(float_str(x)),
            None => None,
        },
    }
}

// A Nova float's string form, byte-identical to `impl Display for Value`
// (interp.rs): an integral finite float prints as `{:.1}` ("5.0", "-0.0"), else
// Rust's default shortest-round-tripping f64 formatting. Reused to bake constant
// float pieces of a native string.
fn float_str(x: f64) -> String {
    if x.fract() == 0.0 && x.is_finite() { format!("{:.1}", x) } else { format!("{}", x) }
}

// ---------------------------------------------------------------------------
// Native string track (StrGen). A Nova string value is represented at run time
// as an i64 handle to a heap NStr (runtime/nova_native_rt.c). This is increment 1
// of dynamic native strings: functions that compose a string from literals,
// string parameters, `+` concatenation, f-strings (string holes), and calls to
// other string functions. Numbers-in-strings, string locals/loops, and string
// builtins are follow-up increments; anything unsupported declines and the
// oracle gate (byte-diff vs `nova run`) guarantees output is never wrong.
// ---------------------------------------------------------------------------

// Functions whose result is such a run-time string. A function qualifies iff it
// is not on a numeric track, its body is a single tail expression / return of a
// string composition, and that composition contains at least one literal /
// concat / f-string (so an identity `fn f(x){x}` is never mistyped). Closed under
// calls by a fixpoint, exactly like `eligible_set`.
fn str_eligible_set(prog: &Program, int_set: &HashSet<String>,
                    float_set: &HashSet<String>, num_set: &HashSet<String>) -> HashSet<String> {
    let mut funcs: HashMap<&str, &Func> = HashMap::new();
    for item in &prog.items {
        if let Item::Func(f) = item { funcs.insert(&f.name, f); }
    }
    fn tail(f: &Func) -> Option<&Expr> {
        if f.body.len() != 1 { return None; }
        match &f.body[0] { Stmt::Expr(e) | Stmt::Return(Some(e)) => Some(e), _ => None }
    }
    let mut set: HashSet<String> = funcs.values().filter(|f| {
        if f.params.len() > MAX_ARITY { return false; }
        if int_set.contains(&f.name) || float_set.contains(&f.name)
            || num_set.contains(&f.name) { return false; }
        let params: HashSet<&str> = f.params.iter().map(|s| s.as_str()).collect();
        tail(f).map_or(false, |e| str_shape(e, &params))
    }).map(|f| f.name.clone()).collect();
    // drop any function that calls a name outside the set, until stable
    loop {
        let mut remove = None;
        for name in &set {
            let e = match tail(funcs[name.as_str()]) { Some(e) => e, None => continue };
            let mut calls = Vec::new();
            str_calls(e, &mut calls);
            if calls.iter().any(|c| !set.contains(c)) { remove = Some(name.clone()); break; }
        }
        match remove { Some(n) => { set.remove(&n); } None => break }
    }
    set
}

// A string composition (calls allowed to anything — the fixpoint validates them),
// requiring at least one string-committing node so pass-through/identity
// functions are excluded.
fn str_shape(e: &Expr, params: &HashSet<&str>) -> bool {
    fn walk(e: &Expr, params: &HashSet<&str>, committed: &mut bool) -> bool {
        match strip_at(e) {
            Expr::Str(_) => { *committed = true; true }
            // an int/float/bool used in a string position is printed as its string
            // form (interpreter coercion / `str()`); a numeric parameter is passed
            // to the producer already boxed to that form, so both are strings here.
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => { *committed = true; true }
            Expr::Ident(p) => params.contains(p.as_str()),
            Expr::Binary { op: BinOp::Add, lhs, rhs } => {
                *committed = true;
                walk(lhs, params, committed) && walk(rhs, params, committed)
            }
            Expr::FmtStr(parts) => {
                *committed = true;
                parts.iter().all(|p| match p {
                    FmtPart::Lit(_) => true,
                    FmtPart::Expr(h) => walk(h, params, committed),
                })
            }
            // `str(x)` / `to_str(x)` is transparent — x's string form.
            Expr::Call { callee, args } if (callee == "str" || callee == "to_str") && args.len() == 1 => {
                *committed = true;
                walk(&args[0], params, committed)
            }
            // a call to a string-eligible function (validated by the fixpoint)
            // produces a string, so it commits — this lets a pure pass-through like
            // `fn deco(s){ wrap(wrap(s)) }` qualify, while an identity `fn f(x){x}`
            // (no call, no literal) stays excluded.
            Expr::Call { args, .. } => {
                *committed = true;
                args.iter().all(|a| walk(a, params, committed))
            }
            _ => false,
        }
    }
    let mut committed = false;
    walk(e, params, &mut committed) && committed
}

fn str_calls(e: &Expr, out: &mut Vec<String>) {
    match strip_at(e) {
        Expr::Binary { lhs, rhs, .. } => { str_calls(lhs, out); str_calls(rhs, out); }
        Expr::FmtStr(parts) =>
            for p in parts { if let FmtPart::Expr(h) = p { str_calls(h, out); } },
        Expr::Call { callee, args } => {
            // string builtins are not user functions the fixpoint validates.
            if !is_str_builtin(callee) { out.push(callee.clone()); }
            for a in args { str_calls(a, out); }
        }
        _ => {}
    }
}

// String-returning builtins the native string track lowers directly (rather than
// as a user-function call): value formatting (`str`/`to_str`) and ASCII case
// folding (`upper`/`lower`).
fn is_str_builtin(name: &str) -> bool {
    matches!(name, "str" | "to_str" | "upper" | "lower")
}

// Lowers a string-composition function body to an i64 string handle. Simpler than
// FnGen: one value kind (handle), no deopt (concat can't fail). C-ABI
// `(i64 handles...) -> i64`.
// The runtime string helpers StrGen calls (all imports from nova_native_rt.c).
#[derive(Clone, Copy)]
struct StrRt { lit: FuncId, concat: FuncId, upper: FuncId, lower: FuncId }

struct StrGen<'a, 'b> {
    b: &'a mut FunctionBuilder<'b>,
    module: &'a mut dyn Module,
    ids: &'a HashMap<String, (FuncId, usize)>, // other string functions
    rt: StrRt,
    params: HashMap<String, Value>,
}

impl<'a, 'b> StrGen<'a, 'b> {
    fn new(b: &'a mut FunctionBuilder<'b>, module: &'a mut dyn Module,
           ids: &'a HashMap<String, (FuncId, usize)>,
           rt: StrRt, params: &[String]) -> Self {
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let pv: Vec<Value> = b.block_params(entry).to_vec();
        let mut pm = HashMap::new();
        for (i, p) in params.iter().enumerate() { pm.insert(p.clone(), pv[i]); }
        StrGen { b, module, ids, rt, params: pm }
    }

    // call a runtime string helper (name -> FuncId) on already-lowered args.
    fn rt_call(&mut self, id: FuncId, args: &[Value]) -> Value {
        let f = self.module.declare_func_in_func(id, self.b.func);
        let call = self.b.ins().call(f, args);
        self.b.inst_results(call)[0]
    }

    // box literal bytes into an NStr handle via nova_str_lit(ptr, len).
    fn lit(&mut self, bytes: &[u8]) -> Option<Value> {
        let data = self.module.declare_anonymous_data(false, false).ok()?;
        let mut desc = DataDescription::new();
        desc.define(bytes.to_vec().into_boxed_slice());
        self.module.define_data(data, &desc).ok()?;
        let gv = self.module.declare_data_in_func(data, self.b.func);
        let ptr = self.b.ins().global_value(I64, gv);
        let n = self.b.ins().iconst(I64, bytes.len() as i64);
        Some(self.rt_call(self.rt.lit, &[ptr, n]))
    }

    fn concat(&mut self, a: Value, c: Value) -> Value {
        self.rt_call(self.rt.concat, &[a, c])
    }

    fn lower_str(&mut self, e: &Expr) -> Option<Value> {
        match strip_at(e) {
            Expr::Str(s) => self.lit(s.as_bytes()),
            // an int/float/bool literal in string position -> its string form,
            // matching the interpreter's Display (decimal ints, float_str, true/false).
            Expr::Int(n) => self.lit(n.to_string().as_bytes()),
            Expr::Float(x) => self.lit(float_str(*x).as_bytes()),
            Expr::Bool(b) => self.lit(if *b { b"true" } else { b"false" }),
            Expr::Ident(p) => self.params.get(p).copied(),
            // `str(x)` / `to_str(x)` -> x's string form (numeric params already
            // arrive boxed to their string form, so this is the identity).
            Expr::Call { callee, args } if (callee == "str" || callee == "to_str") && args.len() == 1 =>
                self.lower_str(&args[0]),
            // ASCII case folding via the runtime.
            Expr::Call { callee, args } if (callee == "upper" || callee == "lower") && args.len() == 1 => {
                let s = self.lower_str(&args[0])?;
                let id = if callee == "upper" { self.rt.upper } else { self.rt.lower };
                Some(self.rt_call(id, &[s]))
            }
            Expr::Binary { op: BinOp::Add, lhs, rhs } => {
                let a = self.lower_str(lhs)?;
                let c = self.lower_str(rhs)?;
                Some(self.concat(a, c))
            }
            Expr::FmtStr(parts) => {
                let mut acc: Option<Value> = None;
                for p in parts {
                    let piece = match p {
                        FmtPart::Lit(s) => self.lit(s.as_bytes())?,
                        FmtPart::Expr(h) => self.lower_str(h)?,
                    };
                    acc = Some(match acc { Some(a) => self.concat(a, piece), None => piece });
                }
                match acc { Some(v) => Some(v), None => self.lit(b"") }
            }
            Expr::Call { callee, args } => {
                let (id, arity) = self.ids.get(callee).copied()?;
                if args.len() != arity { return None; }
                let mut av = Vec::new();
                for a in args { av.push(self.lower_str(a)?); }
                let f = self.module.declare_func_in_func(id, self.b.func);
                let call = self.b.ins().call(f, &av);
                Some(self.b.inst_results(call)[0])
            }
            _ => None,
        }
    }

    fn lower(&mut self, body: &[Stmt]) -> Option<()> {
        let e = match body.first()? {
            Stmt::Expr(e) | Stmt::Return(Some(e)) => e,
            _ => return None,
        };
        let v = self.lower_str(e)?;
        self.b.ins().return_(&[v]);
        Some(())
    }
}

// Which architecture the native object is emitted for. `Host` uses the running
// machine's ISA; the cross targets emit a relocatable object for another arch
// (linked + qemu-verified by the build). The same Cranelift IR lowers to every
// target — the entry's itoa/`write` is arch-independent — so nothing here is
// arch-specific; only the ISA and the object's binary format differ.
#[derive(Clone, Copy, PartialEq)]
pub enum NativeTarget { Host, Aarch64, Riscv64 }

impl NativeTarget {
    // the target triple string, or None for the host (uses cranelift_native).
    fn triple(self) -> Option<&'static str> {
        match self {
            NativeTarget::Host => None,
            NativeTarget::Aarch64 => Some("aarch64-unknown-linux-gnu"),
            NativeTarget::Riscv64 => Some("riscv64gc-unknown-linux-gnu"),
        }
    }
}

// Build a relocatable object for a `print(<expr>)` program, covering the full
// surface the JIT compiles: the integer track (`FnGen`, incl. local integer
// arrays/structs), the float track (`FloatGen`), the numeric-mixed track
// (`NumGen`, result int or f64-bits), plus constant-string values (literals,
// concatenation, f-strings with constant holes, `str()` of a constant) which are
// folded to bytes and printed with a `write` syscall — no runtime needed for a
// pure-string program. Returns `(object bytes, needs_runtime)` —
// `needs_runtime` is true iff the object may reference `runtime/nova_native_rt.c`
// (arena helpers / fmod-fpow / float printer), so the caller knows whether to
// link it. `None` when the program isn't numeric-native (→ C/embed fallback).
// Correctness is backed by the build's oracle gate (byte-diff vs `nova run`).
pub fn compile_object(prog: &Program, target: NativeTarget) -> Option<(Vec<u8>, bool)> {
    // 1) locate `main` and extract its single `print(<expr>)` argument
    let main = prog.items.iter().find_map(|it| match it {
        Item::Func(f) if f.name == "main" => Some(f),
        _ => None,
    })?;
    let printed = main_print_args(main)?; // one expr per print(...) statement

    // 2) pick how each printed value is produced. If it is a direct call
    //    `f(const-args)` to a user function (fib/mandel/sieve all do), call `f`
    //    DIRECTLY with those constants — this covers the numeric track, whose
    //    classifier rejects cross-function calls so a `return f(..)` wrapper can't
    //    be typed. Otherwise (a literal / arithmetic expr) synthesize
    //    `fn __nova_main_val_N() { return <expr> }` and let the analysis type it.
    let user_fn = |name: &str| prog.items.iter().any(|it|
        matches!(it, Item::Func(f) if f.name == name));
    // String-eligible functions (computed on `prog`; the synthesized numeric
    // producers below don't affect real functions' membership). A printed
    // `f(<const strings>)` where `f` is string-eligible compiles natively via the
    // StrGen track instead of falling back.
    let str_set = {
        let i = eligible_set(prog);
        let fl = float_eligible_set(prog, &i);
        let (nu, _) = numeric_eligible_set(prog, &i, &fl);
        str_eligible_set(prog, &i, &fl, &nu)
    };

    // Each printed value is a compile-time-constant string (baked bytes, libc-only),
    // a run-time string from a StrGen function, or a numeric value from a compiled
    // function. `slots` records the print order; `producers`/`items` carry the
    // numeric ones, `str_producers` the string-function ones.
    enum Slot { Str(String), Num(usize), StrFn(usize) }
    let mut slots: Vec<Slot> = Vec::new();
    let mut producers: Vec<(String, Vec<ArgConst>)> = Vec::new();
    let mut str_producers: Vec<(String, Vec<String>)> = Vec::new(); // (fn, const-string args)
    let mut aug = prog.clone();
    for (i, expr) in printed.iter().enumerate() {
        // a constant string prints via baked bytes — no producer function needed.
        if let Some(s) = const_str(expr) { slots.push(Slot::Str(s)); continue; }
        // a string-eligible function called with constant args -> StrGen. Each arg
        // is a constant string, or a constant int/bool boxed to its string form
        // (byte-identical to how the function would print it), so a numeric
        // parameter arrives already stringified.
        if let Expr::Call { callee, args } = strip_at(expr) {
            if str_set.contains(callee) {
                let sargs: Option<Vec<String>> = args.iter()
                    .map(|a| const_str(a).or_else(|| const_hole(a)))
                    .collect();
                if let Some(sargs) = sargs {
                    let idx = str_producers.len();
                    str_producers.push((callee.clone(), sargs));
                    slots.push(Slot::StrFn(idx));
                    continue;
                }
            }
        }
        let direct = match unwrap_at(expr) {
            Expr::Call { callee, args } if user_fn(callee) =>
                args.iter().map(const_arg).collect::<Option<Vec<_>>>()
                    .map(|c| (callee.clone(), c)),
            _ => None,
        };
        let idx = producers.len();
        match direct {
            Some(nv) => producers.push(nv),
            None => {
                let name = format!("__nova_main_val_{i}");
                aug.items.push(Item::Func(Func {
                    name: name.clone(),
                    params: Vec::new(), param_types: Vec::new(), param_modes: Vec::new(),
                    ret_type: None, type_params: Vec::new(), where_bounds: Vec::new(),
                    effects: None, body: vec![Stmt::Return(Some(expr.clone()))],
                    is_async: false, attrs: Vec::new(),
                }));
                producers.push((name, Vec::new()));
            }
        }
        slots.push(Slot::Num(idx));
    }

    // 3) compute the three JIT tracks and resolve each printed value's track. If
    //    any lands on none, the program isn't numeric-native -> C/embed fallback.
    let eligible = eligible_set(&aug);
    let feligible = float_eligible_set(&aug, &eligible);
    let (numeric, nret) = numeric_eligible_set(&aug, &eligible, &feligible);
    // per printed value: (producer name, track, constant args)
    let mut items: Vec<(String, Track, Vec<ArgConst>)> = Vec::new();
    for (name, args) in &producers {
        let pn = name.as_str();
        let track = if eligible.contains(pn) { Track::Int }
            else if feligible.contains(pn) { Track::Float }
            else if numeric.contains(pn) {
                if nret.get(pn) == Some(&FKind::F) { Track::NumFloat } else { Track::NumInt }
            } else { return None };
        // the direct-call ABI carries constants by value: integer/numeric callees
        // take i64 params (all-Int), the float callee takes f64 params. Reject a
        // kind mismatch (e.g. an Int literal to an f64 param) -> fallback.
        let args_ok = args.iter().all(|a| match (track, a) {
            (Track::Float, ArgConst::F(_)) => true,
            (Track::Float, ArgConst::I(_)) => false,
            (_, ArgConst::I(_)) => true,
            (_, ArgConst::F(_)) => false,
        });
        if !args_ok { return None; }
        items.push((name.clone(), track, args.clone()));
    }

    let mut funcs: HashMap<&str, &Func> = HashMap::new();
    let mut sdefs: HashMap<String, Vec<String>> = HashMap::new();
    for item in &aug.items {
        match item {
            Item::Func(f) => { funcs.insert(&f.name, f); }
            Item::Struct(sd) => { sdefs.insert(sd.name.clone(), sd.fields.clone()); }
            _ => {}
        }
    }
    // link the runtime object when the emitted code may reference it: any local
    // array/struct (arena helpers), or any float/numeric function (fmod/fpow and
    // the float printer). Pure-integer array-free programs (fib) link only libc.
    let uses_arena = eligible.iter().any(|n| {
        let f = funcs[n.as_str()];
        !array_vars(f).is_empty() || !struct_vars(f, &sdefs).is_empty()
    });
    // Link the runtime object when a numeric value is printed through the runtime
    // printers: any multi-print `main` that prints at least one number, or any
    // float/numeric function. A single printed number keeps its in-IR itoa, and a
    // program that only prints constant strings writes baked bytes — both link
    // only libc.
    let multi = slots.len() > 1;
    let needs_runtime = (multi && !items.is_empty()) || uses_arena
        || !feligible.is_empty() || !numeric.is_empty()
        || !str_producers.is_empty() // string track uses nova_str_* from the runtime
        || items.iter().any(|(_, t, _)| matches!(t, Track::Float | Track::NumFloat));

    // 4) build the object module for the requested ISA (host, or a cross target)
    let mut flags = settings::builder();
    flags.set("opt_level", "speed").ok()?;
    // the host binary is a PIE (linked with plain `cc`); the cross binaries link
    // static non-PIE, so they emit non-PIC code.
    flags.set("is_pic", if target == NativeTarget::Host { "true" } else { "false" }).ok()?;
    let isa = match target.triple() {
        None => cranelift_native::builder().ok()?
            .finish(settings::Flags::new(flags)).ok()?,
        Some(triple) => {
            let t = triple.parse::<target_lexicon::Triple>().ok()?;
            let mut b = isa::lookup(t).ok()?;
            // riscv64gc: enable the G ISA extensions (IMAFD + Zicsr/Zifencei) and
            // the C compressed set — Cranelift's riscv64 backend has these OFF by
            // default, so integer mul/div (the itoa's sdiv/srem) and float ops
            // wouldn't lower without them.
            if target == NativeTarget::Riscv64 {
                for ext in ["has_m", "has_a", "has_f", "has_d", "has_c",
                            "has_zicsr", "has_zifencei"] {
                    b.enable(ext).ok()?;
                }
            }
            b.finish(settings::Flags::new(flags)).ok()?
        }
    };
    let builder = ObjectBuilder::new(
        isa, "nova", cranelift_module::default_libcall_names()).ok()?;
    let mut module = ObjectModule::new(builder);

    // runtime helper imports (raw i64/f64 ABI matching src/jit.rs's Rust helpers,
    // provided by runtime/nova_native_rt.c). Unreferenced ones emit no relocation.
    let mut helpers: HashMap<&'static str, FuncId> = HashMap::new();
    {
        let f64t = types::F64;
        let sigs: [(&'static str, &[Type], &[Type]); 9] = [
            ("nova_arr_new", &[], &[I64]),
            ("nova_arr_fill", &[I64, I64], &[I64]),
            ("nova_arr_push", &[I64, I64], &[]),
            ("nova_arr_len", &[I64], &[I64]),
            ("nova_arr_get", &[I64, I64, I64], &[I64]),
            ("nova_arr_set", &[I64, I64, I64, I64], &[]),
            ("nova_arr_pop", &[I64, I64], &[I64]),
            ("nova_fmod", &[f64t, f64t], &[f64t]),
            ("nova_fpow", &[f64t, f64t], &[f64t]),
        ];
        for (name, params, rets) in sigs {
            let mut sig = module.make_signature();
            for p in params { sig.params.push(AbiParam::new(*p)); }
            for r in rets { sig.returns.push(AbiParam::new(*r)); }
            let id = module.declare_function(name, Linkage::Import, &sig).ok()?;
            helpers.insert(name, id);
        }
    }
    let libm: HashMap<&'static str, FuncId> = [
        ("fmod", helpers["nova_fmod"]), ("fpow", helpers["nova_fpow"]),
    ].into_iter().collect();

    let mut fctx = FunctionBuilderContext::new();
    // IR-generation is serial (it declares call refs into the shared module), but
    // each function's Context is collected and compiled in parallel below.
    let mut jobs: Vec<(FuncId, Context)> = Vec::new();

    // 5a) integer track (FnGen): (deopt_ptr, i64 args...) -> i64, incl. arrays
    let mut names: Vec<&str> = eligible.iter().map(|s| s.as_str()).collect();
    names.sort();
    let mut ids: HashMap<String, (FuncId, usize)> = HashMap::new();
    for name in &names {
        let f = funcs[*name];
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(I64)); // deopt flag pointer
        for _ in 0..f.params.len() { sig.params.push(AbiParam::new(I64)); }
        sig.returns.push(AbiParam::new(I64));
        let id = module.declare_function(name, Linkage::Local, &sig).ok()?;
        ids.insert(name.to_string(), (id, f.params.len()));
    }
    for name in &names {
        let f = funcs[*name];
        let (id, _) = ids[*name];
        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(I64));
        for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(I64)); }
        ctx.func.signature.returns.push(AbiParam::new(I64));
        {
            let arrays = array_vars(f);
            let structs = struct_vars(f, &sdefs);
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let mut g = FnGen::new(&mut b, &mut module, &ids, &helpers, arrays, structs, &f.params,
                                   target == NativeTarget::Riscv64);
            g.lower(&f.body)?;
            b.finalize();
        }
        jobs.push((id, ctx));
    }

    // 5b) float track (FloatGen): (f64 args...) -> f64, no deopt
    let mut fnames: Vec<&str> = feligible.iter().map(|s| s.as_str()).collect();
    fnames.sort();
    let mut fids: HashMap<String, (FuncId, usize)> = HashMap::new();
    for name in &fnames {
        let f = funcs[*name];
        let mut sig = module.make_signature();
        for _ in 0..f.params.len() { sig.params.push(AbiParam::new(types::F64)); }
        sig.returns.push(AbiParam::new(types::F64));
        let id = module.declare_function(name, Linkage::Local, &sig).ok()?;
        fids.insert(name.to_string(), (id, f.params.len()));
    }
    for name in &fnames {
        let f = funcs[*name];
        let (id, _) = fids[*name];
        let mut ctx = module.make_context();
        for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(types::F64)); }
        ctx.func.signature.returns.push(AbiParam::new(types::F64));
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let mut g = FloatGen::new(&mut b, &mut module, &fids, &libm, &f.params);
            g.lower(&f.body)?;
            b.finalize();
        }
        jobs.push((id, ctx));
    }

    // 5c) numeric-mixed track (NumGen): same i64 ABI as the integer track
    let mut nnames: Vec<&str> = numeric.iter().map(|s| s.as_str()).collect();
    nnames.sort();
    let mut nids: HashMap<String, (FuncId, usize)> = HashMap::new();
    for name in &nnames {
        let f = funcs[*name];
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(I64)); // deopt flag pointer
        for _ in 0..f.params.len() { sig.params.push(AbiParam::new(I64)); }
        sig.returns.push(AbiParam::new(I64));
        let id = module.declare_function(name, Linkage::Local, &sig).ok()?;
        nids.insert(name.to_string(), (id, f.params.len()));
    }
    for name in &nnames {
        let f = funcs[*name];
        let (id, _) = nids[*name];
        let rk = nret[*name];
        let mut ctx = module.make_context();
        ctx.func.signature.params.push(AbiParam::new(I64));
        for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(I64)); }
        ctx.func.signature.returns.push(AbiParam::new(I64));
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
            let mut g = NumGen::new(&mut b, &mut module, &libm, &f.params, rk);
            g.lower(&f.body)?;
            b.finalize();
        }
        jobs.push((id, ctx));
    }

    // 5d) string track (StrGen): (i64 handles...) -> i64 handle, no deopt. Only
    //     built when a string value is actually printed; declares the runtime
    //     string imports and every string-eligible function (closed under calls).
    let (str_lit_id, str_print_id, str_ids): (Option<FuncId>, Option<FuncId>, HashMap<String, (FuncId, usize)>) =
    if str_producers.is_empty() {
        (None, None, HashMap::new())
    } else {
        let lit = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(I64)); // bytes ptr
            sig.params.push(AbiParam::new(I64)); // len
            sig.returns.push(AbiParam::new(I64));
            module.declare_function("nova_str_lit", Linkage::Import, &sig).ok()?
        };
        let cat = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(I64));
            sig.params.push(AbiParam::new(I64));
            sig.returns.push(AbiParam::new(I64));
            module.declare_function("nova_str_concat", Linkage::Import, &sig).ok()?
        };
        let prn = {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(I64));
            module.declare_function("nova_str_print", Linkage::Import, &sig).ok()?
        };
        // (i64 handle) -> i64 handle unary runtime string helpers.
        let mut unary = |name: &str| -> Option<FuncId> {
            let mut sig = module.make_signature();
            sig.params.push(AbiParam::new(I64));
            sig.returns.push(AbiParam::new(I64));
            module.declare_function(name, Linkage::Import, &sig).ok()
        };
        let upp = unary("nova_str_upper")?;
        let low = unary("nova_str_lower")?;
        let rt = StrRt { lit, concat: cat, upper: upp, lower: low };
        let mut snames: Vec<&str> = str_set.iter().map(|s| s.as_str()).collect();
        snames.sort();
        let mut str_ids: HashMap<String, (FuncId, usize)> = HashMap::new();
        for name in &snames {
            let f = funcs[*name];
            let mut sig = module.make_signature();
            for _ in 0..f.params.len() { sig.params.push(AbiParam::new(I64)); }
            sig.returns.push(AbiParam::new(I64));
            let id = module.declare_function(name, Linkage::Local, &sig).ok()?;
            str_ids.insert(name.to_string(), (id, f.params.len()));
        }
        for name in &snames {
            let f = funcs[*name];
            let (id, _) = str_ids[*name];
            let mut ctx = module.make_context();
            for _ in 0..f.params.len() { ctx.func.signature.params.push(AbiParam::new(I64)); }
            ctx.func.signature.returns.push(AbiParam::new(I64));
            {
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
                let mut g = StrGen::new(&mut b, &mut module, &str_ids, rt, &f.params);
                g.lower(&f.body)?;
                b.finalize();
            }
            jobs.push((id, ctx));
        }
        (Some(lit), Some(prn), str_ids)
    };

    // 6) the C-ABI entry point: `int main(void)` that calls __nova_main_val and
    //    prints its result (+'\n'), returning 0 — or 1 on deopt (no output), so
    //    the oracle gate sees a divergence and the caller falls back.
    let write_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(I32)); // fd
        sig.params.push(AbiParam::new(I64)); // buf ptr
        sig.params.push(AbiParam::new(I64)); // count
        sig.returns.push(AbiParam::new(I64));
        module.declare_function("write", Linkage::Import, &sig).ok()?
    };
    let print_f64_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F64));
        module.declare_function("nova_print_f64", Linkage::Import, &sig).ok()?
    };
    let print_i64_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(I64));
        module.declare_function("nova_print_i64", Linkage::Import, &sig).ok()?
    };
    // resolve each printed value in source order: a constant string becomes baked
    // read-only bytes (a data object + its length), a numeric value resolves its
    // producer FuncId from the right track's map.
    let mut resolved: Vec<RItem> = Vec::new();
    for slot in &slots {
        match slot {
            Slot::Str(s) => {
                // `print` appends a newline; bake the string + '\n' as one blob.
                let bytes = format!("{}\n", s).into_bytes();
                let len = bytes.len();
                let data = module.declare_anonymous_data(false, false).ok()?;
                let mut desc = DataDescription::new();
                desc.define(bytes.into_boxed_slice());
                module.define_data(data, &desc).ok()?;
                resolved.push(RItem::Str(data, len));
            }
            Slot::Num(idx) => {
                let (name, track, args) = &items[*idx];
                let id = match track {
                    Track::Int => ids[name].0,
                    Track::Float => fids[name].0,
                    Track::NumInt | Track::NumFloat => nids[name].0,
                };
                resolved.push(RItem::Num(id, *track, args.clone()));
            }
            Slot::StrFn(idx) => {
                let (name, sargs) = &str_producers[*idx];
                let id = str_ids[name].0;
                // bake each constant-string argument as a read-only data object.
                let mut arg_data = Vec::new();
                for s in sargs {
                    let bytes = s.clone().into_bytes();
                    let len = bytes.len();
                    let data = module.declare_anonymous_data(false, false).ok()?;
                    let mut desc = DataDescription::new();
                    desc.define(bytes.into_boxed_slice());
                    module.define_data(data, &desc).ok()?;
                    arg_data.push((data, len));
                }
                resolved.push(RItem::StrProducer(id, arg_data));
            }
        }
    }
    let main_id = {
        let mut sig = module.make_signature();
        sig.returns.push(AbiParam::new(I32));
        module.declare_function("main", Linkage::Export, &sig).ok()?
    };
    let mut ctx = module.make_context();
    ctx.func.signature.returns.push(AbiParam::new(I32));
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        if resolved.len() == 1 {
            match &resolved[0] {
                // single constant string: emit bytes + write, libc-only.
                RItem::Str(data, len) =>
                    build_native_entry_str(&mut b, &mut module, *data, *len, write_id),
                // single number: keep the in-IR itoa path (pure-int stays libc-only).
                RItem::Num(id, track, args) =>
                    build_native_entry(&mut b, &mut module, *id, *track, args, write_id, print_f64_id),
                // single run-time string: box the args, call the producer, print.
                RItem::StrProducer(id, args) =>
                    build_native_entry_str_call(&mut b, &mut module, *id, args,
                                                str_lit_id.unwrap(), str_print_id.unwrap()),
            }
        } else {
            build_native_entry_multi(&mut b, &mut module, &resolved, write_id,
                                     print_i64_id, print_f64_id, str_lit_id, str_print_id);
        }
        b.finalize();
    }
    jobs.push((main_id, ctx));

    // 7) compile all collected functions across the host's cores, then emit the
    //    relocatable object bytes.
    define_parallel(&mut module, jobs)?;
    Some((module.finish().emit().ok()?, needs_runtime))
}

// Emit the body of the C-ABI `main` for a single constant-string print: write the
// baked bytes (string + '\n', already in a read-only data object) to fd 1 and
// return 0. No producer call, no deopt, no runtime — just libc's `write`.
fn build_native_entry_str(b: &mut FunctionBuilder, module: &mut ObjectModule,
                          data: DataId, len: usize, write_id: FuncId) {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.seal_block(entry);
    let gv = module.declare_data_in_func(data, b.func);
    let ptr = b.ins().global_value(I64, gv);
    let n = b.ins().iconst(I64, len as i64);
    let fd1 = b.ins().iconst(I32, 1);
    let wref = module.declare_func_in_func(write_id, b.func);
    b.ins().call(wref, &[fd1, ptr, n]);
    let ok = b.ins().iconst(I32, 0);
    b.ins().return_(&[ok]);
}

// Box each constant-string argument (its bytes already in a data object) via
// nova_str_lit and call the StrGen producer, returning its result handle.
fn call_str_producer(b: &mut FunctionBuilder, module: &mut ObjectModule,
                     producer: FuncId, args: &[(DataId, usize)], str_lit: FuncId) -> Value {
    let mut handles = Vec::new();
    for (data, len) in args {
        let gv = module.declare_data_in_func(*data, b.func);
        let ptr = b.ins().global_value(I64, gv);
        let n = b.ins().iconst(I64, *len as i64);
        let lref = module.declare_func_in_func(str_lit, b.func);
        let call = b.ins().call(lref, &[ptr, n]);
        handles.push(b.inst_results(call)[0]);
    }
    let pref = module.declare_func_in_func(producer, b.func);
    let call = b.ins().call(pref, &handles);
    b.inst_results(call)[0]
}

// Emit the body of the C-ABI `main` for a single run-time string print: box the
// constant-string args, call the StrGen producer, print the handle, return 0.
fn build_native_entry_str_call(b: &mut FunctionBuilder, module: &mut ObjectModule,
                               producer: FuncId, args: &[(DataId, usize)],
                               str_lit: FuncId, str_print: FuncId) {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.seal_block(entry);
    let h = call_str_producer(b, module, producer, args, str_lit);
    let pref = module.declare_func_in_func(str_print, b.func);
    b.ins().call(pref, &[h]);
    let ok = b.ins().iconst(I32, 0);
    b.ins().return_(&[ok]);
}

// Emit the body of the C-ABI `main`: call the nullary value producer, then print
// its result. Integer/numeric-int -> in-IR decimal itoa + write(1,…); float /
// numeric-float -> nova_print_f64 (runtime, byte-identical to the interpreter).
fn build_native_entry(b: &mut FunctionBuilder, module: &mut ObjectModule,
                      producer: FuncId, track: Track, args: &[ArgConst],
                      write_id: FuncId, print_f64_id: FuncId) {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.seal_block(entry);

    let float_out = matches!(track, Track::Float | Track::NumFloat);
    let has_deopt = !matches!(track, Track::Float); // float track never deopts

    let pref = module.declare_func_in_func(producer, b.func);
    let print_blk = b.create_block();
    let diverge_blk = b.create_block();

    // materialise the constant arguments in the callee's ABI: i64 for the
    // integer/numeric tracks, f64 for the float track.
    let mut argvals: Vec<Value> = Vec::new();
    for a in args {
        argvals.push(match a {
            ArgConst::I(n) => b.ins().iconst(I64, *n),
            ArgConst::F(x) => b.ins().f64const(*x),
        });
    }

    // call the producer with its track's ABI; branch to the print block (float
    // track can't deopt, so it jumps straight there).
    let r; // the producer's raw result (i64 for int/numeric, f64 for float)
    if has_deopt {
        let slot = b.func.create_sized_stack_slot(
            StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
        let zero64 = b.ins().iconst(I64, 0);
        b.ins().stack_store(zero64, slot, 0);
        let dptr = b.ins().stack_addr(I64, slot, 0);
        let mut call_args = vec![dptr]; // deopt ptr first
        call_args.extend_from_slice(&argvals);
        let call = b.ins().call(pref, &call_args);
        r = b.inst_results(call)[0];
        let flag = b.ins().stack_load(I64, slot, 0);
        b.ins().brif(flag, diverge_blk, &[], print_blk, &[]);
    } else {
        let call = b.ins().call(pref, &argvals);
        r = b.inst_results(call)[0];
        b.ins().jump(print_blk, &[]);
    }

    b.switch_to_block(print_blk);
    b.seal_block(print_blk);

    // ---- float path: hand the f64 to the runtime formatter ----
    if float_out {
        // numeric-float returns f64 *bits* in an i64; reinterpret. float track
        // already returns an f64.
        let fv = if matches!(track, Track::NumFloat) {
            b.ins().bitcast(types::F64, MemFlags::new(), r)
        } else { r };
        let fref = module.declare_func_in_func(print_f64_id, b.func);
        b.ins().call(fref, &[fv]);
        let ok = b.ins().iconst(I32, 0);
        b.ins().return_(&[ok]);
        b.switch_to_block(diverge_blk);
        b.seal_block(diverge_blk);
        let one = b.ins().iconst(I32, 1);
        b.ins().return_(&[one]);
        return;
    }

    // ---- integer path: decimal itoa into a 32-byte stack buffer, backwards ----
    let buf = b.func.create_sized_stack_slot(
        StackSlotData::new(StackSlotKind::ExplicitSlot, 32, 3));
    let base = b.ins().stack_addr(I64, buf, 0);
    // buf[31] = '\n'
    let nl = b.ins().iconst(I8, 10);
    let end = b.ins().iconst(I64, 31);
    let nl_addr = b.ins().iadd(base, end);
    b.ins().store(MemFlags::new(), nl, nl_addr, 0);
    // neg = r < 0 ; m = |r|
    let zero = b.ins().iconst(I64, 0);
    let neg = b.ins().icmp(IntCC::SignedLessThan, r, zero);
    let negr = b.ins().isub(zero, r);
    let m0 = b.ins().select(neg, negr, r);

    // do-while digit loop: params (i, m). i starts at 31 (the newline slot).
    let loop_blk = b.create_block();
    b.append_block_param(loop_blk, I64); // i
    b.append_block_param(loop_blk, I64); // m
    let i0 = b.ins().iconst(I64, 31);
    b.ins().jump(loop_blk, &[i0, m0]);

    b.switch_to_block(loop_blk);
    let i = b.block_params(loop_blk)[0];
    let m = b.block_params(loop_blk)[1];
    let i2 = b.ins().iadd_imm(i, -1);
    let ten = b.ins().iconst(I64, 10);
    let d = b.ins().srem(m, ten);
    let d8 = b.ins().ireduce(I8, d);
    let ch = b.ins().iadd_imm(d8, 48); // '0' + digit
    let d_addr = b.ins().iadd(base, i2);
    b.ins().store(MemFlags::new(), ch, d_addr, 0);
    let m2 = b.ins().sdiv(m, ten);
    let after_blk = b.create_block();
    b.append_block_param(after_blk, I64); // final index (before optional '-')
    b.ins().brif(m2, loop_blk, &[i2, m2], after_blk, &[i2]);
    b.seal_block(loop_blk);

    // ---- after digits: prepend '-' if negative ----
    b.switch_to_block(after_blk);
    b.seal_block(after_blk);
    let di = b.block_params(after_blk)[0];
    let finish_blk = b.create_block();
    b.append_block_param(finish_blk, I64); // start index
    let neg_blk = b.create_block();
    b.ins().brif(neg, neg_blk, &[], finish_blk, &[di]);

    b.switch_to_block(neg_blk);
    b.seal_block(neg_blk);
    let di1 = b.ins().iadd_imm(di, -1);
    let minus = b.ins().iconst(I8, 45); // '-'
    let minus_addr = b.ins().iadd(base, di1);
    b.ins().store(MemFlags::new(), minus, minus_addr, 0);
    b.ins().jump(finish_blk, &[di1]);

    // ---- finish: write(1, base+start, 32-start), return 0 ----
    b.switch_to_block(finish_blk);
    b.seal_block(finish_blk);
    let start = b.block_params(finish_blk)[0];
    let ptr = b.ins().iadd(base, start);
    let total = b.ins().iconst(I64, 32);
    let len = b.ins().isub(total, start);
    let fd1 = b.ins().iconst(I32, 1);
    let wref = module.declare_func_in_func(write_id, b.func);
    b.ins().call(wref, &[fd1, ptr, len]);
    let ok = b.ins().iconst(I32, 0);
    b.ins().return_(&[ok]);

    // ---- diverge: return 1 (no output) ----
    b.switch_to_block(diverge_blk);
    b.seal_block(diverge_blk);
    let one = b.ins().iconst(I32, 1);
    b.ins().return_(&[one]);
}

// Emit the entry for a multi-`print` main: print each value in source order — a
// constant string via a `write` of its baked bytes, a run-time string via a
// StrGen producer + nova_str_print, a number via the runtime printers
// (nova_print_i64 / nova_print_f64) — then return 0. Any numeric producer that
// deopts jumps to a shared diverge path (return 1, no further output) so the
// oracle gate falls back — never wrong.
fn build_native_entry_multi(b: &mut FunctionBuilder, module: &mut ObjectModule,
                            items: &[RItem], write_id: FuncId,
                            print_i64_id: FuncId, print_f64_id: FuncId,
                            str_lit_id: Option<FuncId>, str_print_id: Option<FuncId>) {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.seal_block(entry);
    // one reusable deopt-flag slot (reset before each fallible call)
    let slot = b.func.create_sized_stack_slot(
        StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
    let diverge_blk = b.create_block();

    for it in items {
        let (id, track, args) = match it {
            RItem::Str(data, len) => {
                // constant string: write its baked bytes (string + '\n'), no deopt.
                let gv = module.declare_data_in_func(*data, b.func);
                let ptr = b.ins().global_value(I64, gv);
                let n = b.ins().iconst(I64, *len as i64);
                let fd1 = b.ins().iconst(I32, 1);
                let wref = module.declare_func_in_func(write_id, b.func);
                b.ins().call(wref, &[fd1, ptr, n]);
                continue;
            }
            RItem::StrProducer(pid, pargs) => {
                // run-time string: box args, call the producer, print the handle.
                let h = call_str_producer(b, module, *pid, pargs, str_lit_id.unwrap());
                let pref = module.declare_func_in_func(str_print_id.unwrap(), b.func);
                b.ins().call(pref, &[h]);
                continue;
            }
            RItem::Num(id, track, args) => (*id, *track, args),
        };
        let pref = module.declare_func_in_func(id, b.func);
        let mut argvals: Vec<Value> = Vec::new();
        for a in args {
            argvals.push(match a {
                ArgConst::I(n) => b.ins().iconst(I64, *n),
                ArgConst::F(x) => b.ins().f64const(*x),
            });
        }
        let float_out = matches!(track, Track::Float | Track::NumFloat);
        let r; // producer result (i64 for int/numeric, f64 for float track)
        if matches!(track, Track::Float) {
            let call = b.ins().call(pref, &argvals);
            r = b.inst_results(call)[0];
        } else {
            let zero = b.ins().iconst(I64, 0);
            b.ins().stack_store(zero, slot, 0);
            let dptr = b.ins().stack_addr(I64, slot, 0);
            let mut ca = vec![dptr];
            ca.extend_from_slice(&argvals);
            let call = b.ins().call(pref, &ca);
            r = b.inst_results(call)[0];
            let flag = b.ins().stack_load(I64, slot, 0);
            let cont = b.create_block();
            b.ins().brif(flag, diverge_blk, &[], cont, &[]);
            b.switch_to_block(cont);
            b.seal_block(cont);
        }
        // print this value
        if float_out {
            let fv = if matches!(track, Track::NumFloat) {
                b.ins().bitcast(types::F64, MemFlags::new(), r)
            } else { r };
            let fref = module.declare_func_in_func(print_f64_id, b.func);
            b.ins().call(fref, &[fv]);
        } else {
            let iref = module.declare_func_in_func(print_i64_id, b.func);
            b.ins().call(iref, &[r]);
        }
    }
    let ok = b.ins().iconst(I32, 0);
    b.ins().return_(&[ok]);

    b.switch_to_block(diverge_blk);
    b.seal_block(diverge_blk);
    let one = b.ins().iconst(I32, 1);
    b.ins().return_(&[one]);
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
    #[test] fn struct_kernel_compiles_and_matches() {
        // a local int-field struct kernel compiles on the i64 track (arena
        // slots) and matches the VM byte-for-byte; aliases share the handle
        let names = same_tiered(
            "struct P { x: Int, y: Int }\n\
             fn walk(n){ let p = P { x: 0, y: 0 }; let i = 0;\n\
               while i < n { p.x = p.x + 2; p.y = p.y + p.x; i = i + 1 } p.x + p.y }\n\
             fn main(){ let t = 0; for k in 0..200 { t = t + walk(50) } t }", 50);
        assert!(names.contains(&"walk".to_string()), "struct kernel must compile: {:?}", names);
    }
    #[test] fn struct_escape_stays_off_jit() {
        // returning the struct (escape) keeps the function ineligible — the
        // interp/VM still run it correctly, just uncompiled
        let names = same_tiered(
            "struct P { x: Int, y: Int }\n\
             fn esc(){ let p = P { x: 1, y: 2 }; p }\n\
             fn main(){ let e = esc(); e.x + e.y }", 0);
        assert!(!names.contains(&"esc".to_string()), "escaping struct must not compile");
    }
    #[test] fn simd_hint_eagerly_compiles_without_loop() {
        // `#[simd]` is an eager-compile JIT hint (like `#[hot]`): a loop-free
        // numeric function marked #[simd] is compiled up-front by warm_loops even
        // when the count threshold would never be reached; a plain loop-free
        // function is not. (Vectorization proper is a documented future step.)
        let mut prog = parse_program(
            "#[simd] fn v(a,b){ a*b + a - b }\nfn plain(a,b){ a + b }\n\
             fn main(){ v(2,3) + plain(1,1) }").expect("parse");
        fold_program(&mut prog);
        let t = TieredJit::new(&prog, 1_000_000); // counting will never compile
        t.warm_loops();
        let names = t.compiled_functions();
        assert!(names.contains(&"v".to_string()), "#[simd] fn must be eagerly compiled: {:?}", names);
        assert!(!names.contains(&"plain".to_string()), "plain no-loop fn must stay uncompiled: {:?}", names);
    }
    // --- Phase 6: int **, f64 % and **, mixed int/float, local arrays ---
    #[test] fn pow_small_ints() {
        same_jit("fn f(a,b){ a ** b } fn main(){ f(2,10) + f(3,0) + f(1,4294967295) + f(0-2,3) }");
    }
    #[test] fn pow_overflow_promotes_bigint() {
        // 2**63 overflows i64 -> deopt -> VM/interp promote to BigInt
        same_jit("fn f(a,b){ a ** b } fn main(){ f(2,63) }");
    }
    #[test] fn pow_negative_exponent_deopts_to_float() {
        same_jit("fn f(a,b){ a ** b } fn main(){ f(2, 0-2) }");
    }
    #[test] fn pow_in_loop() {
        same_jit("fn s(n){ t=0; for i in 0..n { t = t + 2 ** i }; t } fn main(){ s(20) }");
    }
    #[test] fn float_rem_and_pow() {
        same_jit("fn f(a,b){ (a % b) + (a ** b) + ((0.0-a) % b) } fn main(){ f(7.5, 2.25) + f(9.0, 0.5) }");
    }
    #[test] fn float_mixed_int_literals() {
        // int literals in float math now compile natively on the f64 track
        same_jit("fn f(a){ a * 2 + 1 } fn main(){ f(1.5) }");
    }
    #[test] fn float_mixed_compare_with_int_zero() {
        same_jit("fn f(x){ if x > 0 { x * 0.5 } else { 0.0 - x } } fn main(){ f(3.5) + f(0.0-2.0) }");
    }
    #[test] fn float_for_range_body() {
        same_jit("fn s(n){ t = 0.0; for i in 0..1000 { t = t + i * 0.001 + n * 0 }; t } fn main(){ s(0.0) }");
    }
    #[test] fn float_int_division_stays_off_track() {
        // 7/2 is exact integer division in the interp; the f64 track must reject it
        use super::{float_eligible_set, eligible_set};
        let mut prog = parse_program("fn f(a){ (7/2) + a } fn main(){ f(0.5) }").unwrap();
        fold_program(&mut prog);
        let ints = eligible_set(&prog);
        assert!(!float_eligible_set(&prog, &ints).contains("f"));
    }
    #[test] fn array_sum() {
        same_jit("fn s(n){ a=[]; for i in 0..n { push(a, i*i) }; t=0; for i in 0..len(a) { t=t+a[i] }; t } fn main(){ s(100) }");
    }
    #[test] fn array_sieve() {
        same_jit("fn primes(n){ s=[]; for i in 0..n { push(s, 1) }; s[0]=0; s[1]=0; i=2; while i*i<n { if s[i]==1 { j=i*i; while j<n { s[j]=0; j=j+i } }; i=i+1 }; c=0; for k in 0..n { c=c+s[k] }; c } fn main(){ primes(1000) }");
    }
    #[test] fn array_alias_shares_storage() {
        same_jit("fn f(){ a=[1,2,3]; b=a; b[0]=99; a[0]+a[1] } fn main(){ f() }");
    }
    #[test] fn array_pop_and_oob_deopt() {
        // in-range pops native; the out-of-bounds read deopts and re-raises the
        // interpreter's exact error on the VM
        same_jit("fn f(){ a=[10,20]; x=pop(a); a[5] + x } fn main(){ f() }");
    }
    #[test] fn array_fn_is_eligible() {
        use super::eligible_set;
        let mut prog = parse_program(
            "fn s(n){ a=[]; for i in 0..n { push(a, i) }; len(a) } fn main(){ s(5) }").unwrap();
        fold_program(&mut prog);
        assert!(eligible_set(&prog).contains("s"), "array fn must be on the i64 track");
    }
    #[test] fn array_escape_is_ineligible() {
        use super::eligible_set;
        let mut prog = parse_program(
            "fn f(){ a=[1]; a } fn main(){ len(f()) }").unwrap();
        fold_program(&mut prog);
        assert!(!eligible_set(&prog).contains("f"), "escaping array must stay off the JIT");
    }
    #[test] fn array_tiered_hot() {
        let names = same_tiered(
            "fn s(n){ a=[]; for i in 0..n { push(a, i) }; t=0; for i in 0..len(a) { t=t+a[i] }; t }\n\
             fn main(){ t=0; for k in 0..300 { t = t + s(20) }; t }", 50);
        assert!(names.contains(&"s".to_string()), "hot array fn must compile");
    }

    // --- Phase 11: unified numeric (mixed int/float) track ---
    #[test] fn numeric_mandel_kernel() {
        // int loop counters + float math + int accumulator + int return — the
        // shape neither scalar track accepts; result must match the VM exactly
        same_jit("fn count(w,h,m){ total=0; for py in 0..h { for px in 0..w {\n\
            x0 = to_float(px)/to_float(w)*3.5 - 2.5; y0 = to_float(py)/to_float(h)*2.0 - 1.0;\n\
            x=0.0; y=0.0; it=0;\n\
            while x*x+y*y <= 4.0 && it < m { xt=x*x-y*y+x0; y=2.0*x*y+y0; x=xt; it=it+1 }\n\
            if it==m { total=total+1 } } } total }\n\
            fn main(){ count(60,60,50) }");
    }
    #[test] fn numeric_float_return() {
        // mixed body returning a float — result carried back as f64 bits
        same_jit("fn area(n){ s=0.0; for i in 1..n { s = s + 1.0/to_float(i) }; s }\n\
            fn main(){ area(1000) }");
    }
    #[test] fn numeric_int_overflow_deopts() {
        // an int accumulator that overflows must deopt and match BigInt promotion
        same_jit("fn f(n){ p=1; for i in 1..n { p = p * i }; total=0; x=0.0;\n\
            while x < 2.0 { x = x + 0.5; total = total + 1 }; p + total }\n\
            fn main(){ f(25) }");
    }
    #[test] fn numeric_to_int_trunc() {
        same_jit("fn f(n){ acc=0; for i in 0..n { acc = acc + to_int(to_float(i)*1.5) }; acc }\n\
            fn main(){ f(100) }");
    }
    #[test] fn numeric_tiered_hot() {
        let names = same_tiered(
            "fn kern(n){ s=0.0; c=0; for i in 0..n { s = s + to_float(i)*0.5; c = c + 1 }; c }\n\
             fn main(){ t=0; for k in 0..300 { t = t + kern(10) }; t }", 50);
        assert!(names.contains(&"kern".to_string()), "hot numeric fn must compile");
    }

    // The native object backend emits an object across the full numeric surface
    // (int / float / numeric-mixed / arrays), and declines (falls back) for
    // non-numeric programs. Correctness of the emitted code is proven end-to-end
    // by the build's oracle gate; here we assert emit/decline + the needs_runtime
    // flag (pure-integer array-free programs must link only libc).
    #[test] fn native_object_integer_program() {
        let src = "fn fib(n){ if n<2 {n} else {fib(n-1)+fib(n-2)} } fn main(){ print(fib(10)) }";
        let mut prog = parse_program(src).expect("parse");
        fold_program(&mut prog);
        let (obj, needs_rt) = super::compile_object(&prog, super::NativeTarget::Host).expect("integer program must emit");
        assert!(!obj.is_empty(), "object must be non-empty bytes");
        assert!(!needs_rt, "pure-integer program must link only libc (no runtime)");
    }
    #[test] fn native_object_float_program() {
        let src = "fn area(r){ 3.14159 * r * r } fn main(){ print(area(2.0)) }";
        let mut prog = parse_program(src).expect("parse");
        fold_program(&mut prog);
        let (obj, needs_rt) = super::compile_object(&prog, super::NativeTarget::Host).expect("float program must emit");
        assert!(!obj.is_empty());
        assert!(needs_rt, "float program needs the runtime float printer");
    }
    #[test] fn native_object_numeric_and_array_programs() {
        // numeric-mixed (mandel-style: float math, int result) and integer arrays
        for src in [
            "fn k(n){ t=0; for i in 0..n { if to_float(i)*0.5 > 1.0 { t=t+1 } }; t }\n\
             fn main(){ print(k(10)) }",
            "fn c(n){ s=[]; for i in 0..n { push(s,1) }; s[0]=0; t=0; \
             for k in 0..n { t=t+s[k] }; t } fn main(){ print(c(50)) }",
        ] {
            let mut prog = parse_program(src).expect("parse");
            fold_program(&mut prog);
            let (obj, needs_rt) = super::compile_object(&prog, super::NativeTarget::Host)
                .unwrap_or_else(|| panic!("numeric/array program must emit: {}", src));
            assert!(!obj.is_empty());
            assert!(needs_rt, "numeric/array program links the runtime: {}", src);
        }
    }
    // Exercises the multicore compile path: >= PAR_MIN functions so define_parallel
    // fans out across the test host's cores. JIT output must still equal the VM
    // (byte-identical), proving parallel compilation is correctness-preserving.
    #[test] fn parallel_compile_many_functions_match_vm() {
        let mut src = String::new();
        for i in 0..12 { src.push_str(&format!("fn f{i}(n){{ t=0; for k in 0..n {{ t=t+k*{}-{i} }}; t }}\n", i + 1)); }
        src.push_str("fn agg(){ ");
        src.push_str(&(0..12).map(|i| format!("f{i}(9)")).collect::<Vec<_>>().join(" + "));
        src.push_str(" }\nfn main(){ agg() }");
        same_jit(&src);
    }
    #[test] fn native_object_cross_arch() {
        // the same Cranelift IR emits a real object for each cross target — proves
        // the aarch64 + riscv64 backends are compiled in and lower our IR (incl. the
        // riscv64 manual overflow path). Independent of a linker or qemu, which the
        // build's oracle gate needs but this test does not.
        let src = "fn fib(n){ if n<2 {n} else {fib(n-1)+fib(n-2)} } fn main(){ print(fib(10)) }";
        let mut prog = parse_program(src).expect("parse");
        fold_program(&mut prog);
        for (t, label) in [
            (super::NativeTarget::Host, "host"),
            (super::NativeTarget::Aarch64, "aarch64"),
            (super::NativeTarget::Riscv64, "riscv64"),
        ] {
            let (obj, _) = super::compile_object(&prog, t)
                .unwrap_or_else(|| panic!("{} target must emit an object", label));
            assert!(obj.len() > 64, "{} object must be real ELF bytes", label);
        }
    }
    #[test] fn native_object_multi_print() {
        // multiple print()s in main -> all emit (numeric surface), needs runtime.
        for src in [
            "fn fib(n){ if n<2 {n} else {fib(n-1)+fib(n-2)} } fn main(){ print(fib(5)); print(fib(6)) }",
            "fn area(r){ 3.14*r*r } fn main(){ print(1); print(area(2.0)); print(0-7) }",
        ] {
            let mut prog = parse_program(src).expect("parse");
            fold_program(&mut prog);
            let (obj, needs_rt) = super::compile_object(&prog, super::NativeTarget::Host)
                .unwrap_or_else(|| panic!("multi-print must emit: {}", src));
            assert!(!obj.is_empty());
            assert!(needs_rt, "multi-print uses the runtime printers: {}", src);
        }
    }
    #[test] fn native_object_declines_nonnumeric() {
        for src in [
            "fn main(){ print(true) }",                          // bool arg (boxed tier)
            "fn main(){ let x=1; print(x) }",                    // non-print statement
            "fn f(n){ \"v=\" + str(n * 2) } fn main(){ print(f(5)) }", // arithmetic in string
            "fn greet(n){ n } fn main(){ print(greet(\"hi\")) }", // runtime string via fn
        ] {
            let mut prog = parse_program(src).expect("parse");
            fold_program(&mut prog);
            assert!(super::compile_object(&prog, super::NativeTarget::Host).is_none(),
                    "must decline non-native program: {}", src);
        }
    }
    // Constant-string programs compile to a real object: a bare literal, string
    // concatenation, an f-string with a constant hole, `str()` of a constant, and
    // a mix of string + numeric prints. The build's oracle gate verifies the bytes.
    #[test] fn native_object_string_programs() {
        for src in [
            "fn main(){ print(\"Hello, World!\") }",             // single literal
            "fn main(){ print(\"a\" + \"b\") }",                  // concatenation
            "fn main(){ print(f\"n={42}\") }",                   // f-string, const hole
            "fn main(){ print(\"x=\" + str(7)) }",               // str() of a constant
            "fn main(){ print(upper(\"hi\") + lower(\"BYE\")) }", // constant case fold
            "fn main(){ print(\"result:\"); print(6*7) }",       // mixed string + number
        ] {
            let mut prog = parse_program(src).expect("parse");
            fold_program(&mut prog);
            let (obj, _) = super::compile_object(&prog, super::NativeTarget::Host)
                .unwrap_or_else(|| panic!("must emit an object: {}", src));
            assert!(!obj.is_empty(), "non-empty object: {}", src);
        }
    }
    // Dynamic strings, increment 1: a user function that composes a string from a
    // string parameter (concat / f-string / another string function) compiles to a
    // real object via the StrGen track. The build's oracle gate verifies output.
    #[test] fn native_object_string_functions() {
        for src in [
            "fn greet(name){ \"Hello, \" + name + \"!\" } fn main(){ print(greet(\"Nova\")) }",
            "fn full(a, b){ a + \" \" + b } fn main(){ print(full(\"Ada\", \"Lovelace\")) }",
            "fn hi(name){ f\"Hi {name}!\" } fn main(){ print(hi(\"X\")) }",
            "fn wrap(s){ \"[\" + s + \"]\" } fn main(){ print(wrap(\"a\")); print(6*7) }",
            // pure pass-through composition of string functions (no literal of its
            // own) is eligible via calls to string-eligible functions.
            "fn wrap(s){ \"[\" + s + \"]\" } fn deco(s){ wrap(wrap(s)) } fn main(){ print(deco(\"a\")) }",
            // increment 2: numeric parameters, boxed to their string form at the
            // call site — `str(n)`, a numeric f-string hole, implicit `+` coercion.
            "fn label(n){ \"count: \" + str(n) } fn main(){ print(label(5)) }",
            "fn tag(name, n){ f\"{name}: {n}\" } fn main(){ print(tag(\"age\", 30)) }",
            "fn f(name, n){ name + \" has \" + n } fn main(){ print(f(\"bob\", 7)) }",
            // increment 2b: float parameters / literals, formatted via float_str.
            "fn dim(w){ \"w=\" + str(w) } fn main(){ print(dim(2.5)) }",
            "fn pt(name, x){ f\"{name}={x} (pi {3.14})\" } fn main(){ print(pt(\"r\", 1.5)) }",
            // upper/lower: ASCII case folding on a string parameter (runtime).
            "fn shout(s){ upper(s) + \"!\" } fn main(){ print(shout(\"go\")) }",
        ] {
            let mut prog = parse_program(src).expect("parse");
            fold_program(&mut prog);
            let (obj, needs_rt) = super::compile_object(&prog, super::NativeTarget::Host)
                .unwrap_or_else(|| panic!("string-fn program must emit: {}", src));
            assert!(!obj.is_empty(), "non-empty object: {}", src);
            assert!(needs_rt, "string track links the runtime: {}", src);
        }
    }
    // The string track lowers the same IR for every arch — proves the aarch64 and
    // riscv64 backends compile a StrGen function (independent of a linker/qemu,
    // which the build's oracle gate needs but this test does not).
    #[test] fn native_object_string_fn_cross_arch() {
        let src = "fn greet(name){ \"Hi, \" + name } fn main(){ print(greet(\"Nova\")) }";
        let mut prog = parse_program(src).expect("parse");
        fold_program(&mut prog);
        for t in [super::NativeTarget::Host, super::NativeTarget::Aarch64, super::NativeTarget::Riscv64] {
            let (obj, _) = super::compile_object(&prog, t).expect("must emit for each target");
            assert!(obj.len() > 64);
        }
    }
}

} // mod cl (jit feature)
