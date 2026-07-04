// Source obfuscation for `#[obfuscate]` / `nova obfuscate <file>`.
//
// This is a real, defensible transform — NOT encryption. It performs consistent
// alpha-renaming of a function's *local* identifiers (parameters and every
// binding introduced inside the body: `let`s, loop variables, comprehension and
// lambda binders, `catch`/`select` bindings, and match-pattern bindings) to
// opaque names (`_v0`, `_v1`, …). Because the rename is a 1:1 substitution over
// the set of local names, any lexical shadowing is preserved automatically and
// the program's behaviour is byte-identical. Names that are part of the public
// surface — function names, struct/enum names, struct field keys, method names —
// are deliberately left untouched, as are any local names that collide with a
// top-level item name (kept as-is to avoid local/global ambiguity).

use crate::ast::*;
use std::collections::{HashMap, HashSet};

// Obfuscate every function named in `targets` (or all user functions when
// `targets` is None). Impl-block methods are obfuscated too when selected.
pub fn obfuscate_program(prog: &mut Program, targets: &Option<HashSet<String>>) {
    // Top-level names that must never be captured by a local rename.
    let mut globals: HashSet<String> = HashSet::new();
    for it in &prog.items {
        match it {
            Item::Func(f) => { globals.insert(f.name.clone()); }
            Item::Struct(s) => { globals.insert(s.name.clone()); }
            Item::Enum(e) => {
                globals.insert(e.name.clone());
                for v in &e.variants { globals.insert(v.name.clone()); }
            }
            Item::Const { name, .. } => { globals.insert(name.clone()); }
            Item::Trait(t) => { globals.insert(t.name.clone()); }
            Item::Machine(m) => { globals.insert(m.name.clone()); }
            Item::TypeAlias { name, .. } => { globals.insert(name.clone()); }
            _ => {}
        }
    }

    let want = |name: &str| targets.as_ref().map_or(true, |t| t.contains(name));

    for it in prog.items.iter_mut() {
        match it {
            Item::Func(f) if want(&f.name) => obfuscate_func(f, &globals),
            Item::Impl(b) => {
                for m in b.methods.iter_mut() {
                    if want(&m.name) { obfuscate_func(m, &globals); }
                }
            }
            _ => {}
        }
    }
}

fn obfuscate_func(f: &mut Func, globals: &HashSet<String>) {
    // 1) gather every local binding name in the function.
    let mut locals: HashSet<String> = HashSet::new();
    for p in &f.params { locals.insert(p.clone()); }
    for s in &f.body { collect_stmt(s, &mut locals); }
    // 2) build the rename map, skipping names that shadow a global item and the
    //    conventional receiver `self` (kept for readability / method dispatch).
    let mut map: HashMap<String, String> = HashMap::new();
    let mut n = 0usize;
    let mut names: Vec<String> = locals.into_iter().collect();
    names.sort(); // deterministic output
    for name in names {
        if name == "self" || globals.contains(&name) { continue; }
        map.insert(name, format!("_v{}", n));
        n += 1;
    }
    if map.is_empty() { return; }
    // 3) apply it to params, the whole body, and any contract-attribute predicate
    //    expressions (which reference parameters and must track the rename so the
    //    reformatted `#[requires(...)]`/`#[ensures(...)]` stays consistent).
    for p in f.params.iter_mut() { if let Some(r) = map.get(p) { *p = r.clone(); } }
    for s in f.body.iter_mut() { rename_stmt(s, &map); }
    for a in f.attrs.iter_mut() {
        if matches!(a.name.as_str(), "requires" | "ensures" | "assumes") {
            for e in a.exprs.iter_mut() { rename_expr(e, &map); }
        }
    }
}

fn rn(name: &mut String, map: &HashMap<String, String>) {
    if let Some(r) = map.get(name) { *name = r.clone(); }
}

// -------- pass 1: collect binding names --------------------------------------

fn collect_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Let { name, value, .. } => { out.insert(name.clone()); collect_expr(value, out); }
        Stmt::Assign { value, .. } => collect_expr(value, out),
        Stmt::IndexAssign { base, index, value } => { collect_expr(base, out); collect_expr(index, out); collect_expr(value, out); }
        Stmt::FieldAssign { base, value, .. } => { collect_expr(base, out); collect_expr(value, out); }
        Stmt::Expr(e) | Stmt::Throw(e) => collect_expr(e, out),
        Stmt::Return(o) | Stmt::Yield(o) | Stmt::Break(o) => { if let Some(e) = o { collect_expr(e, out); } }
        Stmt::If { cond, then, els } => {
            collect_expr(cond, out);
            for st in then { collect_stmt(st, out); }
            if let Some(e) = els { for st in e { collect_stmt(st, out); } }
        }
        Stmt::While { cond, body } => { collect_expr(cond, out); for st in body { collect_stmt(st, out); } }
        Stmt::ForRange { var, start, end, body, .. } => {
            out.insert(var.clone()); collect_expr(start, out); collect_expr(end, out);
            for st in body { collect_stmt(st, out); }
        }
        Stmt::ForEach { var, iter, body } => {
            out.insert(var.clone()); collect_expr(iter, out);
            for st in body { collect_stmt(st, out); }
        }
        Stmt::Continue => {}
        Stmt::Defer(b) => { for st in b { collect_stmt(st, out); } }
        Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
            for st in body { collect_stmt(st, out); }
            if let Some(v) = catch_var { out.insert(v.clone()); }
            if let Some(b) = catch_body { for st in b { collect_stmt(st, out); } }
            if let Some(b) = finally_body { for st in b { collect_stmt(st, out); } }
        }
    }
}

fn collect_pattern(p: &Pattern, out: &mut HashSet<String>) {
    match p {
        Pattern::Binding(n) => { out.insert(n.clone()); }
        Pattern::EnumVariant { sub, .. } | Pattern::Or(sub) | Pattern::Tuple(sub) => {
            for s in sub { collect_pattern(s, out); }
        }
        Pattern::Struct { fields, .. } => { for (_, sp) in fields { collect_pattern(sp, out); } }
        Pattern::Slice { prefix, rest, suffix } => {
            for s in prefix { collect_pattern(s, out); }
            if let Some(Some(n)) = rest { out.insert(n.clone()); }
            for s in suffix { collect_pattern(s, out); }
        }
        _ => {}
    }
}

fn collect_expr(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Array(xs) | Expr::SetLit(xs) => { for x in xs { collect_expr(x, out); } }
        Expr::MapLit(kvs) => { for (k, v) in kvs { collect_expr(k, out); collect_expr(v, out); } }
        Expr::Comprehension { body, var, iter, cond } => {
            out.insert(var.clone());
            collect_expr(body, out); collect_expr(iter, out);
            if let Some(c) = cond { collect_expr(c, out); }
        }
        Expr::FmtStr(parts) => { for p in parts { if let FmtPart::Expr(x) = p { collect_expr(x, out); } } }
        Expr::Index { base, index } => { collect_expr(base, out); collect_expr(index, out); }
        Expr::RangeLit { lo, hi, .. } => { if let Some(x) = lo { collect_expr(x, out); } if let Some(x) = hi { collect_expr(x, out); } }
        Expr::StructLit { fields, .. } => { for (_, v) in fields { collect_expr(v, out); } }
        Expr::Field { base, .. } | Expr::SafeField { base, .. } => collect_expr(base, out),
        Expr::MethodCall { base, args, .. } => { collect_expr(base, out); for a in args { collect_expr(a, out); } }
        Expr::Lambda { params, body } => {
            for p in params { out.insert(p.clone()); }
            match &**body { LambdaBody::Expr(x) => collect_expr(x, out), LambdaBody::Block(b) => { for st in b { collect_stmt(st, out); } } }
        }
        Expr::CallValue { callee, args } => { collect_expr(callee, out); for a in args { collect_expr(a, out); } }
        Expr::Unary { expr, .. } => collect_expr(expr, out),
        Expr::Binary { lhs, rhs, .. } => { collect_expr(lhs, out); collect_expr(rhs, out); }
        Expr::Call { args, .. } => { for a in args { collect_expr(a, out); } }
        Expr::Block { stmts, tail } => { for st in stmts { collect_stmt(st, out); } if let Some(t) = tail { collect_expr(t, out); } }
        Expr::If { cond, then, els } => { collect_expr(cond, out); collect_expr(then, out); collect_expr(els, out); }
        Expr::Match { scrutinee, arms } => {
            collect_expr(scrutinee, out);
            for a in arms { collect_pattern(&a.pattern, out); if let Some(g) = &a.guard { collect_expr(g, out); } collect_expr(&a.body, out); }
        }
        Expr::Await(x) | Expr::Recv(x) => collect_expr(x, out),
        Expr::Spawn(b) => { for st in b { collect_stmt(st, out); } }
        Expr::Send { chan, value } => { collect_expr(chan, out); collect_expr(value, out); }
        Expr::Select(arms) => {
            for a in arms { collect_expr(&a.chan, out); if let Some(b) = &a.binding { out.insert(b.clone()); } collect_expr(&a.body, out); }
        }
        Expr::At { expr, .. } => collect_expr(expr, out),
        // leaves and non-binding idents
        Expr::Int(_) | Expr::BigIntLit(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_)
        | Expr::Null | Expr::Ident(_) => {}
    }
}

// -------- pass 2: apply the rename -------------------------------------------

fn rename_stmt(s: &mut Stmt, map: &HashMap<String, String>) {
    match s {
        Stmt::Let { name, value, .. } => { rn(name, map); rename_expr(value, map); }
        Stmt::Assign { name, value } => { rn(name, map); rename_expr(value, map); }
        Stmt::IndexAssign { base, index, value } => { rename_expr(base, map); rename_expr(index, map); rename_expr(value, map); }
        Stmt::FieldAssign { base, value, .. } => { rename_expr(base, map); rename_expr(value, map); }
        Stmt::Expr(e) | Stmt::Throw(e) => rename_expr(e, map),
        Stmt::Return(o) | Stmt::Yield(o) | Stmt::Break(o) => { if let Some(e) = o { rename_expr(e, map); } }
        Stmt::If { cond, then, els } => {
            rename_expr(cond, map);
            for st in then { rename_stmt(st, map); }
            if let Some(e) = els { for st in e { rename_stmt(st, map); } }
        }
        Stmt::While { cond, body } => { rename_expr(cond, map); for st in body { rename_stmt(st, map); } }
        Stmt::ForRange { var, start, end, body, .. } => {
            rn(var, map); rename_expr(start, map); rename_expr(end, map);
            for st in body { rename_stmt(st, map); }
        }
        Stmt::ForEach { var, iter, body } => {
            rn(var, map); rename_expr(iter, map);
            for st in body { rename_stmt(st, map); }
        }
        Stmt::Continue => {}
        Stmt::Defer(b) => { for st in b { rename_stmt(st, map); } }
        Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
            for st in body { rename_stmt(st, map); }
            if let Some(v) = catch_var { rn(v, map); }
            if let Some(b) = catch_body { for st in b { rename_stmt(st, map); } }
            if let Some(b) = finally_body { for st in b { rename_stmt(st, map); } }
        }
    }
}

fn rename_pattern(p: &mut Pattern, map: &HashMap<String, String>) {
    match p {
        Pattern::Binding(n) => rn(n, map),
        Pattern::EnumVariant { sub, .. } | Pattern::Or(sub) | Pattern::Tuple(sub) => {
            for s in sub { rename_pattern(s, map); }
        }
        Pattern::Struct { fields, .. } => { for (_, sp) in fields { rename_pattern(sp, map); } }
        Pattern::Slice { prefix, rest, suffix } => {
            for s in prefix { rename_pattern(s, map); }
            if let Some(Some(n)) = rest { rn(n, map); }
            for s in suffix { rename_pattern(s, map); }
        }
        _ => {}
    }
}

fn rename_expr(e: &mut Expr, map: &HashMap<String, String>) {
    match e {
        Expr::Ident(n) => rn(n, map),
        Expr::Array(xs) | Expr::SetLit(xs) => { for x in xs { rename_expr(x, map); } }
        Expr::MapLit(kvs) => { for (k, v) in kvs { rename_expr(k, map); rename_expr(v, map); } }
        Expr::Comprehension { body, var, iter, cond } => {
            rn(var, map); rename_expr(body, map); rename_expr(iter, map);
            if let Some(c) = cond { rename_expr(c, map); }
        }
        Expr::FmtStr(parts) => { for p in parts { if let FmtPart::Expr(x) = p { rename_expr(x, map); } } }
        Expr::Index { base, index } => { rename_expr(base, map); rename_expr(index, map); }
        Expr::RangeLit { lo, hi, .. } => { if let Some(x) = lo { rename_expr(x, map); } if let Some(x) = hi { rename_expr(x, map); } }
        Expr::StructLit { fields, .. } => { for (_, v) in fields { rename_expr(v, map); } }
        Expr::Field { base, .. } | Expr::SafeField { base, .. } => rename_expr(base, map),
        Expr::MethodCall { base, args, .. } => { rename_expr(base, map); for a in args { rename_expr(a, map); } }
        Expr::Lambda { params, body } => {
            for p in params { rn(p, map); }
            match &mut **body { LambdaBody::Expr(x) => rename_expr(x, map), LambdaBody::Block(b) => { for st in b { rename_stmt(st, map); } } }
        }
        Expr::CallValue { callee, args } => { rename_expr(callee, map); for a in args { rename_expr(a, map); } }
        Expr::Unary { expr, .. } => rename_expr(expr, map),
        Expr::Binary { lhs, rhs, .. } => { rename_expr(lhs, map); rename_expr(rhs, map); }
        // `name(args)` may call a *local* holding a closure (e.g. `let c = ...; c()`),
        // parsed as a Call with a string callee. Rename the callee through the map:
        // it only fires when `callee` is a known local (globals are excluded from the
        // map), so genuine top-level function calls are left untouched.
        Expr::Call { callee, args } => { rn(callee, map); for a in args { rename_expr(a, map); } }
        Expr::Block { stmts, tail } => { for st in stmts { rename_stmt(st, map); } if let Some(t) = tail { rename_expr(t, map); } }
        Expr::If { cond, then, els } => { rename_expr(cond, map); rename_expr(then, map); rename_expr(els, map); }
        Expr::Match { scrutinee, arms } => {
            rename_expr(scrutinee, map);
            for a in arms { rename_pattern(&mut a.pattern, map); if let Some(g) = &mut a.guard { rename_expr(g, map); } rename_expr(&mut a.body, map); }
        }
        Expr::Await(x) | Expr::Recv(x) => rename_expr(x, map),
        Expr::Spawn(b) => { for st in b { rename_stmt(st, map); } }
        Expr::Send { chan, value } => { rename_expr(chan, map); rename_expr(value, map); }
        Expr::Select(arms) => {
            for a in arms { rename_expr(&mut a.chan, map); if let Some(b) = &mut a.binding { rn(b, map); } rename_expr(&mut a.body, map); }
        }
        Expr::At { expr, .. } => rename_expr(expr, map),
        Expr::Int(_) | Expr::BigIntLit(_) | Expr::Float(_) | Expr::Str(_) | Expr::Bool(_) | Expr::Null => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_program;
    use crate::fmt::format_program;

    #[test]
    fn renames_locals_and_preserves_public_api() {
        let src = "struct P { x: Int, y: Int }\n\
                   fn area(p){ let width = p.x; let height = p.y; width * height }\n\
                   fn main(){ let quad = P { x: 5, y: 6 }; area(quad) }";
        let mut prog = parse_program(src).expect("parse");
        obfuscate_program(&mut prog, &None);
        let out = format_program(&prog);
        // public surface is untouched
        assert!(out.contains("fn area("), "function name must be kept: {}", out);
        assert!(out.contains(".x") && out.contains(".y"), "field access kept: {}", out);
        // locals are gone, replaced by opaque names
        assert!(!out.contains("width") && !out.contains("height") && !out.contains("quad"),
            "local names must be renamed away: {}", out);
        assert!(out.contains("_v"), "opaque locals must appear: {}", out);
        // and the result still parses
        parse_program(&out).expect("obfuscated output must re-parse");
    }

    #[test]
    fn shadow_names_that_collide_with_globals_are_left_alone() {
        // a local named the same as a top-level function is NOT renamed (avoids
        // local/global ambiguity), so the program stays correct.
        let src = "fn helper(n){ n + 1 }\n\
                   fn main(){ let helper = 3; helper + 1 }";
        let mut prog = parse_program(src).expect("parse");
        obfuscate_program(&mut prog, &None);
        let out = format_program(&prog);
        assert!(out.contains("let helper = 3"), "global-colliding local kept: {}", out);
        parse_program(&out).expect("re-parse");
    }

    #[test]
    fn only_marked_functions_when_targeted() {
        let src = "fn a(p){ let loc = p; loc }\nfn b(q){ let loc = q; loc }\nfn main(){ a(1) + b(2) }";
        let mut prog = parse_program(src).expect("parse");
        let mut targets = std::collections::HashSet::new();
        targets.insert("a".to_string());
        obfuscate_program(&mut prog, &Some(targets));
        let out = format_program(&prog);
        // b's local is untouched; a's is renamed
        assert!(out.contains("fn b(q)"), "b params untouched: {}", out);
        // b still has `loc`, a does not
        let b_start = out.find("fn b(").unwrap();
        assert!(out[b_start..].contains("loc"), "b keeps its local: {}", out);
    }
}
