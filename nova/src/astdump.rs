// Canonical AST dump — the reference for self-hosting stage 2 (the parser).
//
// `nova ast <file>` parses with the real pest parser and prints a fully
// parenthesized, canonical S-expression form of the AST, one top-level item
// per line. `selfhost/parser.nova` re-derives the byte-identical form from the
// token stream. The dump is TOTAL (every ast.rs node has a form — no escapes)
// so nothing can hide, and it is gated byte-identical over ALL real files
// (tests/corpus + std + examples) by tests/selfhost_smoke.sh.
//
// The transparent `At` position wrapper is unwrapped so positions never leak.

use crate::ast::*;

pub fn dump(p: &Program) -> String {
    let mut out = String::new();
    for item in &p.items { out.push_str(&item_s(item)); out.push('\n'); }
    out
}

fn opt(s: &Option<String>) -> String { match s { Some(x) => x.clone(), None => "-".into() } }

// ---- items ----------------------------------------------------------------
fn item_s(it: &Item) -> String {
    match it {
        Item::Func(f) => func_s(f),
        Item::Struct(s) => format!("(struct {} ({}))", s.name, s.fields.join(" ")),
        Item::Enum(e) => {
            let vs: Vec<String> = e.variants.iter().map(|v| format!("{}/{}", v.name, v.arity)).collect();
            format!("(enum {} ({}))", e.name, vs.join(" "))
        }
        Item::Impl(b) => {
            let ms: Vec<String> = b.methods.iter().map(func_s).collect();
            format!("(impl {} {} ({}))", b.type_name, opt(&b.trait_name), ms.join(" "))
        }
        Item::Trait(t) => {
            let ds: Vec<String> = t.defaults.iter().map(func_s).collect();
            format!("(trait {} (req {}) (def {}))", t.name, t.required.join(" "), ds.join(" "))
        }
        Item::Use(u) => format!("(use {} {} {} ({}))", u.module, opt(&u.alias),
                                if u.wildcard { "glob" } else { "named" }, u.names.join(" ")),
        Item::Test(t) => format!("(test {} {})", str_lit(&t.name), block_s(&t.body)),
        Item::Machine(m) => {
            let ts: Vec<String> = m.transitions.iter().map(|(f, t, e)| format!("({} {} {})", f, t, str_lit(e))).collect();
            format!("(machine {} {} ({}))", m.name, m.initial, ts.join(" "))
        }
        Item::Const { name, value } => format!("(const {} {})", name, expr_s(value)),
        Item::Macro(m) => format!("(macro {} ({}) {})", m.name, m.params.join(" "), str_lit(&m.body)),
        Item::TypeAlias { name, target, refinement } => match refinement {
            Some(e) => format!("(type {} {} {})", name, target, expr_s(e)),
            None => format!("(type {} {} -)", name, target),
        },
        Item::Extern(fs) => {
            let xs: Vec<String> = fs.iter().map(|x| format!("({} {} {})", x.name, x.arity, x.variadic)).collect();
            format!("(extern ({}))", xs.join(" "))
        }
        Item::Import { path } => format!("(import {})", str_lit(path)),
        Item::Migration { from, to, body } => format!("(migrate {} {} {})", from, to, block_s(body)),
    }
}

fn func_s(f: &Func) -> String {
    let attrs: Vec<String> = f.attrs.iter().map(|a| str_lit(&a.raw)).collect();
    let params: Vec<String> = (0..f.params.len()).map(|i| {
        let ty = f.param_types.get(i).and_then(|t| t.clone()).unwrap_or_else(|| "-".into());
        let md = f.param_modes.get(i).and_then(|m| m.clone()).unwrap_or_else(|| "-".into());
        format!("({} {} {})", f.params[i], ty, md)
    }).collect();
    let eff = match &f.effects {
        None => "(eff none)".to_string(),
        Some(v) if v.is_empty() => "(eff pure)".to_string(),
        Some(v) => format!("(eff {})", v.join(" ")),
    };
    let wh: Vec<String> = f.where_bounds.iter().map(|(n, bs)| format!("({} {})", n, bs.join(" "))).collect();
    format!("(fn (attrs {}) {} {} (gen {}) (params {}) (ret {}) {} (where {}) {})",
        attrs.join(" "),
        if f.is_async { "async" } else { "sync" },
        f.name,
        f.type_params.join(" "),
        params.join(" "),
        opt(&f.ret_type),
        eff,
        wh.join(" "),
        block_s(&f.body))
}

fn block_s(b: &[Stmt]) -> String {
    let mut s = String::from("(do");
    for st in b { s.push(' '); s.push_str(&stmt_s(st)); }
    s.push(')');
    s
}

// ---- statements -----------------------------------------------------------
fn stmt_s(st: &Stmt) -> String {
    match st {
        Stmt::Let { name, ty, value } => format!("(let {} {} {})", name, opt(ty), expr_s(value)),
        Stmt::Assign { name, value } => format!("(set {} {})", name, expr_s(value)),
        Stmt::IndexAssign { base, index, value } => format!("(iset {} {} {})", expr_s(base), expr_s(index), expr_s(value)),
        Stmt::FieldAssign { base, field, value } => format!("(fset {} {} {})", expr_s(base), field, expr_s(value)),
        Stmt::Expr(e) => format!("(ex {})", expr_s(e)),
        Stmt::Return(None) => "(ret)".into(),
        Stmt::Return(Some(e)) => format!("(ret {})", expr_s(e)),
        Stmt::If { cond, then, els } => match els {
            Some(e) => format!("(if {} {} {})", expr_s(cond), block_s(then), block_s(e)),
            None => format!("(if {} {})", expr_s(cond), block_s(then)),
        },
        Stmt::While { cond, body } => format!("(while {} {})", expr_s(cond), block_s(body)),
        Stmt::ForRange { var, start, end, inclusive, body } =>
            format!("(for {} {} {} {} {})", var, expr_s(start), expr_s(end), if *inclusive { "ie" } else { "in" }, block_s(body)),
        Stmt::ForEach { var, iter, body } => format!("(foreach {} {} {})", var, expr_s(iter), block_s(body)),
        Stmt::Throw(e) => format!("(throw {})", expr_s(e)),
        Stmt::Yield(None) => "(yield)".into(),
        Stmt::Yield(Some(e)) => format!("(yield {})", expr_s(e)),
        Stmt::Break(None) => "(brk)".into(),
        Stmt::Break(Some(e)) => format!("(brk {})", expr_s(e)),
        Stmt::Continue => "(cont)".into(),
        Stmt::Defer(b) => format!("(defer {})", block_s(b)),
        Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
            let cat = match (catch_var, catch_body) {
                (v, Some(cb)) => format!(" (catch {} {})", opt(v), block_s(cb)),
                (_, None) => String::new(),
            };
            let fin = match finally_body { Some(fb) => format!(" (finally {})", block_s(fb)), None => String::new() };
            format!("(try {}{}{})", block_s(body), cat, fin)
        }
    }
}

// ---- expressions ----------------------------------------------------------
fn expr_s(e: &Expr) -> String {
    match e {
        Expr::At { expr, .. } => expr_s(expr),
        Expr::Int(n) => format!("(i {})", n),
        Expr::BigIntLit(s) => format!("(bigint {})", s),
        Expr::Float(x) => format!("(f {})", float_s(*x)),
        Expr::Str(s) => format!("(s {})", str_lit(s)),
        Expr::Bool(b) => format!("(b {})", b),
        Expr::Null => "(null)".into(),
        Expr::Ident(x) => format!("(v {})", x),
        Expr::Array(xs) => seq("arr", xs),
        Expr::MapLit(kv) => {
            let ps: Vec<String> = kv.iter().map(|(k, v)| format!("(kv {} {})", expr_s(k), expr_s(v))).collect();
            format!("(map {})", ps.join(" "))
        }
        Expr::SetLit(xs) => seq("set", xs),
        Expr::Comprehension { body, var, iter, cond } => {
            let c = match cond { Some(e) => expr_s(e), None => "-".into() };
            format!("(comp {} {} {} {})", expr_s(body), var, expr_s(iter), c)
        }
        Expr::FmtStr(parts) => {
            let ps: Vec<String> = parts.iter().map(|p| match p {
                FmtPart::Lit(s) => format!("(lit {})", str_lit(s)),
                FmtPart::Expr(e) => format!("(hole {})", expr_s(e)),
            }).collect();
            format!("(fstr {})", ps.join(" "))
        }
        Expr::Index { base, index } => format!("(idx {} {})", expr_s(base), expr_s(index)),
        Expr::RangeLit { lo, hi, inclusive } => {
            let l = match lo { Some(e) => expr_s(e), None => "-".into() };
            let h = match hi { Some(e) => expr_s(e), None => "-".into() };
            format!("(range {} {} {})", l, h, if *inclusive { "ie" } else { "ex" })
        }
        Expr::StructLit { name, fields } => {
            let fs: Vec<String> = fields.iter().map(|(n, e)| format!("({} {})", n, expr_s(e))).collect();
            format!("(slit {} ({}))", name, fs.join(" "))
        }
        Expr::Field { base, field } => format!("(fld {} {})", expr_s(base), field),
        Expr::SafeField { base, field } => format!("(sfld {} {})", expr_s(base), field),
        Expr::MethodCall { base, method, args } => {
            let mut s = format!("(m {} {}", expr_s(base), method);
            for a in args { s.push(' '); s.push_str(&expr_s(a)); }
            s.push(')'); s
        }
        Expr::Lambda { params, body } => {
            let b = match &**body {
                LambdaBody::Expr(e) => format!("(bx {})", expr_s(e)),
                LambdaBody::Block(sts) => format!("(bb {})", block_s(sts)),
            };
            format!("(lam ({}) {})", params.join(" "), b)
        }
        Expr::CallValue { callee, args } => {
            let mut s = format!("(cv {}", expr_s(callee));
            for a in args { s.push(' '); s.push_str(&expr_s(a)); }
            s.push(')'); s
        }
        Expr::Unary { op, expr } => {
            let o = match op { UnOp::Neg => "neg", UnOp::Not => "not", UnOp::BitNot => "bnot" };
            format!("({} {})", o, expr_s(expr))
        }
        Expr::Binary { op, lhs, rhs } => format!("({} {} {})", binop(*op), expr_s(lhs), expr_s(rhs)),
        Expr::Call { callee, args } => {
            let mut s = format!("(c {}", callee);
            for a in args { s.push(' '); s.push_str(&expr_s(a)); }
            s.push(')'); s
        }
        Expr::Block { stmts, tail } => {
            let t = match tail { Some(e) => expr_s(e), None => "-".into() };
            format!("(block {} {})", block_s(stmts), t)
        }
        Expr::If { cond, then, els } => format!("(ife {} {} {})", expr_s(cond), expr_s(then), expr_s(els)),
        Expr::Match { scrutinee, arms } => {
            let a: Vec<String> = arms.iter().map(|arm| {
                let g = match &arm.guard { Some(e) => expr_s(e), None => "-".into() };
                format!("(arm {} {} {})", pat_s(&arm.pattern), g, expr_s(&arm.body))
            }).collect();
            format!("(match {} {})", expr_s(scrutinee), a.join(" "))
        }
        Expr::Await(e) => format!("(await {})", expr_s(e)),
        Expr::Spawn(sts) => format!("(spawn {})", block_s(sts)),
        Expr::Recv(e) => format!("(recv {})", expr_s(e)),
        Expr::Send { chan, value } => format!("(sendc {} {})", expr_s(chan), expr_s(value)),
        Expr::Select(arms) => {
            let a: Vec<String> = arms.iter().map(|arm| {
                format!("(sarm {} {} {})", expr_s(&arm.chan), opt(&arm.binding), expr_s(&arm.body))
            }).collect();
            format!("(select {})", a.join(" "))
        }
    }
}

fn seq(head: &str, xs: &[Expr]) -> String {
    let mut s = format!("({}", head);
    for x in xs { s.push(' '); s.push_str(&expr_s(x)); }
    s.push(')'); s
}

// ---- patterns -------------------------------------------------------------
fn pat_s(p: &Pattern) -> String {
    match p {
        Pattern::Wildcard => "(pany)".into(),
        Pattern::Int(n) => format!("(pi {})", n),
        Pattern::Float(x) => format!("(pf {})", float_s(*x)),
        Pattern::Str(s) => format!("(ps {})", str_lit(s)),
        Pattern::Bool(b) => format!("(pb {})", b),
        Pattern::Null => "(pnull)".into(),
        Pattern::Binding(x) => format!("(pbind {})", x),
        Pattern::EnumVariant { name, sub } => {
            let mut s = format!("(penum {}", name);
            for p in sub { s.push(' '); s.push_str(&pat_s(p)); }
            s.push(')'); s
        }
        Pattern::Or(ps) => { let mut s = String::from("(por"); for p in ps { s.push(' '); s.push_str(&pat_s(p)); } s.push(')'); s }
        Pattern::Range { lo, hi, inclusive } => format!("(prange {} {} {})", lo, hi, if *inclusive { "ie" } else { "ex" }),
        Pattern::Tuple(ps) => { let mut s = String::from("(ptuple"); for p in ps { s.push(' '); s.push_str(&pat_s(p)); } s.push(')'); s }
        Pattern::Struct { name, fields } => {
            let fs: Vec<String> = fields.iter().map(|(n, p)| format!("({} {})", n, pat_s(p))).collect();
            format!("(pstruct {} ({}))", name, fs.join(" "))
        }
        Pattern::Slice { prefix, rest, suffix } => {
            let pre: Vec<String> = prefix.iter().map(pat_s).collect();
            let suf: Vec<String> = suffix.iter().map(pat_s).collect();
            let r = match rest { None => "-".into(), Some(None) => "(rest)".into(), Some(Some(n)) => format!("(rest {})", n) };
            format!("(pslice ({}) {} ({}))", pre.join(" "), r, suf.join(" "))
        }
    }
}

fn binop(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/",
        BinOp::Rem => "%", BinOp::Pow => "**",
        BinOp::Eq => "==", BinOp::Ne => "!=", BinOp::Lt => "<", BinOp::Le => "<=",
        BinOp::Gt => ">", BinOp::Ge => ">=",
        BinOp::And => "&&", BinOp::Or => "||",
        BinOp::BitOr => "bor", BinOp::BitXor => "bxor", BinOp::BitAnd => "band",
        BinOp::Shl => "shl", BinOp::Shr => "shr",
    }
}

// mirror Nova's Value::Float Display (interp.rs) so the dump matches the Nova
// side's `str(float)` exactly: integral finite floats print as `{:.1}`.
fn float_s(x: f64) -> String {
    if x.fract() == 0.0 && x.is_finite() { format!("{:.1}", x) } else { format!("{}", x) }
}

fn str_lit(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}
