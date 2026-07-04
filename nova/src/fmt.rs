// Nova source formatter: an AST pretty-printer that emits clean, canonical Nova
// that is semantically identical to the input. Binary/unary children that are
// themselves binary expressions are parenthesized, which makes the printed
// precedence explicit and guarantees the output re-parses to the same structure.
//
// Limitations (documented): comments are not preserved (the grammar discards
// them) and macro calls print in their already-expanded form.

use crate::ast::*;

pub fn format_program(p: &Program) -> String {
    let mut out = String::new();
    for item in &p.items {
        let mut f = Fmt { buf: String::new(), depth: 0 };
        f.item(item);
        if f.buf.trim().is_empty() { continue; } // e.g. macros (expanded away)
        if !out.is_empty() { out.push('\n'); }
        out.push_str(&f.buf);
    }
    if !out.ends_with('\n') { out.push('\n'); }
    out
}

struct Fmt {
    buf: String,
    depth: usize,
}

impl Fmt {
    fn pad(&mut self) {
        for _ in 0..self.depth { self.buf.push_str("  "); }
    }
    fn line(&mut self, s: &str) {
        self.pad();
        self.buf.push_str(s);
        self.buf.push('\n');
    }

    // ---- items ----
    fn item(&mut self, it: &Item) {
        match it {
            Item::Func(func) => self.func(func, true),
            // records print in canonical `data` form (the AST keeps field names,
            // not their declared types, so `data` — whose field types are optional
            // — is the faithful round-tripping form).
            Item::Struct(s) => self.line(&format!("data {}({});", s.name, s.fields.join(", "))),
            Item::Enum(e) => {
                let vs: Vec<String> = e.variants.iter().map(|v| {
                    if v.arity == 0 { v.name.clone() }
                    else {
                        let slots: Vec<&str> = (0..v.arity).map(|_| "_").collect();
                        format!("{}({})", v.name, slots.join(", "))
                    }
                }).collect();
                self.line(&format!("enum {} {{ {} }}", e.name, vs.join(", ")));
            }
            Item::Impl(im) => {
                let head = match &im.trait_name {
                    Some(t) => format!("impl {} for {} {{", t, im.type_name),
                    None => format!("impl {} {{", im.type_name),
                };
                self.line(&head);
                self.depth += 1;
                for (i, m) in im.methods.iter().enumerate() {
                    if i > 0 { self.buf.push('\n'); }
                    self.func(m, true);
                }
                self.depth -= 1;
                self.line("}");
            }
            Item::Trait(t) => {
                self.line(&format!("trait {} {{", t.name));
                self.depth += 1;
                for r in &t.required { self.line(&format!("fn {}(self);", r)); }
                for d in &t.defaults { self.func(d, true); }
                self.depth -= 1;
                self.line("}");
            }
            Item::Const { name, value } => {
                let v = self.expr_s(value);
                self.line(&format!("const {} = {};", name, v));
            }
            Item::TypeAlias { name, target, refinement } => {
                match refinement {
                    Some(pred) => {
                        let p = self.expr_s(pred);
                        self.line(&format!("type {} = {} if {};", name, target, p));
                    }
                    None => self.line(&format!("type {} = {};", name, target)),
                }
            }
            Item::Extern(fns) => {
                self.line("extern {");
                self.depth += 1;
                for ef in fns {
                    let slots: Vec<&str> = (0..ef.arity).map(|_| "_").collect();
                    let mut args = slots.join(", ");
                    if ef.variadic { if args.is_empty() { args = "...".into(); } else { args.push_str(", ..."); } }
                    self.line(&format!("fn {}({});", ef.name, args));
                }
                self.depth -= 1;
                self.line("}");
            }
            Item::Use(u) => {
                let mut s = format!("use {}", u.module);
                if u.wildcard { s.push_str(".*"); }
                if let Some(a) = &u.alias { s.push_str(&format!(" as {}", a)); }
                s.push(';');
                self.line(&s);
            }
            Item::Import { path } => self.line(&format!("use \"{}\";", path)),
            Item::Test(t) => {
                self.line(&format!("test \"{}\" {{", t.name));
                self.depth += 1;
                self.tail_block(&t.body);
                self.depth -= 1;
                self.line("}");
            }
            Item::Machine(m) => {
                self.line(&format!("machine {} {{", m.name));
                self.depth += 1;
                self.line(&format!("initial {}", m.initial));
                for (from, to, ev) in &m.transitions {
                    self.line(&format!("{} -> {} on \"{}\"", from, to, ev));
                }
                self.depth -= 1;
                self.line("}");
            }
            Item::Macro(_) => { /* macros are expanded at parse time; nothing to print */ }
            Item::Migration { from, to, body } => {
                self.line(&format!("migrate from {} to {} {{", from, to));
                self.depth += 1;
                for s in body { self.stmt(s); }
                self.depth -= 1;
                self.line("}");
            }
        }
    }

    fn func(&mut self, f: &Func, _top: bool) {
        let mut sig = String::new();
        if f.is_async { sig.push_str("async "); }
        sig.push_str("fn ");
        sig.push_str(&f.name);
        if !f.type_params.is_empty() {
            sig.push_str(&format!("[{}]", f.type_params.join(", ")));
        }
        // parameters with optional mode + type
        let mut ps = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            let mut s = p.clone();
            if let Some(Some(ty)) = f.param_types.get(i) {
                let mode = f.param_modes.get(i).and_then(|m| m.as_deref());
                match mode {
                    Some(m) => s.push_str(&format!(": {} {}", m, ty)),
                    None => s.push_str(&format!(": {}", ty)),
                }
            }
            ps.push(s);
        }
        sig.push_str(&format!("({})", ps.join(", ")));
        if let Some(rt) = &f.ret_type { sig.push_str(&format!(" -> {}", rt)); }
        if let Some(effs) = &f.effects { sig.push_str(&format!(" ![{}]", effs.join(", "))); }
        if !f.where_bounds.is_empty() {
            let ws: Vec<String> = f.where_bounds.iter()
                .map(|(g, ts)| format!("{}: {}", g, ts.join(" + "))).collect();
            sig.push_str(&format!(" where {}", ws.join(", ")));
        }
        sig.push_str(" {");
        self.line(&sig);
        self.depth += 1;
        self.tail_block(&f.body);
        self.depth -= 1;
        self.line("}");
    }

    // ---- statements ----
    fn block_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts { self.stmt(s); }
    }

    // Print the body of a function / test / spawn — contexts where the parser's
    // `make_trailing_implicit_return` turned the trailing expression into a
    // `return`. We invert that: the final implicit return prints as a bare
    // expression (idiomatic, and it round-trips exactly when re-parsed).
    fn tail_block(&mut self, stmts: &[Stmt]) {
        if stmts.is_empty() { return; }
        let last = stmts.len() - 1;
        for s in &stmts[..last] { self.stmt(s); }
        self.tail_stmt(&stmts[last]);
    }

    fn tail_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Return(Some(e)) => {
                // a bare trailing if-expression re-parses as an if *statement* whose
                // branches get implicit returns, so print it that way for round-trip.
                let mut inner = e;
                while let Expr::At { expr, .. } = inner { inner = expr; }
                if let Expr::If { cond, then, els } = inner {
                    let c = self.expr_s(cond);
                    self.line(&format!("if {} {{", c));
                    self.depth += 1; self.tail_block(&block_to_stmts(then)); self.depth -= 1;
                    self.line("} else {");
                    self.depth += 1; self.tail_block(&block_to_stmts(els)); self.depth -= 1;
                    self.line("}");
                } else {
                    let v = self.expr_s(e);
                    self.line(&format!("{};", v));
                }
            }
            Stmt::If { cond, then, els } => {
                let c = self.expr_s(cond);
                self.line(&format!("if {} {{", c));
                self.depth += 1; self.tail_block(then); self.depth -= 1;
                match els {
                    Some(e) => {
                        self.line("} else {");
                        self.depth += 1; self.tail_block(e); self.depth -= 1;
                        self.line("}");
                    }
                    None => self.line("}"),
                }
            }
            other => self.stmt(other),
        }
    }

    // inline variant of tail_block, for spawn bodies rendered on one line
    fn inline_body(&self, stmts: &[Stmt]) -> String {
        if stmts.is_empty() { return String::new(); }
        let last = stmts.len() - 1;
        let mut parts: Vec<String> = stmts[..last].iter().map(|s| self.inline_stmt(s)).collect();
        parts.push(self.inline_tail_stmt(&stmts[last]));
        parts.join("; ")
    }

    fn inline_tail_stmt(&self, s: &Stmt) -> String {
        match s {
            Stmt::Return(Some(e)) => {
                let mut inner = e;
                while let Expr::At { expr, .. } = inner { inner = expr; }
                if let Expr::If { cond, then, els } = inner {
                    format!("if {} {{ {} }} else {{ {} }}",
                        self.expr_s(cond),
                        self.inline_body(&block_to_stmts(then)),
                        self.inline_body(&block_to_stmts(els)))
                } else {
                    self.expr_s(e)
                }
            }
            Stmt::If { cond, then, els } => {
                let mut t = format!("if {} {{ {} }}", self.expr_s(cond), self.inline_body(then));
                if let Some(e) = els { t.push_str(&format!(" else {{ {} }}", self.inline_body(e))); }
                t
            }
            other => self.inline_stmt(other),
        }
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, ty, value } => {
                let v = self.expr_s(value);
                match ty {
                    Some(t) => self.line(&format!("let {}: {} = {};", name, t, v)),
                    None => self.line(&format!("let {} = {};", name, v)),
                }
            }
            Stmt::Assign { name, value } => {
                let v = self.expr_s(value);
                self.line(&format!("{} = {};", name, v));
            }
            Stmt::IndexAssign { base, index, value } => {
                let b = self.expr_s(base); let i = self.expr_s(index); let v = self.expr_s(value);
                self.line(&format!("{}[{}] = {};", b, i, v));
            }
            Stmt::FieldAssign { base, field, value } => {
                let b = self.expr_s(base); let v = self.expr_s(value);
                self.line(&format!("{}.{} = {};", b, field, v));
            }
            Stmt::Expr(e) => { let s = self.expr_s(e); self.line(&format!("{};", s)); }
            Stmt::Return(Some(e)) => { let s = self.expr_s(e); self.line(&format!("return {};", s)); }
            Stmt::Return(None) => self.line("return;"),
            Stmt::Throw(e) => { let s = self.expr_s(e); self.line(&format!("throw {};", s)); }
            Stmt::Yield(Some(e)) => { let s = self.expr_s(e); self.line(&format!("yield {};", s)); }
            Stmt::Yield(None) => self.line("yield;"),
            Stmt::Break(Some(e)) => { let s = self.expr_s(e); self.line(&format!("break {};", s)); }
            Stmt::Break(None) => self.line("break;"),
            Stmt::Continue => self.line("continue;"),
            Stmt::Defer(body) => {
                self.line("defer {");
                self.depth += 1; self.block_stmts(body); self.depth -= 1;
                self.line("}");
            }
            Stmt::If { cond, then, els } => self.if_stmt(cond, then, els.as_deref()),
            Stmt::While { cond, body } => {
                let c = self.expr_s(cond);
                self.line(&format!("while {} {{", c));
                self.depth += 1; self.block_stmts(body); self.depth -= 1;
                self.line("}");
            }
            Stmt::ForRange { var, start, end, inclusive, body } => {
                let a = self.expr_s(start); let b = self.expr_s(end);
                let op = if *inclusive { "..=" } else { ".." };
                self.line(&format!("for {} in {}{}{} {{", var, a, op, b));
                self.depth += 1; self.block_stmts(body); self.depth -= 1;
                self.line("}");
            }
            Stmt::ForEach { var, iter, body } => {
                let it = self.expr_s(iter);
                self.line(&format!("for {} in {} {{", var, it));
                self.depth += 1; self.block_stmts(body); self.depth -= 1;
                self.line("}");
            }
            Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
                self.line("try {");
                self.depth += 1; self.block_stmts(body); self.depth -= 1;
                if let Some(cb) = catch_body {
                    match catch_var {
                        Some(v) => self.line(&format!("}} catch {} {{", v)),
                        None => self.line("} catch {"),
                    }
                    self.depth += 1; self.block_stmts(cb); self.depth -= 1;
                }
                if let Some(fb) = finally_body {
                    self.line("} finally {");
                    self.depth += 1; self.block_stmts(fb); self.depth -= 1;
                }
                self.line("}");
            }
        }
    }

    fn if_stmt(&mut self, cond: &Expr, then: &[Stmt], els: Option<&[Stmt]>) {
        let c = self.expr_s(cond);
        self.line(&format!("if {} {{", c));
        self.depth += 1; self.block_stmts(then); self.depth -= 1;
        match els {
            Some(e) => {
                self.line("} else {");
                self.depth += 1; self.block_stmts(e); self.depth -= 1;
                self.line("}");
            }
            None => self.line("}"),
        }
    }

    // ---- expressions (returned as strings; blocks render inline) ----
    fn expr_s(&self, e: &Expr) -> String {
        match e {
            Expr::At { expr, .. } => self.expr_s(expr),
            Expr::Int(n) => n.to_string(),
            Expr::BigIntLit(s) => s.clone(),
            Expr::Float(x) => {
                if x.fract() == 0.0 && x.is_finite() { format!("{:.1}", x) } else { x.to_string() }
            }
            Expr::Str(s) => format!("\"{}\"", esc(s)),
            Expr::Bool(b) => b.to_string(),
            Expr::Null => "null".to_string(),
            Expr::Ident(name) => name.clone(),
            Expr::Array(xs) => format!("[{}]", self.list(xs)),
            Expr::SetLit(xs) => format!("#({})", self.list(xs)),
            Expr::MapLit(pairs) => {
                let parts: Vec<String> = pairs.iter()
                    .map(|(k, v)| format!("{}: {}", self.expr_s(k), self.expr_s(v))).collect();
                format!("#{{{}}}", parts.join(", "))
            }
            Expr::Comprehension { body, var, iter, cond } => {
                let mut s = format!("[{} for {} in {}", self.expr_s(body), var, self.expr_s(iter));
                if let Some(c) = cond { s.push_str(&format!(" if {}", self.expr_s(c))); }
                s.push(']');
                s
            }
            Expr::FmtStr(parts) => {
                let mut s = String::from("f\"");
                for p in parts {
                    match p {
                        FmtPart::Lit(t) => s.push_str(&esc_fstr(t)),
                        FmtPart::Expr(ex) => s.push_str(&format!("{{{}}}", self.expr_s(ex))),
                    }
                }
                s.push('"');
                s
            }
            Expr::Index { base, index } => format!("{}[{}]", self.atom(base), self.expr_s(index)),
            Expr::RangeLit { lo, hi, inclusive } => {
                let l = lo.as_ref().map(|e| self.expr_s(e)).unwrap_or_default();
                let h = hi.as_ref().map(|e| self.expr_s(e)).unwrap_or_default();
                format!("{}{}{}", l, if *inclusive { "..=" } else { ".." }, h)
            }
            Expr::StructLit { name, fields } => {
                let parts: Vec<String> = fields.iter()
                    .map(|(k, v)| format!("{}: {}", k, self.expr_s(v))).collect();
                format!("{} {{ {} }}", name, parts.join(", "))
            }
            Expr::Field { base, field } => format!("{}.{}", self.atom(base), field),
            Expr::SafeField { base, field } => format!("{}?.{}", self.atom(base), field),
            Expr::MethodCall { base, method, args } =>
                format!("{}.{}({})", self.atom(base), method, self.list(args)),
            Expr::Call { callee, args } => format!("{}({})", callee, self.list(args)),
            Expr::CallValue { callee, args } => format!("{}({})", self.atom(callee), self.list(args)),
            Expr::Lambda { params, body } => {
                let head = format!("({})", params.join(", "));
                match body.as_ref() {
                    LambdaBody::Expr(ex) => format!("{} => {}", head, self.expr_s(ex)),
                    LambdaBody::Block(stmts) => format!("{} => {{ {} }}", head, self.inline_stmts(stmts)),
                }
            }
            Expr::Unary { op, expr } => {
                let sym = match op { UnOp::Neg => "-", UnOp::Not => "!", UnOp::BitNot => "~" };
                format!("{}{}", sym, self.atom(expr))
            }
            Expr::Binary { op, lhs, rhs } =>
                format!("{} {} {}", self.atom(lhs), binop_sym(*op), self.atom(rhs)),
            Expr::Block { stmts, tail } => {
                let mut inner = self.inline_stmts(stmts);
                if let Some(t) = tail {
                    if !inner.is_empty() { inner.push_str("; "); }
                    inner.push_str(&self.expr_s(t));
                }
                format!("{{ {} }}", inner)
            }
            Expr::If { cond, then, els } => {
                // then/els are already Block expressions, so they render their own braces
                format!("if {} {} else {}", self.expr_s(cond), self.block_expr(then), self.block_expr(els))
            }
            Expr::Match { scrutinee, arms } => {
                let parts: Vec<String> = arms.iter().map(|a| {
                    let g = a.guard.as_ref().map(|g| format!(" if {}", self.expr_s(g))).unwrap_or_default();
                    format!("{}{} => {}", pattern_s(&a.pattern), g, self.expr_s(&a.body))
                }).collect();
                format!("match {} {{ {} }}", self.expr_s(scrutinee), parts.join(", "))
            }
            Expr::Await(x) => format!("{}.await", self.atom(x)),
            Expr::Spawn(stmts) => format!("spawn {{ {} }}", self.inline_body(stmts)),
            Expr::Recv(x) => format!("<- {}", self.atom(x)),
            Expr::Send { chan, value } => format!("{} <- {}", self.atom(chan), self.expr_s(value)),
            Expr::Select(arms) => {
                // arm syntax is `<chan> => <body>`; the received value is `_recv`
                let parts: Vec<String> = arms.iter()
                    .map(|a| format!("{} => {}", self.expr_s(&a.chan), self.expr_s(&a.body)))
                    .collect();
                format!("select {{ {} }}", parts.join(", "))
            }
        }
    }

    // an expression that needs parentheses when it is a compound (binary/unary/etc.)
    // operand, so the printed precedence/associativity is unambiguous.
    fn atom(&self, e: &Expr) -> String {
        match e {
            Expr::At { expr, .. } => self.atom(expr),
            Expr::Binary { .. } | Expr::Unary { .. } | Expr::Lambda { .. }
            | Expr::If { .. } | Expr::Match { .. } | Expr::Block { .. }
            | Expr::RangeLit { .. } | Expr::Send { .. } | Expr::Recv(_) | Expr::Await(_) => {
                format!("({})", self.expr_s(e))
            }
            _ => self.expr_s(e),
        }
    }

    // render an if-expression branch: already-block / nested-if print as-is,
    // anything else is wrapped in braces so it stays a valid block.
    fn block_expr(&self, e: &Expr) -> String {
        let mut inner = e;
        while let Expr::At { expr, .. } = inner { inner = expr; }
        match inner {
            Expr::Block { .. } | Expr::If { .. } => self.expr_s(e),
            _ => format!("{{ {} }}", self.expr_s(e)),
        }
    }

    fn list(&self, xs: &[Expr]) -> String {
        xs.iter().map(|e| self.expr_s(e)).collect::<Vec<_>>().join(", ")
    }

    // render statements on one line (for inline blocks / lambdas / spawn)
    fn inline_stmts(&self, stmts: &[Stmt]) -> String {
        stmts.iter().map(|s| self.inline_stmt(s)).collect::<Vec<_>>().join("; ")
    }

    // a single statement rendered on one line; nested blocks stay inline too.
    fn inline_stmt(&self, s: &Stmt) -> String {
        match s {
            Stmt::Let { name, ty, value } => match ty {
                Some(t) => format!("let {}: {} = {}", name, t, self.expr_s(value)),
                None => format!("let {} = {}", name, self.expr_s(value)),
            },
            Stmt::Assign { name, value } => format!("{} = {}", name, self.expr_s(value)),
            Stmt::IndexAssign { base, index, value } =>
                format!("{}[{}] = {}", self.expr_s(base), self.expr_s(index), self.expr_s(value)),
            Stmt::FieldAssign { base, field, value } =>
                format!("{}.{} = {}", self.expr_s(base), field, self.expr_s(value)),
            Stmt::Expr(e) => self.expr_s(e),
            Stmt::Return(Some(e)) => format!("return {}", self.expr_s(e)),
            Stmt::Return(None) => "return".to_string(),
            Stmt::Throw(e) => format!("throw {}", self.expr_s(e)),
            Stmt::Yield(Some(e)) => format!("yield {}", self.expr_s(e)),
            Stmt::Yield(None) => "yield".to_string(),
            Stmt::Break(Some(e)) => format!("break {}", self.expr_s(e)),
            Stmt::Break(None) => "break".to_string(),
            Stmt::Continue => "continue".to_string(),
            Stmt::Defer(b) => format!("defer {{ {} }}", self.inline_stmts(b)),
            Stmt::If { cond, then, els } => {
                let mut s = format!("if {} {{ {} }}", self.expr_s(cond), self.inline_stmts(then));
                if let Some(e) = els { s.push_str(&format!(" else {{ {} }}", self.inline_stmts(e))); }
                s
            }
            Stmt::While { cond, body } =>
                format!("while {} {{ {} }}", self.expr_s(cond), self.inline_stmts(body)),
            Stmt::ForRange { var, start, end, inclusive, body } =>
                format!("for {} in {}{}{} {{ {} }}", var, self.expr_s(start),
                    if *inclusive { "..=" } else { ".." }, self.expr_s(end), self.inline_stmts(body)),
            Stmt::ForEach { var, iter, body } =>
                format!("for {} in {} {{ {} }}", var, self.expr_s(iter), self.inline_stmts(body)),
            Stmt::TryCatch { body, catch_var, catch_body, finally_body } => {
                let mut s = format!("try {{ {} }}", self.inline_stmts(body));
                if let Some(cb) = catch_body {
                    match catch_var {
                        Some(v) => s.push_str(&format!(" catch {} {{ {} }}", v, self.inline_stmts(cb))),
                        None => s.push_str(&format!(" catch {{ {} }}", self.inline_stmts(cb))),
                    }
                }
                if let Some(fb) = finally_body { s.push_str(&format!(" finally {{ {} }}", self.inline_stmts(fb))); }
                s
            }
        }
    }
}

// Flatten an if-branch (a Block expression) into a statement list whose trailing
// value becomes an implicit return, so `tail_block` can print it idiomatically.
fn block_to_stmts(e: &Expr) -> Vec<Stmt> {
    let mut inner = e;
    while let Expr::At { expr, .. } = inner { inner = expr; }
    match inner {
        Expr::Block { stmts, tail } => {
            let mut v = stmts.clone();
            if let Some(t) = tail { v.push(Stmt::Return(Some((**t).clone()))); }
            v
        }
        _ => vec![Stmt::Return(Some(e.clone()))],
    }
}

fn pattern_s(p: &Pattern) -> String {
    match p {
        Pattern::Wildcard => "_".to_string(),
        Pattern::Int(n) => n.to_string(),
        Pattern::Float(x) => x.to_string(),
        Pattern::Str(s) => format!("\"{}\"", esc(s)),
        Pattern::Bool(b) => b.to_string(),
        Pattern::Null => "null".to_string(),
        Pattern::Binding(name) => name.clone(),
        Pattern::EnumVariant { name, sub } => {
            if sub.is_empty() { name.clone() }
            else {
                let ps: Vec<String> = sub.iter().map(pattern_s).collect();
                format!("{}({})", name, ps.join(", "))
            }
        }
        Pattern::Or(alts) => alts.iter().map(pattern_s).collect::<Vec<_>>().join(" | "),
        Pattern::Range { lo, hi, inclusive } =>
            format!("{}{}{}", lo, if *inclusive { "..=" } else { ".." }, hi),
        Pattern::Tuple(ps) => format!("({})", ps.iter().map(pattern_s).collect::<Vec<_>>().join(", ")),
        Pattern::Struct { name, fields } => {
            let ps: Vec<String> = fields.iter().map(|(k, sub)| format!("{}: {}", k, pattern_s(sub))).collect();
            format!("{} {{ {} }}", name, ps.join(", "))
        }
        Pattern::Slice { prefix, rest, suffix } => {
            let mut parts: Vec<String> = prefix.iter().map(pattern_s).collect();
            if let Some(r) = rest {
                match r {
                    Some(name) => parts.push(format!("...{}", name)),
                    None => parts.push("...".to_string()),
                }
            }
            parts.extend(suffix.iter().map(pattern_s));
            format!("[{}]", parts.join(", "))
        }
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

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")
}

fn esc_fstr(s: &str) -> String {
    esc(s).replace('{', "{{").replace('}', "}}")
}

#[cfg(test)]
mod fmt_tests {
    use crate::parser::parse_program;
    use super::format_program;

    // formatting is idempotent and produces re-parseable output
    fn round(src: &str) -> String {
        let p = parse_program(src).expect("parse");
        format_program(&p)
    }

    #[test]
    fn idempotent_and_stable() {
        let src = r#"
fn max(a, b) { if a > b { a } else { b } }
fn main() {
  let xs = [1, 2, 3]
  for x in xs { print(x * 2) }
  print(max(3, 7))
}
"#;
        let once = round(src);
        let twice = round(&once);
        assert_eq!(once, twice, "fmt must be idempotent");
        // and the formatted text must itself parse
        assert!(parse_program(&once).is_ok(), "formatted output must re-parse");
    }
}
