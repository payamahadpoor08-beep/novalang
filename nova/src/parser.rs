// Lowers pest's parse tree into the typed Nova AST.
// We only handle the Nova Core subset here; unsupported constructs
// produce a clear error so we always know exactly what's not yet wired up.

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;

thread_local! {
    // Macro definitions collected in a pre-pass, used to expand `name!(...)`
    // call sites during lowering. Maps macro name -> (param names, body template).
    static MACROS: std::cell::RefCell<std::collections::HashMap<String, (Vec<String>, String)>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[derive(Parser)]
#[grammar = "nova.pest"]
struct NovaParser;

// Render a pest syntax error with human-readable rule names instead of the
// internal grammar identifiers (e.g. "an expression" rather than "unary_expr").
fn fmt_parse_error(e: pest::error::Error<Rule>) -> String {
    let e = e.renamed_rules(|r| friendly_rule(&format!("{:?}", r)));
    format!("Parse error:\n{}", e)
}

fn friendly_rule(rule: &str) -> String {
    match rule {
        "unary_expr" | "primary" | "postfix_expr" | "expression" | "expr"
        | "atom" | "value" => "an expression".into(),
        "doc_comment" | "inner_doc" => "a doc comment".into(),
        "statement" | "stmt" => "a statement".into(),
        "ident" | "identifier" | "name" => "a name".into(),
        "type_expr" | "type_name" | "ty" => "a type".into(),
        "EOI" => "end of input".into(),
        other => other.replace('_', " "),
    }
}

pub fn parse_program(src: &str) -> Result<Program, String> {
    // PASS 0: collect macro definitions so call sites can expand them.
    let mut pre = NovaParser::parse(Rule::program, src)
        .map_err(fmt_parse_error)?;
    let prog0 = pre.next().unwrap();
    MACROS.with(|m| m.borrow_mut().clear());
    for p in prog0.into_inner() {
        if p.as_rule() == Rule::top_level {
            if let Some(md) = find_rule(p.clone(), Rule::macro_decl) {
                if let Ok(def) = lower_macro_decl(md) {
                    MACROS.with(|m| {
                        m.borrow_mut().insert(def.name.clone(), (def.params.clone(), def.body.clone()));
                    });
                }
            }
        }
    }
    parse_program_inner(src)
}

// Lowers the program without (re)collecting macros — used both as the second
// phase of parse_program and for re-parsing macro expansions.
fn parse_program_inner(src: &str) -> Result<Program, String> {
    let mut pairs = NovaParser::parse(Rule::program, src)
        .map_err(fmt_parse_error)?;
    let program = pairs.next().unwrap(); // Rule::program
    let mut items = Vec::new();
    for p in program.into_inner() {
        match p.as_rule() {
            Rule::top_level => {
                if let Some(item) = lower_top_level(p)? {
                    items.push(item);
                }
            }
            Rule::EOI => {}
            _ => {}
        }
    }
    Ok(Program { items })
}

// REPL input may be either top-level items (fn/struct/enum/impl/use) or a
// sequence of statements/expressions. Returns (items, statements). Items are
// registered into the interpreter; statements are evaluated in the live scope.
pub fn parse_repl(src: &str) -> Result<(Program, Vec<Stmt>), String> {
    let trimmed = src.trim();
    // Heuristic: if it starts with an item keyword, parse as a program.
    let is_item = trimmed.starts_with("fn ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("impl ")
        || trimmed.starts_with("use ")
        || trimmed.starts_with("pub ");
    if is_item {
        let prog = parse_program(src)?;
        return Ok((prog, Vec::new()));
    }
    // Otherwise treat as statements: wrap in a block and lower its body.
    let wrapped = format!("{{ {} }}", src);
    let mut pairs = NovaParser::parse(Rule::block, &wrapped)
        .map_err(fmt_parse_error)?;
    let block = pairs.next().unwrap();
    let stmts = lower_block_stmts(block)?;
    Ok((Program { items: Vec::new() }, stmts))
}

fn lower_top_level(p: Pair<Rule>) -> Result<Option<Item>, String> {
    // top_level = { attribute* ~ visibility? ~ (use_decl | item) }
    for inner in p.into_inner() {
        match inner.as_rule() {
            Rule::item => return lower_item(inner),
            Rule::use_decl => {
                // `use "file.nova"` imports a file; `use math.*` imports a stdlib module
                if let Some(s) = inner.clone().into_inner().find(|x| x.as_rule() == Rule::string_lit) {
                    let raw = s.as_str();
                    let path = raw[1..raw.len().saturating_sub(1)].to_string();
                    return Ok(Some(Item::Import { path }));
                }
                return Ok(Some(Item::Use(lower_use(inner)?)));
            }
            Rule::attribute | Rule::visibility => {}
            _ => {}
        }
    }
    Ok(None)
}

fn lower_item(p: Pair<Rule>) -> Result<Option<Item>, String> {
    let inner = p.into_inner().next().ok_or("empty item")?;
    match inner.as_rule() {
        Rule::func_decl => Ok(Some(Item::Func(lower_func(inner)?))),
        Rule::struct_decl => Ok(Some(Item::Struct(lower_struct(inner)?))),
        Rule::impl_block => Ok(Some(Item::Impl(lower_impl(inner)?))),
        Rule::enum_decl => Ok(Some(Item::Enum(lower_enum(inner)?))),
        Rule::test_block => Ok(Some(Item::Test(lower_test(inner)?))),
        Rule::machine_decl => Ok(Some(Item::Machine(lower_machine(inner)?))),
        Rule::const_decl => lower_const(inner).map(Some),
        Rule::static_decl => lower_const(inner).map(Some),
        Rule::data_decl => Ok(Some(Item::Struct(lower_data(inner)?))),
        Rule::trait_decl => Ok(Some(Item::Trait(lower_trait(inner)?))),
        Rule::macro_decl => Ok(Some(Item::Macro(lower_macro_decl(inner)?))),
        Rule::type_decl => lower_type_alias(inner).map(Some),
        Rule::extern_block => lower_extern(inner).map(Some),
        other => Err(format!("Unsupported top-level item (not yet in Nova Core): {:?}", other)),
    }
}

fn lower_try_catch(p: Pair<Rule>) -> Result<Stmt, String> {
    // try_catch_expr = { kw_try ~ block ~ (kw_catch ~ ident? ~ block)? ~ (kw_finally ~ block)? }
    let mut catch_var: Option<String> = None;
    let mut body: Vec<Stmt> = Vec::new();
    let mut catch_body: Option<Vec<Stmt>> = None;
    let mut finally_body: Option<Vec<Stmt>> = None;
    let mut state = 0; // 0=try, 1=catch, 2=finally

    for part in p.into_inner() {
        match part.as_rule() {
            Rule::kw_try => state = 0,
            Rule::kw_catch => state = 1,
            Rule::kw_finally => state = 2,
            Rule::ident => catch_var = Some(part.as_str().to_string()),
            Rule::block => {
                let stmts = lower_block_stmts(part)?;
                match state {
                    0 => body = stmts,
                    1 => catch_body = Some(stmts),
                    _ => finally_body = Some(stmts),
                }
            }
            _ => {}
        }
    }
    Ok(Stmt::TryCatch { body, catch_var, catch_body, finally_body })
}

fn lower_machine(p: Pair<Rule>) -> Result<MachineDef, String> {
    // machine_decl = { kw_machine ~ ident ~ "{" ~ machine_initial ~ machine_transition* ~ "}" }
    let mut name = String::new();
    let mut initial = String::new();
    let mut transitions = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => { if name.is_empty() { name = part.as_str().to_string(); } }
            Rule::machine_initial => {
                // machine_initial = { kw_initial ~ ident }
                if let Some(id) = part.into_inner().find(|x| x.as_rule() == Rule::ident) {
                    initial = id.as_str().to_string();
                }
            }
            Rule::machine_transition => {
                // machine_transition = { ident ~ "->" ~ ident ~ kw_on ~ string_lit }
                let mut ids = Vec::new();
                let mut event = String::new();
                for x in part.into_inner() {
                    match x.as_rule() {
                        Rule::ident => ids.push(x.as_str().to_string()),
                        Rule::string_lit => {
                            let r = x.as_str();
                            event = r[1..r.len().saturating_sub(1)].to_string();
                        }
                        _ => {}
                    }
                }
                if ids.len() == 2 {
                    transitions.push((ids[0].clone(), ids[1].clone(), event));
                }
            }
            _ => {}
        }
    }
    Ok(MachineDef { name, initial, transitions })
}

fn lower_test(p: Pair<Rule>) -> Result<TestBlock, String> {
    // test_block = { kw_test ~ string_lit ~ block }
    let mut name = String::new();
    let mut body = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::string_lit => {
                let raw = part.as_str();
                name = raw[1..raw.len().saturating_sub(1)].to_string();
            }
            Rule::block => body = lower_block_stmts(part)?,
            _ => {}
        }
    }
    Ok(TestBlock { name, body })
}

fn lower_use(p: Pair<Rule>) -> Result<UseDecl, String> {
    // new grammar:
    //   use_decl  = { kw_use ~ use_tree ~ ";"? }
    //   use_tree  = { use_group | use_glob | use_named }
    //   use_named = { use_prefix ~ (kw_as ~ ident)? }
    //   use_glob  = { use_prefix ~ "." ~ "*" }
    //   use_group = { use_prefix ~ "." ~ "{" ~ use_tree (~ "," use_tree)* ~ "}" }
    let tree = p.into_inner().find(|x| x.as_rule() == Rule::use_tree)
        .ok_or("empty use declaration")?;
    let full = tree.as_str();
    let wildcard = full.contains(".*");

    // the module root is the first use_prefix found anywhere in the tree
    let prefix = find_rule(tree.clone(), Rule::use_prefix)
        .map(|x| x.as_str().trim().to_string())
        .unwrap_or_default();

    let mut names = Vec::new();
    let mut alias = None;

    // collect group leaf names and an optional alias
    let kind = tree.clone().into_inner().next();
    if let Some(k) = kind {
        match k.as_rule() {
            Rule::use_group => {
                for t in k.into_inner() {
                    if t.as_rule() == Rule::use_tree {
                        names.push(t.as_str().trim().to_string());
                    }
                }
            }
            Rule::use_named => {
                // alias: the ident after `as`
                let idents: Vec<_> = k.clone().into_inner()
                    .filter(|x| x.as_rule() == Rule::ident).collect();
                if let Some(a) = idents.last() {
                    alias = Some(a.as_str().to_string());
                }
            }
            _ => {}
        }
    }

    Ok(UseDecl { module: prefix, names, wildcard, alias })
}

fn lower_enum(p: Pair<Rule>) -> Result<EnumDef, String> {
    // enum_decl = { kw_enum ~ ident ~ generic_params? ~ where_clause? ~ "{" ~ enum_variant* ~ "}" }
    let mut name = String::new();
    let mut variants = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => {
                if name.is_empty() { name = part.as_str().to_string(); }
            }
            Rule::enum_variant => {
                // enum_variant = { attribute* ~ ident ~ variant_data? }
                let mut vname = String::new();
                let mut arity = 0;
                for x in part.into_inner() {
                    match x.as_rule() {
                        Rule::ident => vname = x.as_str().to_string(),
                        Rule::variant_data => {
                            // count tuple slots (type_expr children)
                            arity = x.into_inner().filter(|t| t.as_rule() == Rule::type_expr).count();
                        }
                        _ => {}
                    }
                }
                variants.push(VariantDef { name: vname, arity });
            }
            _ => {}
        }
    }
    Ok(EnumDef { name, variants })
}

fn lower_match(p: Pair<Rule>) -> Result<Expr, String> {
    // match_expr = { kw_match ~ cond ~ "{" ~ match_arm ~ ("," ~ match_arm)* ~ "}" }
    let mut scrutinee: Option<Expr> = None;
    let mut arms = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::cond => scrutinee = Some(lower_cond(part)?),
            Rule::match_arm => arms.push(lower_match_arm(part)?),
            _ => {}
        }
    }
    let scrutinee = scrutinee.ok_or("match without scrutinee")?;
    Ok(Expr::Match { scrutinee: Box::new(scrutinee), arms })
}

fn lower_match_arm(p: Pair<Rule>) -> Result<MatchArm, String> {
    // match_arm = { pattern ~ (kw_if ~ cond)? ~ "=>" ~ expr }
    let mut pattern: Option<Pattern> = None;
    let mut guard: Option<Expr> = None;
    let mut body: Option<Expr> = None;
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::pattern => pattern = Some(lower_pattern(part)?),
            Rule::cond => guard = Some(lower_cond(part)?),
            Rule::expr => body = Some(lower_expr(part)?),
            _ => {}
        }
    }
    Ok(MatchArm {
        pattern: pattern.ok_or("match arm without pattern")?,
        guard,
        body: body.ok_or("match arm without body")?,
    })
}

fn lower_pattern(p: Pair<Rule>) -> Result<Pattern, String> {
    // pattern = { pattern_or }
    let inner = p.into_inner().next().ok_or("empty pattern")?;
    lower_pattern_or(inner)
}

fn lower_pattern_or(p: Pair<Rule>) -> Result<Pattern, String> {
    // pattern_or = { pattern_range ~ ("|" ~ pattern_range)* }
    let alts: Vec<Pair<Rule>> = p.into_inner()
        .filter(|x| x.as_rule() == Rule::pattern_range).collect();
    if alts.len() == 1 {
        return lower_pattern_range(alts.into_iter().next().unwrap());
    }
    let mut pats = Vec::new();
    for a in alts {
        pats.push(lower_pattern_range(a)?);
    }
    Ok(Pattern::Or(pats))
}

fn lower_pattern_range(p: Pair<Rule>) -> Result<Pattern, String> {
    // pattern_range = { pattern_atom ~ (("..="|"..") ~ pattern_atom)? }
    let txt = p.as_str();
    let atoms: Vec<Pair<Rule>> = p.into_inner()
        .filter(|x| x.as_rule() == Rule::pattern_atom).collect();
    if atoms.len() == 2 {
        // numeric range pattern
        let lo = pattern_atom_int(&atoms[0])?;
        let hi = pattern_atom_int(&atoms[1])?;
        let inclusive = txt.contains("..=");
        return Ok(Pattern::Range { lo, hi, inclusive });
    }
    lower_pattern_atom(atoms.into_iter().next().ok_or("empty pattern range")?)
}

fn pattern_atom_int(p: &Pair<Rule>) -> Result<i64, String> {
    let txt = p.as_str().trim();
    let cleaned: String = txt.chars().filter(|c| *c != '_').collect();
    cleaned.parse::<i64>()
        .map_err(|_| format!("range pattern bound must be an integer, got: {}", txt))
}

fn lower_pattern_atom(p: Pair<Rule>) -> Result<Pattern, String> {
    // pattern_atom = { "_" | "-"? literal | tuple_pattern | slice_pattern
    //                | struct_pattern | enum_pattern | binding_pattern | "..." }
    let txt = p.as_str().trim();
    if txt == "_" {
        return Ok(Pattern::Wildcard);
    }
    let inner = match p.into_inner().next() {
        Some(i) => i,
        None => {
            // bare "_" or "..." matched as literal text
            if txt == "_" { return Ok(Pattern::Wildcard); }
            return Err(format!("unsupported pattern: {}", txt));
        }
    };
    match inner.as_rule() {
        Rule::literal => lower_literal_pattern(inner, txt.starts_with('-')),
        Rule::enum_pattern => lower_enum_pattern(inner),
        Rule::binding_pattern => {
            // binding_pattern = { kw_ref? ~ kw_mut? ~ ident ~ ("@" ~ pattern)? }
            let name = inner.into_inner()
                .find(|x| x.as_rule() == Rule::ident)
                .map(|x| x.as_str().to_string())
                .ok_or("binding pattern without name")?;
            // A bare capitalized identifier with no payload is a unit enum variant (e.g. None, Nil)
            if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                Ok(Pattern::EnumVariant { name, sub: vec![] })
            } else {
                Ok(Pattern::Binding(name))
            }
        }
        Rule::tuple_pattern => {
            let subs: Result<Vec<_>, _> = inner.into_inner()
                .filter(|x| x.as_rule() == Rule::pattern)
                .map(lower_pattern).collect();
            Ok(Pattern::Tuple(subs?))
        }
        Rule::struct_pattern => lower_struct_pattern(inner),
        Rule::slice_pattern => lower_slice_pattern(inner),
        other => Err(format!("pattern {:?} not yet in Nova Core", other)),
    }
}

fn lower_literal_pattern(p: Pair<Rule>, negative: bool) -> Result<Pattern, String> {
    let e = lower_literal(p)?;
    Ok(match e {
        Expr::Int(n) => Pattern::Int(if negative { -n } else { n }),
        Expr::Float(x) => Pattern::Float(if negative { -x } else { x }),
        Expr::Str(s) => Pattern::Str(s),
        Expr::Bool(b) => Pattern::Bool(b),
        Expr::Null => Pattern::Null,
        _ => return Err("unsupported literal pattern".into()),
    })
}

fn lower_enum_pattern(p: Pair<Rule>) -> Result<Pattern, String> {
    // enum_pattern = { type_path ~ "(" ~ (pattern ~ ("," ~ pattern)* ~ ","?)? ~ ")" }
    let mut name = String::new();
    let mut sub = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::type_path => {
                let txt = part.as_str().trim();
                name = txt.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            }
            Rule::pattern => sub.push(lower_pattern(part)?),
            _ => {}
        }
    }
    Ok(Pattern::EnumVariant { name, sub })
}

fn lower_struct_pattern(p: Pair<Rule>) -> Result<Pattern, String> {
    // struct_pattern = { type_path ~ "{" ~ (field_pattern ~ ...)? ~ "}" }
    // field_pattern  = { ident ~ (":" ~ pattern)? }
    let mut name = String::new();
    let mut fields = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::type_path => {
                name = part.as_str().trim().chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            }
            Rule::field_pattern => {
                let mut fname = String::new();
                let mut fpat: Option<Pattern> = None;
                for fp in part.into_inner() {
                    match fp.as_rule() {
                        Rule::ident if fname.is_empty() => fname = fp.as_str().to_string(),
                        Rule::pattern => fpat = Some(lower_pattern(fp)?),
                        _ => {}
                    }
                }
                // shorthand `{ x }` binds field x to a variable named x
                let pat = fpat.unwrap_or_else(|| Pattern::Binding(fname.clone()));
                fields.push((fname, pat));
            }
            _ => {}
        }
    }
    Ok(Pattern::Struct { name, fields })
}

fn lower_const(p: Pair<Rule>) -> Result<Item, String> {
    // const_decl  = { kw_const  ~ ident ~ (":" ~ type_expr)? ~ "=" ~ expr ~ ";" }
    // static_decl = { kw_static ~ kw_mut? ~ ident ~ (":" ~ type_expr)? ~ "=" ~ expr ~ ";" }
    // Both lower to a global binding; the leading `mut`/keyword tokens are skipped.
    let mut name = String::new();
    let mut value = None;
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => { if name.is_empty() { name = part.as_str().to_string(); } }
            Rule::expr => value = Some(lower_expr(part)?),
            _ => {}
        }
    }
    Ok(Item::Const { name, value: value.ok_or("const without value")? })
}

fn lower_type_alias(p: Pair<Rule>) -> Result<Item, String> {
    // type_decl = { kw_type ~ ident ~ generic_params? ~ where_clause?
    //               ~ "=" ~ type_expr ~ ("schema" ...)? ~ ";" }
    // We keep the alias's head target name; the type checker resolves it.
    let mut name = String::new();
    let mut target = String::new();
    let mut refinement: Option<Expr> = None;
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident if name.is_empty() => name = part.as_str().to_string(),
            Rule::type_expr => {
                // take the leading type name, e.g. "Int" from "Int" or "Point"
                target = part.as_str().trim()
                    .chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                // a refinement `Int if <pred>` carries a predicate over the value `it`
                if let Some(rt) = find_rule(part.clone(), Rule::refinement_type) {
                    if let Some(pred) = rt.into_inner().find(|x| x.as_rule() == Rule::expr) {
                        refinement = Some(lower_expr(pred)?);
                    }
                }
            }
            _ => {}
        }
    }
    if name.is_empty() { return Err("type alias without a name".into()); }
    Ok(Item::TypeAlias { name, target, refinement })
}

fn lower_extern(p: Pair<Rule>) -> Result<Item, String> {
    // extern_block = { kw_extern ~ string_lit? ~ "{" ~ extern_item* ~ "}" }
    // extern_item  = { attribute* ~ kw_fn ~ ident ~ "(" ~ (extern_arg ~ ...)? ~ ")" ~ ret_type? ~ ";" }
    // extern_arg   = _{ variadic | param }
    let mut funcs = Vec::new();
    for item in p.into_inner() {
        if item.as_rule() != Rule::extern_item { continue; }
        let mut name = String::new();
        let mut arity = 0;
        let mut variadic = false;
        for x in item.into_inner() {
            match x.as_rule() {
                Rule::ident if name.is_empty() => name = x.as_str().to_string(),
                Rule::param => arity += 1,
                Rule::variadic => variadic = true,
                _ => {}
            }
        }
        if !name.is_empty() {
            funcs.push(ExternFn { name, arity, variadic });
        }
    }
    Ok(Item::Extern(funcs))
}

fn lower_slice_pattern(p: Pair<Rule>) -> Result<Pattern, String> {
    // slice_pattern = { "[" ~ (slice_elem ~ ...)? ~ "]" }
    // slice_elem    = _{ rest_pattern | pattern }
    // rest_pattern  = { "..." ~ ident? }
    // At most one rest: elements before it form the prefix, after it the suffix.
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    let mut rest: Option<Option<String>> = None;
    for el in p.into_inner() {
        match el.as_rule() {
            Rule::rest_pattern => {
                if rest.is_some() {
                    return Err("a slice pattern may have at most one `...` rest".into());
                }
                let bind = el.into_inner()
                    .find(|x| x.as_rule() == Rule::ident)
                    .map(|x| x.as_str().to_string());
                rest = Some(bind);
            }
            Rule::pattern => {
                let pat = lower_pattern(el)?;
                if rest.is_none() { prefix.push(pat); } else { suffix.push(pat); }
            }
            _ => {}
        }
    }
    Ok(Pattern::Slice { prefix, rest, suffix })
}

thread_local! {
    static HYG_COUNTER: std::cell::Cell<u64> = std::cell::Cell::new(0);
}

// Desugar `src ->> sink` (stream every element of `src` into channel `sink`) into
// `{ let s = sink; for x in src { s <- x }; s }`, evaluating each side once and
// yielding the channel. Fresh hygienic names avoid clashes with user variables.
fn build_stream_into(src: Expr, sink: Expr) -> Expr {
    let n = HYG_COUNTER.with(|c| { let v = c.get(); c.set(v + 1); v });
    let s = format!("__strm_sink_{}", n);
    let x = format!("__strm_x_{}", n);
    Expr::Block {
        stmts: vec![
            Stmt::Let { name: s.clone(), ty: None, value: sink },
            Stmt::ForEach {
                var: x.clone(),
                iter: src,
                body: vec![Stmt::Expr(Expr::Send {
                    chan: Box::new(Expr::Ident(s.clone())),
                    value: Box::new(Expr::Ident(x)),
                })],
            },
        ],
        tail: Some(Box::new(Expr::Ident(s))),
    }
}

// Make a macro body hygienic: rename every identifier introduced by a `let`
// (or `let mut`) inside the body — except macro parameters — to a unique
// `name_hygN`, consistently. String literals are skipped, and matching is
// identifier-aware (whole words only), so substitution is safe.
fn hygienic_macro_body(body: &str, params: &[String]) -> String {
    let bytes = body.as_bytes();
    let mut idents: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    let mut in_str = false;
    let mut esc = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if in_str {
            if esc { esc = false; }
            else if c == '\\' { esc = true; }
            else if c == '"' { in_str = false; }
            i += 1;
            continue;
        }
        if c == '"' { in_str = true; i += 1; continue; }
        if c.is_ascii_alphabetic() || c == '_' {
            let s = i;
            while i < bytes.len() {
                let d = bytes[i] as char;
                if d.is_ascii_alphanumeric() || d == '_' { i += 1; } else { break; }
            }
            idents.push((s, i));
        } else {
            i += 1;
        }
    }
    // collect `let [mut] NAME` binding names (excluding macro params)
    let mut locals: Vec<String> = Vec::new();
    for k in 0..idents.len() {
        if &body[idents[k].0..idents[k].1] == "let" {
            let mut m = k + 1;
            if m < idents.len() && &body[idents[m].0..idents[m].1] == "mut" { m += 1; }
            if m < idents.len() {
                let nm = body[idents[m].0..idents[m].1].to_string();
                if !params.iter().any(|p| p == &nm) && !locals.contains(&nm) {
                    locals.push(nm);
                }
            }
        }
    }
    if locals.is_empty() { return body.to_string(); }
    let n = HYG_COUNTER.with(|c| { let v = c.get(); c.set(v + 1); v });
    let mut out = String::new();
    let mut last = 0;
    for (s, e) in &idents {
        let name = &body[*s..*e];
        if locals.iter().any(|l| l == name) {
            out.push_str(&body[last..*s]);
            out.push_str(&format!("{}_hyg{}", name, n));
            last = *e;
        }
    }
    out.push_str(&body[last..]);
    out
}

fn lower_macro_call(p: Pair<Rule>) -> Result<Expr, String> {
    // macro_call = { expr_path ~ "!" ~ !"=" ~ token_tree }
    // Expand the macro by substituting each argument for its $param in the body,
    // then parse the resulting source as an expression.
    let mut name = String::new();
    let mut arg_text = String::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::expr_path => name = part.as_str().to_string(),
            Rule::token_tree => arg_text = part.as_str().to_string(),
            _ => {}
        }
    }
    let (params, body) = MACROS.with(|m| m.borrow().get(&name).cloned())
        .ok_or_else(|| format!("unknown macro: {}!", name))?;

    // split the call arguments: strip outer ( ) / { } and split on top-level commas
    let inner = {
        let t = arg_text.trim();
        if t.len() >= 2 && (t.starts_with('(') || t.starts_with('{') || t.starts_with('[')) {
            &t[1..t.len()-1]
        } else { t }
    };
    let args = split_top_level_commas(inner);
    if args.len() != params.len() {
        return Err(format!(
            "macro {}! expects {} argument(s), got {}",
            name, params.len(), args.len()
        ));
    }

    // hygiene: rename the macro body's own `let` bindings to unique names so they
    // can't capture or be captured by identifiers at the call site.
    let body = hygienic_macro_body(&body, &params);

    // substitute $param -> (arg) in the body template. Parenthesising the
    // argument preserves precedence: square!(3+1) becomes (3+1)*(3+1), not 3+1*3+1.
    let mut expanded = body.clone();
    for (param, arg) in params.iter().zip(args.iter()) {
        expanded = expanded.replace(&format!("${}", param), &format!("({})", arg.trim()));
    }

    // parse the expanded text as a Nova expression (without clearing macros)
    let wrapped = format!("fn __m__() {{ {} }}", expanded);
    let prog = parse_program_inner(&wrapped)
        .map_err(|e| format!("macro {}! expansion failed to parse: {}", name, e))?;
    if let Some(Item::Func(f)) = prog.items.into_iter().next() {
        let mut body = f.body;
        // the trailing implicit return holds the expanded value; a multi-statement
        // body (e.g. `let t = ..; t + t`) becomes a block-valued expression.
        if let Some(Stmt::Return(Some(e))) = body.pop() {
            if body.is_empty() {
                return Ok(e);
            }
            return Ok(Expr::Block { stmts: body, tail: Some(Box::new(e)) });
        }
    }
    Err(format!("macro {}! did not expand to an expression", name))
}

// Split a string on commas that are not nested inside (), [], {}, or strings.
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut in_str = false;
    let mut prev = ' ';
    for c in s.chars() {
        match c {
            '"' if prev != '\\' => { in_str = !in_str; cur.push(c); }
            '(' | '[' | '{' if !in_str => { depth += 1; cur.push(c); }
            ')' | ']' | '}' if !in_str => { depth -= 1; cur.push(c); }
            ',' if depth == 0 && !in_str => { out.push(cur.trim().to_string()); cur.clear(); }
            _ => cur.push(c),
        }
        prev = c;
    }
    if !cur.trim().is_empty() { out.push(cur.trim().to_string()); }
    out
}

fn lower_macro_decl(p: Pair<Rule>) -> Result<MacroDef, String> {
    // macro_decl    = { kw_macro ~ ident ~ "{" ~ macro_rule ~ ... ~ "}" }
    // macro_rule    = { macro_matcher ~ "=>" ~ macro_body }
    // macro_matcher = { "(" ~ macro_token* ~ ")" }
    // macro_body    = { "{" ~ macro_token* ~ "}" | "(" ~ macro_token* ~ ")" }
    let mut name = String::new();
    let mut params = Vec::new();
    let mut body = String::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => { if name.is_empty() { name = part.as_str().to_string(); } }
            Rule::macro_rule => {
                for r in part.into_inner() {
                    match r.as_rule() {
                        Rule::macro_matcher => {
                            // collect $name placeholders in declaration order
                            let txt = r.as_str();
                            params = extract_macro_params(txt);
                        }
                        Rule::macro_body => {
                            // strip the outer { } or ( ) delimiters, keep inner text
                            let raw = r.as_str().trim();
                            let inner = &raw[1..raw.len().saturating_sub(1)];
                            body = inner.trim().to_string();
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if name.is_empty() { return Err("macro without a name".into()); }
    Ok(MacroDef { name, params, body })
}

// Pull `$ident` placeholder names out of a matcher's source text, in order.
fn extract_macro_params(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '$' {
            let mut j = i + 1;
            let mut name = String::new();
            while j < bytes.len() && (bytes[j].is_alphanumeric() || bytes[j] == '_') {
                name.push(bytes[j]);
                j += 1;
            }
            if !name.is_empty() && !out.contains(&name) { out.push(name); }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

fn lower_trait(p: Pair<Rule>) -> Result<TraitDef, String> {
    // trait_decl = { kw_trait ~ ident ~ ... ~ "{" ~ trait_item* ~ "}" }
    // trait_item = { attribute* ~ (method_sig ~ (block | ";") | kw_const ...) }
    let mut name = String::new();
    let mut required = Vec::new();
    let mut defaults = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => { if name.is_empty() { name = part.as_str().to_string(); } }
            Rule::trait_item => {
                let mut method_name = String::new();
                let mut params: Vec<String> = Vec::new();
                let mut body: Option<Vec<Stmt>> = None;
                for x in part.into_inner() {
                    match x.as_rule() {
                        Rule::method_sig => {
                            for y in x.into_inner() {
                                match y.as_rule() {
                                    Rule::ident => { if method_name.is_empty() { method_name = y.as_str().to_string(); } }
                                    Rule::params => params = lower_params(y)?,
                                    _ => {}
                                }
                            }
                        }
                        Rule::block => body = Some(lower_block_stmts(x)?),
                        _ => {}
                    }
                }
                if method_name.is_empty() { continue; }
                match body {
                    Some(mut b) => {
                        make_trailing_implicit_return(&mut b);
                        defaults.push(Func {
                            name: method_name, params,
                            param_types: Vec::new(), param_modes: Vec::new(),
                            ret_type: None, type_params: Vec::new(),
                            where_bounds: Vec::new(), effects: None, body: b, is_async: false,
                        });
                    }
                    None => required.push(method_name),
                }
            }
            _ => {}
        }
    }
    Ok(TraitDef { name, required, defaults })
}

fn lower_data(p: Pair<Rule>) -> Result<StructDef, String> {
    // data_decl = { kw_data ~ ident ~ generic_params? ~ "(" ~ data_fields? ~ ")" ~ ";"? }
    // data_field = { kw_mut? ~ ident ~ (":" ~ type_expr)? ~ ("=" ~ expr)? }
    let mut name = String::new();
    let mut fields = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => { if name.is_empty() { name = part.as_str().to_string(); } }
            Rule::data_fields => {
                for f in part.into_inner() {
                    if f.as_rule() == Rule::data_field {
                        for x in f.into_inner() {
                            if x.as_rule() == Rule::ident {
                                fields.push(x.as_str().to_string());
                                break;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(StructDef { name, fields })
}

fn lower_struct(p: Pair<Rule>) -> Result<StructDef, String> {
    // struct_decl = { kw_struct ~ ident ~ generic_params? ~ where_clause? ~ struct_body }
    let mut name = String::new();
    let mut fields = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => name = part.as_str().to_string(),
            Rule::struct_body => {
                // struct_body has named_field/struct_field children
                for f in part.into_inner() {
                    if f.as_rule() == Rule::struct_field {
                        // struct_field = { attribute* ~ visibility? ~ kw_mut? ~ ident ~ ":" ~ type_expr }
                        for x in f.into_inner() {
                            if x.as_rule() == Rule::ident {
                                fields.push(x.as_str().to_string());
                                break;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(StructDef { name, fields })
}

fn lower_impl(p: Pair<Rule>) -> Result<ImplBlock, String> {
    // impl_block = { kw_impl ~ generic_params? ~ type_expr ~ (kw_for ~ type_expr)? ~ where_clause? ~ "{" ~ impl_item* ~ "}" }
    // With two type_exprs it is `impl Trait for Type` (first=trait, second=type).
    // With one it is `impl Type`.
    let mut type_exprs: Vec<String> = Vec::new();
    let mut methods = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::type_expr => {
                let txt = part.as_str().trim();
                let base: String = txt.chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                type_exprs.push(base);
            }
            Rule::impl_item => {
                for ii in part.into_inner() {
                    if ii.as_rule() == Rule::func_decl {
                        methods.push(lower_func(ii)?);
                    }
                }
            }
            _ => {}
        }
    }
    let (trait_name, type_name) = match type_exprs.len() {
        0 => return Err("impl block without a type name".into()),
        1 => (None, type_exprs.remove(0)),
        _ => {
            let t = type_exprs.remove(0); // trait
            let ty = type_exprs.remove(0); // concrete type
            (Some(t), ty)
        }
    };
    Ok(ImplBlock { type_name, trait_name, methods })
}

fn lower_func(p: Pair<Rule>) -> Result<Func, String> {
    // func_decl = { kw_async? ~ kw_fn ~ ident ~ generic_params? ~ "(" ~ params? ~ ")"
    //              ~ ret_type? ~ where_clause? ~ (block | "=>" ~ expr ~ ";") }
    let mut name = String::new();
    let mut params = Vec::new();
    let mut param_types = Vec::new();
    let mut param_modes: Vec<Option<String>> = Vec::new();
    let mut ret_type: Option<String> = None;
    let mut type_params: Vec<String> = Vec::new();
    let mut body: Vec<Stmt> = Vec::new();
    let mut arrow_expr: Option<Expr> = None;
    let mut is_async = false;
    let mut where_bounds: Vec<(String, Vec<String>)> = Vec::new();
    let mut effects: Option<Vec<String>> = None;

    for part in p.into_inner() {
        match part.as_rule() {
            Rule::ident => name = part.as_str().to_string(),
            Rule::params => {
                let (ns, ts, ms) = lower_params_typed(part)?;
                params = ns;
                param_types = ts;
                param_modes = ms;
            }
            Rule::generic_params => {
                let (names, inline) = lower_generic_params(part);
                type_params = names;
                where_bounds.extend(inline);
            }
            Rule::ret_type => ret_type = Some(type_head(part.as_str())),
            Rule::effects_clause => effects = Some(lower_effects(part)),
            Rule::where_clause => where_bounds.extend(where_clause_bounds(part)),
            Rule::block => body = lower_block_stmts(part)?,
            Rule::expr => arrow_expr = Some(lower_expr(part)?), // arrow-body fn
            Rule::kw_async => is_async = true,
            Rule::kw_fn => {}
            _ => {}
        }
    }

    if let Some(e) = arrow_expr {
        body = vec![Stmt::Return(Some(e))];
    } else {
        make_trailing_implicit_return(&mut body);
    }
    Ok(Func { name, params, param_types, param_modes, ret_type, type_params, where_bounds, effects, body, is_async })
}

// Collect the effect head names from an effects clause: `![IO, Net]` -> ["IO","Net"].
// effects_clause = { "!" ~ "[" ~ effect ~ ("," ~ effect)* ~ "]" }; effect = { type_path | lifetime }
fn lower_effects(p: Pair<Rule>) -> Vec<String> {
    let mut out = Vec::new();
    for e in p.into_inner() {
        if e.as_rule() == Rule::effect {
            let name = type_head(e.as_str());
            if !name.is_empty() { out.push(name); }
        }
    }
    out
}

// Extract generic parameter names and any inline trait bounds (`[T: Trait + Other]`).
// generic_params = { "[" ~ generic_param ~ ... }
// generic_param  = { ident ~ ("[_]")? ~ (":" ~ bounds)? ~ ("=" ~ type_expr)? }
fn lower_generic_params(p: Pair<Rule>) -> (Vec<String>, Vec<(String, Vec<String>)>) {
    let mut names = Vec::new();
    let mut bounds = Vec::new();
    for gp in p.into_inner() {
        if gp.as_rule() == Rule::generic_param {
            let mut nm = String::new();
            let mut traits = Vec::new();
            for x in gp.into_inner() {
                match x.as_rule() {
                    Rule::ident if nm.is_empty() => nm = x.as_str().to_string(),
                    Rule::bounds => traits = bound_trait_names(x),
                    _ => {}
                }
            }
            if !nm.is_empty() {
                if !traits.is_empty() { bounds.push((nm.clone(), traits)); }
                names.push(nm);
            }
        }
    }
    (names, bounds)
}

// Collect (subject, [trait names]) pairs from a `where` clause.
// where_clause = { kw_where ~ where_item ~ ... }; where_item = { type_expr ~ ":" ~ bounds }
fn where_clause_bounds(p: Pair<Rule>) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for wi in p.into_inner() {
        if wi.as_rule() == Rule::where_item {
            let mut subj = String::new();
            let mut traits = Vec::new();
            for x in wi.into_inner() {
                match x.as_rule() {
                    Rule::type_expr if subj.is_empty() => subj = type_head(x.as_str()),
                    Rule::bounds => traits = bound_trait_names(x),
                    _ => {}
                }
            }
            if !subj.is_empty() && !traits.is_empty() { out.push((subj, traits)); }
        }
    }
    out
}

// Trait names from a `bounds` pair: bounds = { bound ~ ("+" ~ bound)* }, bound = type_path | lifetime
fn bound_trait_names(bounds: Pair<Rule>) -> Vec<String> {
    let mut out = Vec::new();
    for b in bounds.into_inner() {
        if b.as_rule() == Rule::bound {
            if let Some(tp) = b.into_inner().find(|x| x.as_rule() == Rule::type_path) {
                let name = type_head(tp.as_str());
                if !name.is_empty() { out.push(name); }
            }
        }
    }
    out
}

// Reduce a type annotation to its leading identifier ("Int", "Array[Int]" -> "Array",
// "-> Str" -> "Str"). Good enough for the gradual checker.
fn type_head(s: &str) -> String {
    s.trim_start_matches("->").trim()
        .chars().skip_while(|c| !c.is_alphanumeric() && *c != '_')
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

fn lower_params(p: Pair<Rule>) -> Result<Vec<String>, String> {
    Ok(lower_params_typed(p)?.0)
}

// Returns (names, optional declared type heads) parallel to each other.
fn lower_params_typed(p: Pair<Rule>)
    -> Result<(Vec<String>, Vec<Option<String>>, Vec<Option<String>>), String>
{
    let mut names = Vec::new();
    let mut types = Vec::new();
    let mut modes = Vec::new();
    for param in p.into_inner() {
        if param.as_rule() == Rule::param {
            // param = { self_param | kw_mut? ~ kw_ref? ~ ident ~ (":" ~ type_expr)? ~ ... }
            let mut name: Option<String> = None;
            let mut ty: Option<String> = None;
            let mut mode: Option<String> = None;
            for x in param.into_inner() {
                match x.as_rule() {
                    Rule::self_param => { name = Some("self".to_string()); }
                    Rule::ident => { if name.is_none() { name = Some(x.as_str().to_string()); } }
                    Rule::type_expr => {
                        let txt = x.as_str().trim();
                        // a `linear`/`affine`/`shared` modifier precedes the type head
                        if let Some(m) = find_rule(x.clone(), Rule::type_modifier) {
                            let word = m.as_str().trim();
                            if word == "linear" || word == "affine" {
                                mode = Some(word.to_string());
                            }
                            let after = txt.strip_prefix(word).unwrap_or(txt).trim_start();
                            ty = Some(type_head(after));
                        } else {
                            ty = Some(type_head(txt));
                        }
                    }
                    _ => {}
                }
            }
            if let Some(n) = name {
                names.push(n);
                types.push(ty);
                modes.push(mode);
            }
        }
    }
    Ok((names, types, modes))
}

fn lower_block_stmts(p: Pair<Rule>) -> Result<Vec<Stmt>, String> {
    // block = { "{" ~ stmt* ~ expr? ~ "}" }
    let mut stmts = Vec::new();
    for inner in p.into_inner() {
        match inner.as_rule() {
            Rule::stmt => stmts.push(lower_stmt(inner)?),
            Rule::expr => {
                // trailing expr without semicolon — treat as implicit return value
                let e = lower_expr(inner)?;
                stmts.push(Stmt::Return(Some(e)));
            }
            _ => {}
        }
    }
    Ok(stmts)
}

// Turn a trailing bare expression statement into an implicit return value.
// This gives Rust/Ruby-style "last expression is the result" semantics.
// Also recurses into trailing if/else branches so `fn f() { if c { a } else { b } }`
// returns a or b.
fn make_trailing_implicit_return(stmts: &mut Vec<Stmt>) {
    match stmts.last_mut() {
        Some(Stmt::Expr(_)) => {
            if let Some(Stmt::Expr(e)) = stmts.pop() {
                stmts.push(Stmt::Return(Some(e)));
            }
        }
        Some(Stmt::If { then, els, .. }) => {
            make_trailing_implicit_return(then);
            if let Some(els) = els {
                make_trailing_implicit_return(els);
            }
        }
        _ => {}
    }
}

fn lower_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    // stmt = { let_stmt | item_stmt | expr_stmt }
    let (line, col) = p.as_span().start_pos().line_col();
    let pos = (line as u32, col as u32);
    let inner = p.into_inner().next().ok_or("empty stmt")?;
    let stmt = match inner.as_rule() {
        Rule::let_stmt => lower_let(inner),
        Rule::yield_stmt => {
            // yield_stmt = { kw_yield ~ expr? ~ ";"? }
            let e = inner.into_inner().find(|x| x.as_rule() == Rule::expr)
                .map(lower_expr).transpose()?;
            Ok(Stmt::Yield(e))
        }
        Rule::expr_stmt => lower_expr_stmt(inner),
        Rule::defer_stmt => lower_defer(inner),
        Rule::item_stmt => Err("nested items not yet supported in Nova Core".into()),
        other => Err(format!("Unsupported statement: {:?}", other)),
    };
    // annotate lowering errors and the statement's value expressions with the line
    match stmt {
        Ok(s) => Ok(attach_pos(s, pos)),
        Err(e) => Err(format!("line {}: {}", line, e)),
    }
}

// Wrap the value-producing expressions of a statement with `Expr::At` so the
// interpreter/type checker can report a source position. Nested statement blocks
// are positioned by their own `lower_stmt`, so we only touch immediate expressions.
fn attach_pos(stmt: Stmt, pos: (u32, u32)) -> Stmt {
    let at = |e: Expr| Expr::At { pos, expr: Box::new(e) };
    match stmt {
        Stmt::Expr(e) => Stmt::Expr(at(e)),
        Stmt::Return(Some(e)) => Stmt::Return(Some(at(e))),
        Stmt::Throw(e) => Stmt::Throw(at(e)),
        Stmt::Yield(Some(e)) => Stmt::Yield(Some(at(e))),
        Stmt::Let { name, ty, value } => Stmt::Let { name, ty, value: at(value) },
        Stmt::Assign { name, value } => Stmt::Assign { name, value: at(value) },
        Stmt::IndexAssign { base, index, value } =>
            Stmt::IndexAssign { base, index, value: at(value) },
        Stmt::FieldAssign { base, field, value } =>
            Stmt::FieldAssign { base, field, value: at(value) },
        Stmt::While { cond, body } => Stmt::While { cond: at(cond), body },
        Stmt::If { cond, then, els } => Stmt::If { cond: at(cond), then, els },
        Stmt::ForRange { var, start, end, inclusive, body } =>
            Stmt::ForRange { var, start: at(start), end: at(end), inclusive, body },
        Stmt::ForEach { var, iter, body } => Stmt::ForEach { var, iter: at(iter), body },
        Stmt::Break(Some(e)) => Stmt::Break(Some(at(e))),
        other => other, // Return(None), Break(None), Continue, Defer, TryCatch
    }
}

fn lower_defer_expr(p: Pair<Rule>) -> Result<Stmt, String> {
    // defer_expr = { kw_defer ~ (block | expr) }
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::block => return Ok(Stmt::Defer(lower_block_stmts(part)?)),
            Rule::expr => return Ok(Stmt::Defer(vec![Stmt::Expr(lower_expr(part)?)])),
            _ => {}
        }
    }
    Err("empty defer".into())
}

fn lower_defer(p: Pair<Rule>) -> Result<Stmt, String> {
    // defer_stmt = { kw_defer ~ (block | expr) ~ ";"? }
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::block => return Ok(Stmt::Defer(lower_block_stmts(part)?)),
            Rule::expr => return Ok(Stmt::Defer(vec![Stmt::Expr(lower_expr(part)?)])),
            _ => {}
        }
    }
    Err("empty defer".into())
}

fn lower_let(p: Pair<Rule>) -> Result<Stmt, String> {
    // let_stmt = { kw_let ~ kw_mut? ~ pattern ~ (":" ~ type_expr)? ~ ("=" ~ expr)? ~ ";" }
    let mut name = String::new();
    let mut ty: Option<String> = None;
    let mut value: Option<Expr> = None;
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::pattern => name = simple_pattern_name(part)?,
            Rule::type_expr => ty = Some(type_head(part.as_str())),
            Rule::expr => value = Some(lower_expr(part)?),
            _ => {}
        }
    }
    let value = value.unwrap_or(Expr::Null);
    Ok(Stmt::Let { name, ty, value })
}

fn simple_pattern_name(p: Pair<Rule>) -> Result<String, String> {
    // dig down pattern -> ... -> binding_pattern -> ident
    let s = p.as_str().trim();
    // For Nova Core we only allow a plain identifier pattern.
    if s.chars().all(|c| c.is_alphanumeric() || c == '_') && !s.is_empty() {
        Ok(s.to_string())
    } else {
        Err(format!("Only simple identifier patterns supported in Nova Core, got: {}", s))
    }
}

fn lower_expr_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    // expr_stmt = { expr_with_block ~ ";"? | expr ~ ";" }
    let inner = p.into_inner().next().ok_or("empty expr_stmt")?;
    match inner.as_rule() {
        Rule::expr_with_block => lower_expr_with_block_stmt(inner),
        Rule::expr => {
            // a bare `defer ...` parses as an expression via primary; route it to defer
            if let Some(d) = find_rule(inner.clone(), Rule::defer_expr) {
                return lower_defer_expr(d);
            }
            lower_expr_as_stmt(inner)
        }
        other => Err(format!("Unsupported expr_stmt: {:?}", other)),
    }
}

fn lower_expr_with_block_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    let inner = p.into_inner().next().ok_or("empty expr_with_block")?;
    match inner.as_rule() {
        Rule::if_expr => lower_if_stmt(inner),
        Rule::while_expr => lower_while_stmt(inner),
        Rule::loop_expr => lower_loop_stmt(inner),
        Rule::for_expr => lower_for_stmt(inner),
        Rule::match_expr => Ok(Stmt::Expr(lower_match(inner)?)),
        Rule::try_catch_expr => lower_try_catch(inner),
        Rule::block_expr => {
            // a bare `{ ... }` statement is a scoped block expression whose value
            // is discarded; lowering it to `Stmt::Expr(Block)` (rather than a
            // synthetic `if true`) is what the formatter round-trips to.
            let blk = inner.into_inner().next().unwrap(); // block
            Ok(Stmt::Expr(lower_block_value(blk)?))
        }
        other => Err(format!("Unsupported block-expr statement: {:?}", other)),
    }
}

fn lower_if_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    // if_expr = { kw_if ~ cond ~ block ~ (kw_else ~ (if_expr | block))? }
    let mut cond: Option<Expr> = None;
    let mut then: Vec<Stmt> = Vec::new();
    let mut els: Option<Vec<Stmt>> = None;
    let mut seen_block = false;
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::cond => cond = Some(lower_cond(part)?),
            Rule::block => {
                if !seen_block {
                    then = lower_block_stmts(part)?;
                    seen_block = true;
                } else {
                    els = Some(lower_block_stmts(part)?);
                }
            }
            Rule::if_expr => {
                els = Some(vec![lower_if_stmt(part)?]);
            }
            _ => {}
        }
    }
    Ok(Stmt::If { cond: cond.ok_or("if without condition")?, then, els })
}

fn lower_loop_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    // loop { ... } desugars to while true { ... } — break exits, continue repeats
    let mut body: Vec<Stmt> = Vec::new();
    for part in p.into_inner() {
        if part.as_rule() == Rule::block {
            body = lower_block_stmts(part)?;
        }
    }
    Ok(Stmt::While { cond: Expr::Bool(true), body })
}

fn lower_while_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    let mut cond: Option<Expr> = None;
    let mut body: Vec<Stmt> = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::cond => cond = Some(lower_cond(part)?),
            Rule::block => body = lower_block_stmts(part)?,
            _ => {}
        }
    }
    Ok(Stmt::While { cond: cond.ok_or("while without condition")?, body })
}

fn lower_for_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    // for_expr = { label? ~ kw_for ~ pattern ~ kw_in ~ cond ~ block }
    let mut var = String::new();
    let mut iter_cond: Option<Pair<Rule>> = None;
    let mut body: Vec<Stmt> = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::pattern => var = simple_pattern_name(part)?,
            Rule::cond => iter_cond = Some(part),
            Rule::block => body = lower_block_stmts(part)?,
            _ => {}
        }
    }
    let cond = iter_cond.ok_or("for without iterable")?;
    // Try `for x in a..b` first; otherwise treat as for-each over a collection.
    if let Some(rng) = find_rule(cond.clone(), Rule::ns_range_expr) {
        // ns_range_expr children: count ns_or_expr sides
        let sides: Vec<_> = rng.clone().into_inner()
            .filter(|x| x.as_rule() == Rule::ns_or_expr).collect();
        if sides.len() == 2 {
            let inclusive = rng.as_str().contains("..=");
            let mut vals = Vec::new();
            for s in sides { vals.push(lower_ns_or(s)?); }
            let end = vals.pop().unwrap();
            let start = vals.pop().unwrap();
            return Ok(Stmt::ForRange { var, start, end, inclusive, body });
        }
    }
    // for-each over an iterable expression (array)
    let iter = lower_cond(cond)?;
    Ok(Stmt::ForEach { var, iter, body })
}

fn find_rule(p: Pair<Rule>, target: Rule) -> Option<Pair<Rule>> {
    if p.as_rule() == target {
        return Some(p);
    }
    for inner in p.into_inner() {
        if let Some(found) = find_rule(inner, target) {
            return Some(found);
        }
    }
    None
}

fn lower_expr_as_stmt(p: Pair<Rule>) -> Result<Stmt, String> {
    // new grammar: expr = { stream_expr }
    //   stream_expr = { assign_expr ~ (stream_op ~ assign_expr)* }
    //   assign_expr = { pipeline_expr ~ (assign_op ~ assign_expr)? }
    let stream = p.into_inner().next().ok_or("empty expr")?; // stream_expr
    // If the statement is a channel send (`ch <- v`), the stream layer has a
    // stream_op. Detect it before descending into assign_expr.
    {
        // any stream operator (`<-` send, `->>` stream-into) makes this a stream
        // expression statement; lower the whole chain via the expression fold.
        let mut sinner = stream.clone().into_inner();
        let _lhs = sinner.next();
        if let Some(op) = sinner.next() {
            if op.as_rule() == Rule::stream_op {
                return Ok(Stmt::Expr(lower_stream_expr(stream)?));
            }
        }
    }
    let assign = stream.into_inner().next().ok_or("empty stream_expr")?; // assign_expr
    let mut inner = assign.clone().into_inner();
    let first = inner.next().ok_or("empty assign_expr")?; // pipeline_expr
    if let Some(op) = inner.next() {
        if op.as_rule() == Rule::assign_op {
            let op_str = op.as_str().trim();
            let rhs = inner.next().ok_or("assignment missing rhs")?;
            let mut value = lower_assign_expr(rhs)?;
            let lhs_expr = lower_pipeline_expr(first.clone())?;
            // desugar  x += e  into  x = x + e
            if op_str != "=" {
                let bin = match op_str {
                    "+=" => BinOp::Add, "-=" => BinOp::Sub, "*=" => BinOp::Mul,
                    "/=" => BinOp::Div, "%=" => BinOp::Rem,
                    "&=" => BinOp::BitAnd, "|=" => BinOp::BitOr, "^=" => BinOp::BitXor,
                    "<<=" => BinOp::Shl, ">>=" => BinOp::Shr,
                    other => return Err(format!("Compound assignment {} not yet in Nova Core", other)),
                };
                value = Expr::Binary { op: bin, lhs: Box::new(lhs_expr.clone()), rhs: Box::new(value) };
            }
            match lhs_expr {
                Expr::Ident(name) => return Ok(Stmt::Assign { name, value }),
                Expr::Index { base, index } => return Ok(Stmt::IndexAssign { base: *base, index: *index, value }),
                Expr::Field { base, field } => return Ok(Stmt::FieldAssign { base: *base, field, value }),
                _ => return Err("invalid assignment target (Nova Core supports `x = ...`, `a[i] = ...`, `obj.field = ...`)".into()),
            }
        }
    }
    // Not an assignment — check for return/flow inside.
    if let Some(ret) = find_rule(first.clone(), Rule::return_expr) {
        let mut e = None;
        for x in ret.into_inner() {
            if x.as_rule() == Rule::expr {
                e = Some(lower_expr(x)?);
            }
        }
        return Ok(Stmt::Return(e));
    }
    if let Some(thr) = find_rule(first.clone(), Rule::throw_expr) {
        let e = thr.into_inner().find(|x| x.as_rule() == Rule::expr)
            .ok_or("throw without value")?;
        return Ok(Stmt::Throw(lower_expr(e)?));
    }
    if let Some(brk) = find_rule(first.clone(), Rule::break_expr) {
        let val = brk.into_inner().find(|x| x.as_rule() == Rule::expr)
            .map(lower_expr).transpose()?;
        return Ok(Stmt::Break(val));
    }
    if find_rule(first.clone(), Rule::continue_expr).is_some() {
        return Ok(Stmt::Continue);
    }
    let e = lower_pipeline_expr(first)?;
    Ok(Stmt::Expr(e))
}

fn lower_cond(p: Pair<Rule>) -> Result<Expr, String> {
    // new grammar: cond = { ns_assign_expr }
    // ns_assign_expr = { ns_pipeline_expr ~ (assign_op ~ ns_assign_expr)? }
    // ns_pipeline_expr = { ns_ternary_expr ~ ("|>" ~ ns_ternary_expr)* }
    // ns_ternary_expr  = { ns_range_expr ~ ("??" ~ ns_range_expr)* }
    // We descend through these wrappers to the existing ns_range handling.
    let inner = p.into_inner().next().ok_or("empty cond")?;
    lower_ns_from(inner)
}

// Descend the ns_* chain wrappers (assign/pipeline/ternary) to reach the
// range level, applying ?? as null-coalesce. Reuses lower_ns_range below.
fn lower_ns_from(p: Pair<Rule>) -> Result<Expr, String> {
    match p.as_rule() {
        Rule::ns_assign_expr => {
            // ignore trailing assignment in condition position; take the lhs
            let first = p.into_inner().next().ok_or("empty ns_assign")?;
            lower_ns_from(first)
        }
        Rule::ns_pipeline_expr => {
            let first = p.into_inner().next().ok_or("empty ns_pipeline")?;
            lower_ns_from(first)
        }
        Rule::ns_ternary_expr => {
            let mut inner = p.into_inner();
            let first = inner.next().ok_or("empty ns_ternary")?;
            let mut acc = lower_ns_range(first)?;
            for nxt in inner {
                if nxt.as_rule() == Rule::ns_range_expr {
                    let rhs = lower_ns_range(nxt)?;
                    acc = Expr::If {
                        cond: Box::new(Expr::Binary { op: BinOp::Ne, lhs: Box::new(acc.clone()), rhs: Box::new(Expr::Null) }),
                        then: Box::new(acc),
                        els: Box::new(rhs),
                    };
                }
            }
            Ok(acc)
        }
        Rule::ns_range_expr => lower_ns_range(p),
        other => Err(format!("unexpected condition node {:?}", other)),
    }
}

// ---------- Expression lowering ----------

fn lower_expr(p: Pair<Rule>) -> Result<Expr, String> {
    // new grammar: expr = { stream_expr }
    // stream_expr = { assign_expr ~ (stream_op ~ assign_expr)* }
    let inner = p.into_inner().next().ok_or("empty expr")?;
    lower_stream_expr(inner)
}

fn lower_stream_expr(p: Pair<Rule>) -> Result<Expr, String> {
    // stream_expr = { assign_expr ~ (stream_op ~ assign_expr)* }
    // We give `<-` channel-send semantics: `ch <- v` desugars to Send{ch, v}.
    // `->>` (stream forward) is not implemented; reject it explicitly.
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty stream_expr")?;
    let mut acc = lower_assign_expr(first)?;
    while let Some(op) = inner.next() {
        if op.as_rule() != Rule::stream_op {
            // not an operator token (shouldn't happen); treat as operand fallthrough
            acc = lower_assign_expr(op)?;
            continue;
        }
        let op_txt = op.as_str().trim().to_string();
        let rhs_pair = inner.next().ok_or("stream op without right operand")?;
        let rhs = lower_assign_expr(rhs_pair)?;
        match op_txt.as_str() {
            "<-" => {
                acc = Expr::Send { chan: Box::new(acc), value: Box::new(rhs) };
            }
            "->>" => {
                acc = build_stream_into(acc, rhs);
            }
            other => return Err(format!("stream operator '{}' not yet in Nova Core", other)),
        }
    }
    Ok(acc)
}

fn lower_assign_expr(p: Pair<Rule>) -> Result<Expr, String> {
    // assign_expr = { pipeline_expr ~ (assign_op ~ assign_expr)? }
    let first = p.into_inner().next().ok_or("empty assign_expr")?;
    lower_pipeline_expr(first)
}

fn lower_pipeline_expr(p: Pair<Rule>) -> Result<Expr, String> {
    // pipeline_expr = { ternary_expr ~ ("|>" ~ ternary_expr)* }
    // a |> f  desugars to  f(a) ; a |> f(x)  to  f(x, a)
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty pipeline_expr")?;
    let mut acc = lower_ternary(first)?;
    for stage in inner {
        if stage.as_rule() != Rule::ternary_expr { continue; }
        let staged = lower_ternary(stage)?;
        acc = match staged {
            Expr::Call { callee, mut args } => {
                args.push(acc);
                Expr::Call { callee, args }
            }
            Expr::MethodCall { base, method, mut args } => {
                args.push(acc);
                Expr::MethodCall { base, method, args }
            }
            Expr::Ident(callee) => Expr::Call { callee, args: vec![acc] },
            other => Expr::CallValue { callee: Box::new(other), args: vec![acc] },
        };
    }
    Ok(acc)
}

fn lower_ternary(p: Pair<Rule>) -> Result<Expr, String> {
    // ternary_expr = { range_expr ~ ("??" ~ range_expr)* }
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty ternary")?;
    let mut acc = lower_range(first)?;
    // chain `??` as a null-coalesce -> Nova Core: a ?? b == (if a != null { a } else { b })
    for nxt in inner {
        if nxt.as_rule() == Rule::range_expr {
            let rhs = lower_range(nxt)?;
            acc = Expr::If {
                cond: Box::new(Expr::Binary {
                    op: BinOp::Ne,
                    lhs: Box::new(acc.clone()),
                    rhs: Box::new(Expr::Null),
                }),
                then: Box::new(acc),
                els: Box::new(rhs),
            };
        }
    }
    Ok(acc)
}

fn lower_range(p: Pair<Rule>) -> Result<Expr, String> {
    // range_expr = { or_expr ~ (("..="|"..") ~ or_expr?)? | ("..="|"..") ~ or_expr? }
    // The range operator belongs to THIS rule only if it sits between/around the
    // direct or_expr children — not if ".." appears inside a child (e.g. a[1..3]).
    let full = p.as_str();
    let base = p.as_span().start();
    let children: Vec<Pair<Rule>> = p.into_inner()
        .filter(|x| x.as_rule() == Rule::or_expr).collect();
    let child_spans: Vec<(usize, usize)> = children.iter()
        .map(|c| (c.as_span().start() - base, c.as_span().end() - base)).collect();
    let in_child = |idx: usize| child_spans.iter().any(|(s, e)| idx >= *s && idx < *e);
    // find a ".." that is NOT inside any child's span (i.e. a top-level operator)
    let bytes = full.as_bytes();
    let mut op_pos = None;
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'.' && bytes[i + 1] == b'.' && !in_child(i) {
            op_pos = Some(i);
            break;
        }
        i += 1;
    }
    let op_pos = match op_pos {
        None => return lower_or(children.into_iter().next().ok_or("empty range")?),
        Some(p) => p,
    };
    let inclusive = full[op_pos..].starts_with("..=");
    let left_empty = full[..op_pos].trim().is_empty();
    let mut it = children.into_iter();
    let (lo, hi) = if left_empty {
        (None, it.next().map(lower_or).transpose()?.map(Box::new))
    } else {
        let lo = Some(Box::new(lower_or(it.next().ok_or("range missing lo")?)?));
        let hi = it.next().map(lower_or).transpose()?.map(Box::new);
        (lo, hi)
    };
    Ok(Expr::RangeLit { lo, hi, inclusive })
}

fn lower_or(p: Pair<Rule>) -> Result<Expr, String> {
    // or_expr = { and_expr ~ ("||" ~ and_expr)* }
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty or")?;
    let mut acc = lower_and(first)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::and_expr {
            let rhs = lower_and(nxt)?;
            acc = Expr::Binary { op: BinOp::Or, lhs: Box::new(acc), rhs: Box::new(rhs) };
        }
    }
    Ok(acc)
}

fn lower_and(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty and")?;
    let mut acc = lower_eq(first)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::eq_expr {
            let rhs = lower_eq(nxt)?;
            acc = Expr::Binary { op: BinOp::And, lhs: Box::new(acc), rhs: Box::new(rhs) };
        }
    }
    Ok(acc)
}

fn lower_eq(p: Pair<Rule>) -> Result<Expr, String> {
    // eq_expr = { rel_expr ~ (eq_op ~ rel_expr)* }
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty eq")?;
    let mut acc = lower_rel(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("eq op missing rhs")?;
        let rhs = lower_rel(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

fn lower_rel(p: Pair<Rule>) -> Result<Expr, String> {
    // rel_expr = { bitor_expr ~ (rel_op ~ bitor_expr)* }
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty rel")?;
    let mut acc = lower_bitor(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("rel op missing rhs")?;
        let rhs = lower_bitor(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

// Bitwise/shift operators, left-associative, descending bitor -> bitxor -> bitand -> shift -> add.
fn lower_bitor(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let mut acc = lower_bitxor(inner.next().ok_or("empty bitor")?)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::bitxor_expr {
            acc = Expr::Binary { op: BinOp::BitOr, lhs: Box::new(acc), rhs: Box::new(lower_bitxor(nxt)?) };
        }
    }
    Ok(acc)
}
fn lower_bitxor(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let mut acc = lower_bitand(inner.next().ok_or("empty bitxor")?)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::bitand_expr {
            acc = Expr::Binary { op: BinOp::BitXor, lhs: Box::new(acc), rhs: Box::new(lower_bitand(nxt)?) };
        }
    }
    Ok(acc)
}
fn lower_bitand(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let mut acc = lower_shift(inner.next().ok_or("empty bitand")?)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::shift_expr {
            acc = Expr::Binary { op: BinOp::BitAnd, lhs: Box::new(acc), rhs: Box::new(lower_shift(nxt)?) };
        }
    }
    Ok(acc)
}
fn lower_shift(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let mut acc = lower_add(inner.next().ok_or("empty shift")?)?;
    let mut op = BinOp::Shl;
    for nxt in inner {
        match nxt.as_rule() {
            Rule::shift_op => op = if nxt.as_str().contains(">>") { BinOp::Shr } else { BinOp::Shl },
            Rule::add_expr => {
                acc = Expr::Binary { op, lhs: Box::new(acc), rhs: Box::new(lower_add(nxt)?) };
            }
            _ => {}
        }
    }
    Ok(acc)
}

fn lower_add(p: Pair<Rule>) -> Result<Expr, String> {
    // add_expr = { mul_expr ~ (("+"|"-") ~ mul_expr)* }
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty add")?;
    let mut acc = lower_mul(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("add op missing rhs")?;
        let rhs = lower_mul(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

fn lower_mul(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty mul")?;
    let mut acc = lower_cast(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("mul op missing rhs")?;
        let rhs = lower_cast(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

fn lower_cast(p: Pair<Rule>) -> Result<Expr, String> {
    // cast_expr = { pow_expr ~ (kw_as ~ type_expr)* }
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty cast")?;
    let mut expr = lower_pow(first)?;
    // apply each `as Type` as a real conversion when the target is a known scalar
    for t in inner {
        if t.as_rule() == Rule::type_expr {
            let ty = type_head(t.as_str());
            let callee = match ty.as_str() {
                "Int" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => Some("int"),
                "Float" | "f32" | "f64" => Some("float"),
                "Str" | "String" => Some("str"),
                _ => None,
            };
            if let Some(c) = callee {
                expr = Expr::Call { callee: c.into(), args: vec![expr] };
            }
        }
    }
    Ok(expr)
}

fn lower_pow(p: Pair<Rule>) -> Result<Expr, String> {
    // pow_expr = { postfix_expr ~ (pow_op ~ pow_expr)? }  (right-assoc)
    let mut inner = p.into_inner();
    let base = inner.next().ok_or("empty pow")?;
    let base_expr = lower_postfix(base)?;
    // optional: pow_op then pow_expr
    let mut rest: Vec<Pair<Rule>> = inner.collect();
    if rest.is_empty() {
        return Ok(base_expr);
    }
    // rest = [pow_op, pow_expr]
    let rhs_pair = rest.pop().ok_or("pow missing rhs")?;
    let rhs = lower_pow(rhs_pair)?;
    Ok(Expr::Binary { op: BinOp::Pow, lhs: Box::new(base_expr), rhs: Box::new(rhs) })
}

fn lower_postfix(p: Pair<Rule>) -> Result<Expr, String> {
    // postfix_expr = { unary_expr ~ postfix_op* }
    // Prefix operators bind LOOSER than postfix: `!f(x)` is `!(f(x))`,
    // `-a[i]` is `-(a[i])` — so postfix ops apply to the primary first.
    let mut inner = p.into_inner();
    let unary = inner.next().ok_or("empty postfix")?;
    let ops: Vec<Pair<Rule>> = inner.collect();
    let (prefix_ops, base) = split_unary(unary)?;
    let with_postfix = apply_postfix_ops(base, &ops)?;
    apply_prefix_ops(with_postfix, prefix_ops)
}

// pull a unary_expr apart: its prefix-op strings + the lowered primary
fn split_unary(p: Pair<Rule>) -> Result<(Vec<String>, Expr), String> {
    let mut ops = Vec::new();
    let mut primary = None;
    for inner in p.into_inner() {
        match inner.as_rule() {
            Rule::prefix_op => ops.push(inner.as_str().trim().to_string()),
            Rule::primary_expr => primary = Some(lower_primary(inner)?),
            _ => {}
        }
    }
    Ok((ops, primary.ok_or("unary without primary")?))
}

fn apply_prefix_ops(mut expr: Expr, ops: Vec<String>) -> Result<Expr, String> {
    for op in ops.into_iter().rev() {
        if op.as_str() == "await" {
            expr = Expr::Await(Box::new(expr));
            continue;
        }
        let uop = match op.as_str() {
            "-" => UnOp::Neg,
            "!" => UnOp::Not,
            "~" => UnOp::BitNot,
            "+" => continue,
            other => return Err(format!("Prefix op '{}' not yet in Nova Core", other)),
        };
        expr = Expr::Unary { op: uop, expr: Box::new(expr) };
    }
    Ok(expr)
}

// Shared postfix application for both the main and no_struct chains.
fn is_module_name(name: &str) -> bool {
    matches!(name, "math" | "strings" | "arrays" | "collections" | "io" | "std")
}

fn apply_postfix_ops(mut base: Expr, ops: &[Pair<Rule>]) -> Result<Expr, String> {
    let mut i = 0;
    while i < ops.len() {
        let op = &ops[i];
        if op.as_rule() != Rule::postfix_op {
            i += 1;
            continue;
        }
        let inner_op = op.clone().into_inner().next().ok_or("empty postfix_op")?;
        match inner_op.as_rule() {
            Rule::call_op => {
                let args = lower_args(inner_op)?;
                match base {
                    Expr::Ident(name) => {
                        base = Expr::Call { callee: name, args };
                    }
                    Expr::Field { base: obj, field } => {
                        // module.fn(...) -> qualified stdlib call, not a method call
                        if let Expr::Ident(root) = &*obj {
                            if is_module_name(root) {
                                base = Expr::Call { callee: format!("{}.{}", root, field), args };
                                i += 1;
                                continue;
                            }
                        }
                        base = Expr::MethodCall { base: obj, method: field, args };
                    }
                    // calling the result of any other expression (closure value, etc.)
                    other => {
                        base = Expr::CallValue { callee: Box::new(other), args };
                    }
                }
            }
            Rule::index_op => {
                base = lower_index_op(inner_op, base)?;
            }
            Rule::field_op => {
                let field = field_name(&inner_op)?;
                base = Expr::Field { base: Box::new(base), field };
            }
            Rule::safe_field_op => {
                // a?.b -> Null if a is Null, else a.b
                let field = inner_op.into_inner()
                    .find(|x| x.as_rule() == Rule::ident || x.as_rule() == Rule::int_lit)
                    .map(|x| x.as_str().to_string())
                    .ok_or("safe field without name")?;
                base = Expr::SafeField { base: Box::new(base), field };
            }
            Rule::await_op => {
                // expr.await -> drive the future/handle to its value
                base = Expr::Await(Box::new(base));
            }
            other => return Err(format!("Postfix op {:?} not yet in Nova Core", other)),
        }
        i += 1;
    }
    Ok(base)
}

fn field_name(field_op: &Pair<Rule>) -> Result<String, String> {
    for x in field_op.clone().into_inner() {
        match x.as_rule() {
            Rule::ident => return Ok(x.as_str().to_string()),
            Rule::int_lit => return Ok(x.as_str().to_string()),
            _ => {}
        }
    }
    Err("empty field name".into())
}

fn lower_index_op(p: Pair<Rule>, base: Expr) -> Result<Expr, String> {
    // index_op = { "[" ~ subscript ~ "]" }; the subscript is a single expr.
    // If that expr is a RangeLit, indexing produces a slice (handled at eval).
    let sub = p.into_inner().next().ok_or("empty index")?;
    let idx_expr = sub.into_inner().find(|x| x.as_rule() == Rule::expr)
        .ok_or("empty subscript")?;
    let idx = lower_expr(idx_expr)?;
    Ok(Expr::Index { base: Box::new(base), index: Box::new(idx) })
}

fn lower_array(p: Pair<Rule>) -> Result<Expr, String> {
    // array_expr = { "[" ~ (expr ~ ";" ~ expr | (expr ~ ("," ~ expr)* ~ ","?)?) ~ "]" }
    let txt = p.as_str();
    let parts: Vec<Pair<Rule>> = p.into_inner().filter(|x| x.as_rule() == Rule::expr).collect();
    // detect `[value; count]` repeat form by checking for ';' in source
    let is_repeat = {
        // crude but reliable: a ';' between the brackets at top level
        let inner = &txt[1..txt.len().saturating_sub(1)];
        inner.contains(';') && parts.len() == 2
    };
    if is_repeat {
        // [value; count] -> array_fill(value, count)
        let val = lower_expr(parts[0].clone())?;
        let count = lower_expr(parts[1].clone())?;
        return Ok(Expr::Call { callee: "array_fill".into(), args: vec![val, count] });
    }
    let mut elems = Vec::new();
    for e in parts {
        elems.push(lower_expr(e)?);
    }
    Ok(Expr::Array(elems))
}

fn lower_args(p: Pair<Rule>) -> Result<Vec<Expr>, String> {
    let mut out = Vec::new();
    for inner in p.into_inner() {
        if inner.as_rule() == Rule::args {
            for arg in inner.into_inner() {
                if arg.as_rule() == Rule::arg {
                    // arg = { ident ~ ":" ~ expr | expr }
                    let mut got = None;
                    for x in arg.into_inner() {
                        if x.as_rule() == Rule::expr {
                            got = Some(lower_expr(x)?);
                        }
                    }
                    if let Some(e) = got {
                        out.push(e);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn lower_qualified_path(p: Pair<Rule>) -> Result<Expr, String> {
    // expr_path = { (kw_self|kw_super|kw_crate|ident) ~ ("." ~ ident)* }
    // Dotted field access a.b.c / self.x. Indexing a[i] now comes from the
    // postfix index_op in the new grammar, not from generic args.
    let segments: Vec<Pair<Rule>> = p.into_inner().collect();
    let mut base: Option<Expr> = None;

    for seg in segments {
        match seg.as_rule() {
            Rule::ident => {
                let name = seg.as_str().to_string();
                base = Some(match base.take() {
                    None => Expr::Ident(name),
                    Some(b) => Expr::Field { base: Box::new(b), field: name },
                });
            }
            Rule::kw_self => {
                base = Some(Expr::Ident("self".to_string()));
            }
            Rule::kw_super | Rule::kw_crate => {
                return Err("super/crate paths not supported in Nova Core".into());
            }
            _ => {}
        }
    }
    base.ok_or_else(|| "empty path".into())
}

fn lower_lambda(p: Pair<Rule>) -> Result<Expr, String> {
    // lambda_expr = { kw_async? ~ kw_move? ~ (ident | "(" ~ params? ~ ")") ~ "=>" ~ (block | expr) }
    let mut params: Vec<String> = Vec::new();
    let mut body: Option<LambdaBody> = None;
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::kw_async => return Err("async closures not yet in Nova Core".into()),
            Rule::kw_move => {} // capture-by-move is the default here anyway
            Rule::ident => params.push(part.as_str().to_string()), // single-param shorthand: x => ...
            Rule::params => params = lower_params(part)?,
            Rule::block => {
                // a block-bodied lambda returns its trailing expression, exactly
                // like a function body (the interpreter/VM treat a lone trailing
                // expression as the return value only once it is marked `return`)
                let mut stmts = lower_block_stmts(part)?;
                make_trailing_implicit_return(&mut stmts);
                body = Some(LambdaBody::Block(stmts));
            }
            Rule::expr => {
                body = Some(LambdaBody::Expr(lower_expr(part)?));
            }
            _ => {}
        }
    }
    let body = body.ok_or("lambda without body")?;
    Ok(Expr::Lambda { params, body: Box::new(body) })
}

fn lower_struct_lit(p: Pair<Rule>) -> Result<Expr, String> {
    // struct_literal = { type_path ~ "{" ~ (field_init ~ ...)? ~ "}" }
    let mut name = String::new();
    let mut fields = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::type_path => {
                let txt = part.as_str().trim();
                name = txt.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            }
            Rule::field_init => {
                // field_init = { ident ~ (":" ~ expr)? }
                let mut fname = String::new();
                let mut fval: Option<Expr> = None;
                for x in part.into_inner() {
                    match x.as_rule() {
                        Rule::ident => fname = x.as_str().to_string(),
                        Rule::expr => fval = Some(lower_expr(x)?),
                        _ => {}
                    }
                }
                // shorthand `Point { x }` means field x = variable x
                let value = match fval {
                    Some(e) => e,
                    None => Expr::Ident(fname.clone()),
                };
                fields.push((fname, value));
            }
            _ => {}
        }
    }
    Ok(Expr::StructLit { name, fields })
}

fn lower_comprehension(p: Pair<Rule>) -> Result<Expr, String> {
    // comprehension_expr = { "[" ~ expr ~ comp_clause+ ~ "]" }
    // comp_clause = { kw_for ~ pattern ~ kw_in ~ expr | kw_if ~ expr }
    let mut parts = p.into_inner();
    let body = lower_expr(parts.next().ok_or("empty comprehension body")?)?;
    let mut var = String::new();
    let mut iter: Option<Expr> = None;
    let mut cond: Option<Expr> = None;
    for clause in parts {
        if clause.as_rule() != Rule::comp_clause { continue; }
        let txt = clause.as_str().trim_start();
        let mut inner = clause.into_inner();
        if txt.starts_with("for") {
            let pat = inner.find(|x| x.as_rule() == Rule::pattern)
                .ok_or("comprehension for missing pattern")?;
            var = pat.as_str().trim().to_string();
            let it = inner.find(|x| x.as_rule() == Rule::expr)
                .ok_or("comprehension for missing iterable")?;
            iter = Some(lower_expr(it)?);
        } else {
            let c = inner.find(|x| x.as_rule() == Rule::expr)
                .ok_or("comprehension if missing condition")?;
            cond = Some(lower_expr(c)?);
        }
    }
    Ok(Expr::Comprehension {
        body: Box::new(body),
        var,
        iter: Box::new(iter.ok_or("comprehension without for-clause")?),
        cond: cond.map(Box::new),
    })
}

fn lower_fmt_string(p: Pair<Rule>) -> Result<Expr, String> {
    // fmt_string_lit = ${ "f\"" ~ (fmt_text | fmt_interp)* ~ "\"" }
    let mut out = Vec::new();
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::fmt_text => {
                let t = part.as_str().replace("{{", "{").replace("}}", "}");
                out.push(FmtPart::Lit(unescape_inner(&t)));
            }
            Rule::fmt_interp => {
                // fmt_interp = !{ "{" ~ expr ~ (":" ~ fmt_spec)? ~ "}" }
                if let Some(e) = part.into_inner().find(|x| x.as_rule() == Rule::expr) {
                    out.push(FmtPart::Expr(lower_expr(e)?));
                }
            }
            _ => {}
        }
    }
    Ok(Expr::FmtStr(out))
}

fn unescape_inner(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(other) => { out.push('\\'); out.push(other); }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn lower_map_literal(p: Pair<Rule>) -> Result<Expr, String> {
    // map_literal = { "#{" ~ (map_entry ~ ("," ~ map_entry)*)? ~ "}" }
    // map_entry   = { expr ~ ":" ~ expr }
    let mut entries = Vec::new();
    for entry in p.into_inner() {
        if entry.as_rule() == Rule::map_entry {
            let mut it = entry.into_inner();
            let k = lower_expr(it.next().ok_or("map entry missing key")?)?;
            let v = lower_expr(it.next().ok_or("map entry missing value")?)?;
            entries.push((k, v));
        }
    }
    Ok(Expr::MapLit(entries))
}

fn lower_set_literal(p: Pair<Rule>) -> Result<Expr, String> {
    // set_literal = { "#(" ~ (expr ~ ("," ~ expr)*)? ~ ")" }
    let mut elems = Vec::new();
    for e in p.into_inner() {
        if e.as_rule() == Rule::expr {
            elems.push(lower_expr(e)?);
        }
    }
    Ok(Expr::SetLit(elems))
}

// spawn_expr = { kw_spawn ~ block }
fn lower_spawn(p: Pair<Rule>) -> Result<Expr, String> {
    let block = p.into_inner().find(|x| x.as_rule() == Rule::block)
        .ok_or("spawn without block")?;
    let mut stmts = lower_block_stmts(block)?;
    // a trailing expression becomes the task's result, like a function body
    make_trailing_implicit_return(&mut stmts);
    Ok(Expr::Spawn(stmts))
}

// select_expr = { kw_select ~ "{" ~ select_arm ~ ("," ~ select_arm)* ~ ","? ~ "}" }
// select_arm  = { (kw_chan ~ expr | expr) ~ "=>" ~ expr }
// We support the form: `select { <- ch => body, ... }`, optionally binding the
// received value with `x = <- ch => body` is *not* grammar; instead an arm whose
// channel expr is a Recv binds nothing, so bodies read the value via the channel.
fn lower_select(p: Pair<Rule>) -> Result<crate::ast::Expr, String> {
    let mut arms = Vec::new();
    for arm in p.into_inner() {
        if arm.as_rule() != Rule::select_arm { continue; }
        let parts: Vec<Pair<Rule>> = arm.into_inner().collect();
        let chan_pair = parts.iter().find(|x| x.as_rule() == Rule::expr_path)
            .ok_or("select arm missing channel")?.clone();
        let body_pair = parts.iter().rev().find(|x| x.as_rule() == Rule::assign_expr)
            .ok_or("select arm missing body")?.clone();
        let chan = lower_qualified_path(chan_pair)?;
        let body = lower_assign_expr(body_pair)?;
        arms.push(crate::ast::SelectArm { chan, binding: None, body });
    }
    if arms.is_empty() {
        return Err("select needs at least one arm".into());
    }
    Ok(Expr::Select(arms))
}

fn lower_primary(p: Pair<Rule>) -> Result<Expr, String> {
    let inner = p.into_inner().next().ok_or("empty primary")?;
    match inner.as_rule() {
        Rule::literal => lower_literal(inner),
        Rule::map_literal => lower_map_literal(inner),
        Rule::set_literal => lower_set_literal(inner),
        Rule::comprehension_expr => lower_comprehension(inner),
        Rule::array_expr => lower_array(inner),
        Rule::struct_literal => lower_struct_lit(inner),
        Rule::lambda_expr => lower_lambda(inner),
        Rule::expr_path => lower_qualified_path(inner),
        Rule::grouped_expr => {
            // grouped_expr = { "(" ~ (expr ~ ...)? ~ ")" }
            let e = inner.into_inner().next().ok_or("empty grouped expr")?;
            lower_expr(e)
        }
        Rule::if_expr => lower_if_expr_value(inner),
        Rule::match_expr => lower_match(inner),
        Rule::macro_call => lower_macro_call(inner),
        Rule::spawn_expr => lower_spawn(inner),
        Rule::select_expr => lower_select(inner),
        Rule::block_expr => {
            let blk = inner.into_inner().next().unwrap();
            lower_block_value(blk)
        }
        Rule::flow_expr => {
            Err("flow expressions (return/break) only allowed as statements in Nova Core".into())
        }
        other => Err(format!("Expression form {:?} not yet in Nova Core", other)),
    }
}

fn lower_literal(p: Pair<Rule>) -> Result<Expr, String> {
    let inner = p.into_inner().next().ok_or("empty literal")?;
    let text = inner.as_str();
    match inner.as_rule() {
        Rule::int_lit => {
            let cleaned: String = text.chars().filter(|c| *c != '_').collect();
            match parse_int(&cleaned) {
                Some(v) => Ok(Expr::Int(v)),
                None => Ok(Expr::BigIntLit(cleaned)),
            }
        }
        Rule::float_lit => {
            let cleaned: String = text.chars().filter(|c| *c != '_').collect();
            let v: f64 = cleaned.trim_end_matches("f32").trim_end_matches("f64")
                .parse().map_err(|_| format!("bad float literal: {}", text))?;
            Ok(Expr::Float(v))
        }
        Rule::string_lit => Ok(Expr::Str(unescape(text))),
        Rule::fmt_string_lit => lower_fmt_string(inner),
        Rule::bool_lit => Ok(Expr::Bool(text.starts_with("true"))),
        Rule::null_lit => Ok(Expr::Null),
        other => Err(format!("Literal {:?} not yet in Nova Core", other)),
    }
}

fn parse_int(s: &str) -> Option<i64> {
    let s = s.trim_end_matches(|c: char| c == 'i' || c == 'u' || c.is_ascii_digit() && false);
    // strip optional integer suffix like i32/u64
    let core = s.split(|c| c == 'i' || c == 'u').next().unwrap_or(s);
    if let Some(hex) = core.strip_prefix("0x") { return i64::from_str_radix(hex, 16).ok(); }
    if let Some(bin) = core.strip_prefix("0b") { return i64::from_str_radix(bin, 2).ok(); }
    if let Some(oct) = core.strip_prefix("0o") { return i64::from_str_radix(oct, 8).ok(); }
    core.parse().ok()
}

fn unescape(raw: &str) -> String {
    // strip surrounding quotes and handle common escapes
    let inner = &raw[1..raw.len().saturating_sub(1)];
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('0') => out.push('\0'),
                Some(other) => { out.push('\\'); out.push(other); }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn lower_if_expr_value(p: Pair<Rule>) -> Result<Expr, String> {
    // if used as a value: if cond { a } else { b }
    let mut cond = None;
    let mut blocks = Vec::new();
    let mut else_if: Option<Expr> = None;
    for part in p.into_inner() {
        match part.as_rule() {
            Rule::cond => cond = Some(lower_cond(part)?),
            Rule::block => blocks.push(lower_block_value(part)?),
            Rule::if_expr => else_if = Some(lower_if_expr_value(part)?),
            _ => {}
        }
    }
    let cond = cond.ok_or("if-expr without condition")?;
    let then = blocks.get(0).cloned().ok_or("if-expr without then block")?;
    let els = if let Some(ei) = else_if {
        ei
    } else if let Some(b) = blocks.get(1) {
        b.clone()
    } else {
        Expr::Null
    };
    Ok(Expr::If { cond: Box::new(cond), then: Box::new(then), els: Box::new(els) })
}

fn lower_block_value(p: Pair<Rule>) -> Result<Expr, String> {
    // block = { "{" ~ stmt* ~ expr? ~ "}" }  -> value is the trailing expr
    let mut stmts = Vec::new();
    let mut tail = None;
    for inner in p.into_inner() {
        match inner.as_rule() {
            Rule::stmt => stmts.push(lower_stmt(inner)?),
            Rule::expr => tail = Some(Box::new(lower_expr(inner)?)),
            _ => {}
        }
    }
    // PEG's `stmt*` greedily consumes the trailing expression, so a value block
    // like `{ a }` lands `a` as a statement with no tail. In value position the
    // last expression statement *is* the block's value: promote it to the tail.
    if tail.is_none() {
        if let Some(Stmt::Expr(e)) = stmts.last() {
            let e = e.clone();
            stmts.pop();
            tail = Some(Box::new(e));
        }
    }
    Ok(Expr::Block { stmts, tail })
}

// ----- helpers for the no_struct (condition) chain -----

fn lower_ns_range(p: Pair<Rule>) -> Result<Expr, String> {
    let first = p.into_inner().next().ok_or("empty ns_range")?;
    if first.as_rule() == Rule::ns_or_expr {
        lower_ns_or(first)
    } else {
        Err("range not allowed in this condition position in Nova Core".into())
    }
}

fn lower_ns_or(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty ns_or")?;
    let mut acc = lower_ns_and(first)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::ns_and_expr {
            let rhs = lower_ns_and(nxt)?;
            acc = Expr::Binary { op: BinOp::Or, lhs: Box::new(acc), rhs: Box::new(rhs) };
        }
    }
    Ok(acc)
}

fn lower_ns_and(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty ns_and")?;
    let mut acc = lower_ns_eq(first)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::ns_eq_expr {
            let rhs = lower_ns_eq(nxt)?;
            acc = Expr::Binary { op: BinOp::And, lhs: Box::new(acc), rhs: Box::new(rhs) };
        }
    }
    Ok(acc)
}

fn lower_ns_eq(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty ns_eq")?;
    let mut acc = lower_ns_rel(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("ns_eq op missing rhs")?;
        let rhs = lower_ns_rel(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

fn lower_ns_rel(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty ns_rel")?;
    let mut acc = lower_ns_bitor(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("ns_rel op missing rhs")?;
        let rhs = lower_ns_bitor(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

fn lower_ns_bitor(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let mut acc = descend_ns_to_add(inner.next().ok_or("empty ns_bitor")?)?;
    for nxt in inner {
        if nxt.as_rule() == Rule::ns_bitxor_expr {
            acc = Expr::Binary { op: BinOp::BitOr, lhs: Box::new(acc), rhs: Box::new(descend_ns_to_add(nxt)?) };
        }
    }
    Ok(acc)
}

fn descend_ns_to_add(p: Pair<Rule>) -> Result<Expr, String> {
    match p.as_rule() {
        Rule::ns_bitxor_expr => {
            let mut inner = p.into_inner();
            let mut acc = descend_ns_to_add(inner.next().ok_or("empty")?)?;
            for nxt in inner {
                if nxt.as_rule() == Rule::ns_bitand_expr {
                    acc = Expr::Binary { op: BinOp::BitXor, lhs: Box::new(acc), rhs: Box::new(descend_ns_to_add(nxt)?) };
                }
            }
            Ok(acc)
        }
        Rule::ns_bitand_expr => {
            let mut inner = p.into_inner();
            let mut acc = descend_ns_to_add(inner.next().ok_or("empty")?)?;
            for nxt in inner {
                if nxt.as_rule() == Rule::ns_shift_expr {
                    acc = Expr::Binary { op: BinOp::BitAnd, lhs: Box::new(acc), rhs: Box::new(descend_ns_to_add(nxt)?) };
                }
            }
            Ok(acc)
        }
        Rule::ns_shift_expr => {
            let mut inner = p.into_inner();
            let mut acc = descend_ns_to_add(inner.next().ok_or("empty")?)?;
            let mut op = BinOp::Shl;
            for nxt in inner {
                match nxt.as_rule() {
                    Rule::shift_op => op = if nxt.as_str().contains(">>") { BinOp::Shr } else { BinOp::Shl },
                    Rule::ns_add_expr => acc = Expr::Binary { op, lhs: Box::new(acc), rhs: Box::new(lower_ns_add(nxt)?) },
                    _ => {}
                }
            }
            Ok(acc)
        }
        Rule::ns_add_expr => lower_ns_add(p),
        other => Err(format!("unexpected ns node {:?}", other)),
    }
}

fn lower_ns_add(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty ns_add")?;
    let mut acc = lower_ns_mul(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("ns_add op missing rhs")?;
        let rhs = lower_ns_mul(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

fn lower_ns_mul(p: Pair<Rule>) -> Result<Expr, String> {
    let mut inner = p.into_inner();
    let first = inner.next().ok_or("empty ns_mul")?;
    let mut acc = lower_ns_cast(first)?;
    while let Some(op) = inner.next() {
        let rhs_pair = inner.next().ok_or("ns_mul op missing rhs")?;
        let rhs = lower_ns_cast(rhs_pair)?;
        let bop = str_to_binop(op.as_str().trim())?;
        acc = Expr::Binary { op: bop, lhs: Box::new(acc), rhs: Box::new(rhs) };
    }
    Ok(acc)
}

fn lower_ns_cast(p: Pair<Rule>) -> Result<Expr, String> {
    let first = p.into_inner().next().ok_or("empty ns_cast")?;
    lower_ns_pow(first)
}

fn lower_ns_pow(p: Pair<Rule>) -> Result<Expr, String> {
    // ns_pow_expr = { ns_postfix_expr ~ (pow_op ~ ns_pow_expr)? }
    let mut inner = p.into_inner();
    let base = inner.next().ok_or("empty ns_pow")?;
    let base_expr = lower_ns_postfix(base)?;
    let mut rest: Vec<Pair<Rule>> = inner.collect();
    if rest.is_empty() {
        return Ok(base_expr);
    }
    let rhs_pair = rest.pop().ok_or("ns_pow missing rhs")?;
    let rhs = lower_ns_pow(rhs_pair)?;
    Ok(Expr::Binary { op: BinOp::Pow, lhs: Box::new(base_expr), rhs: Box::new(rhs) })
}

fn lower_ns_postfix(p: Pair<Rule>) -> Result<Expr, String> {
    // prefix binds looser than postfix here too: `!f(x)` in a condition
    let mut inner = p.into_inner();
    let unary = inner.next().ok_or("empty ns_postfix")?;
    let ops: Vec<Pair<Rule>> = inner.collect();
    let (prefix_ops, base) = split_ns_unary(unary)?;
    let with_postfix = apply_postfix_ops(base, &ops)?;
    apply_prefix_ops(with_postfix, prefix_ops)
}

fn split_ns_unary(p: Pair<Rule>) -> Result<(Vec<String>, Expr), String> {
    let mut ops = Vec::new();
    let mut primary = None;
    for inner in p.into_inner() {
        match inner.as_rule() {
            Rule::prefix_op => ops.push(inner.as_str().trim().to_string()),
            Rule::ns_primary_expr => primary = Some(lower_ns_primary(inner)?),
            _ => {}
        }
    }
    Ok((ops, primary.ok_or("ns_unary without primary")?))
}

fn lower_ns_primary(p: Pair<Rule>) -> Result<Expr, String> {
    let inner = p.into_inner().next().ok_or("empty ns_primary")?;
    match inner.as_rule() {
        Rule::literal => lower_literal(inner),
        Rule::array_expr => lower_array(inner),
        Rule::lambda_expr => lower_lambda(inner),
        Rule::expr_path => lower_qualified_path(inner),
        Rule::grouped_expr => {
            let e = inner.into_inner().next().ok_or("empty grouped expr")?;
            lower_expr(e)
        }
        Rule::if_expr => lower_if_expr_value(inner),
        other => Err(format!("Condition expr form {:?} not yet in Nova Core", other)),
    }
}

fn str_to_binop(s: &str) -> Result<BinOp, String> {
    Ok(match s {
        "+" => BinOp::Add,
        "-" => BinOp::Sub,
        "*" => BinOp::Mul,
        "/" => BinOp::Div,
        "%" => BinOp::Rem,
        "**" => BinOp::Pow,
        "==" => BinOp::Eq,
        "!=" => BinOp::Ne,
        "<" => BinOp::Lt,
        "<=" => BinOp::Le,
        ">" => BinOp::Gt,
        ">=" => BinOp::Ge,
        "&&" => BinOp::And,
        "||" => BinOp::Or,
        other => return Err(format!("Operator '{}' not yet in Nova Core", other)),
    })
}
