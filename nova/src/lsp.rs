// A full-featured Language Server for Nova (`nova lsp`), speaking LSP over stdio
// (Content-Length framed JSON-RPC). It backs the editor experience with a symbol
// model built from the AST (`parser::parse_program`) plus a lexical index over
// the document text (for precise declaration positions and cursor queries), and
// the type checker (`types::Checker`) for diagnostics.
//
// Capabilities: diagnostics, hover (signature/kind), completion (keywords,
// symbols, locals, builtins, fields after `.`), signatureHelp, goto-definition,
// references, documentHighlight, documentSymbol, workspaceSymbol, rename,
// formatting (via `fmt`), semanticTokens, foldingRange.
//
// JSON is hand-rolled (no external crate) — we read a handful of fields and emit
// compact objects, so a small extractor + escaper suffice.

use crate::ast::*;
use std::collections::HashMap;
use std::io::{self, Read, Write};

pub fn run() {
    let mut stdin = io::stdin();
    let mut docs: HashMap<String, String> = HashMap::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let msg = match read_message(&mut stdin, &mut buf, &mut tmp) { Some(m) => m, None => break };
        let method = str_field(&msg, "method").unwrap_or_default();
        let uri = str_field(&msg, "uri");
        let text = uri.as_ref().and_then(|u| docs.get(u)).cloned();
        match method.as_str() {
            "initialize" => reply(&msg, INIT_RESULT),
            "initialized" | "$/setTrace" | "workspace/didChangeConfiguration" => {}
            "textDocument/didOpen" | "textDocument/didChange" => {
                if let (Some(u), Some(t)) = (uri.clone(), str_field(&msg, "text")) {
                    docs.insert(u.clone(), t.clone());
                    publish_diagnostics(&u, &t);
                }
            }
            "textDocument/didClose" => { if let Some(u) = &uri { docs.remove(u); } }
            "textDocument/hover" => reply(&msg, &hover(&text, pos(&msg))),
            "textDocument/completion" => reply(&msg, &completion(&text, pos(&msg))),
            "textDocument/signatureHelp" => reply(&msg, &signature_help(&text, pos(&msg))),
            "textDocument/definition" => reply(&msg, &definition(uri.as_deref(), &text, pos(&msg))),
            "textDocument/references" => reply(&msg, &references(uri.as_deref(), &text, pos(&msg))),
            "textDocument/documentHighlight" => reply(&msg, &highlights(&text, pos(&msg))),
            "textDocument/documentSymbol" => reply(&msg, &document_symbols(&text)),
            "workspace/symbol" => reply(&msg, &workspace_symbols(&docs, &str_field(&msg, "query").unwrap_or_default())),
            "textDocument/rename" => reply(&msg, &rename(uri.as_deref(), &text, pos(&msg), &str_field(&msg, "newName").unwrap_or_default())),
            "textDocument/formatting" => reply(&msg, &formatting(&text)),
            "textDocument/semanticTokens/full" => reply(&msg, &semantic_tokens(&text)),
            "textDocument/foldingRange" => reply(&msg, &folding_ranges(&text)),
            "shutdown" => reply(&msg, "null"),
            "exit" => break,
            _ => { if raw_id(&msg) != "null" { reply(&msg, "null"); } }
        }
    }
}

const INIT_RESULT: &str = "{\"capabilities\":{\
    \"textDocumentSync\":1,\
    \"hoverProvider\":true,\
    \"completionProvider\":{\"triggerCharacters\":[\".\",\":\"]},\
    \"signatureHelpProvider\":{\"triggerCharacters\":[\"(\",\",\"]},\
    \"definitionProvider\":true,\
    \"referencesProvider\":true,\
    \"documentHighlightProvider\":true,\
    \"documentSymbolProvider\":true,\
    \"workspaceSymbolProvider\":true,\
    \"renameProvider\":true,\
    \"documentFormattingProvider\":true,\
    \"foldingRangeProvider\":true,\
    \"semanticTokensProvider\":{\"legend\":{\"tokenTypes\":[\"keyword\",\"function\",\"type\",\"variable\",\"parameter\",\"string\",\"number\",\"comment\",\"operator\",\"property\",\"enumMember\"],\"tokenModifiers\":[]},\"full\":true}\
    },\"serverInfo\":{\"name\":\"nova-lsp\",\"version\":\"2\"}}";

// ---- symbol model -----------------------------------------------------------

#[derive(Clone)]
struct Sym {
    name: String,
    kind: &'static str,   // function/struct/enum/trait/const/machine/field/variant
    detail: String,       // signature or kind detail
    line: u32, col: u32,  // 0-indexed declaration position
    children: Vec<Sym>,
}

// LSP SymbolKind numbers
fn sym_kind_num(k: &str) -> u8 {
    match k { "function" => 12, "struct" => 23, "enum" => 10, "trait" => 11,
              "const" => 14, "machine" => 5, "field" => 8, "variant" => 22, _ => 13 }
}

// Build the top-level symbol table from AST + text-located declaration positions.
fn symbols(text: &str) -> Vec<Sym> {
    let prog = match crate::parser::parse_program(text) { Ok(p) => p, Err(_) => return text_symbols(text) };
    let mut out = Vec::new();
    for it in &prog.items {
        match it {
            Item::Func(f) => {
                let (l, c) = decl_pos(text, &["fn"], &f.name);
                out.push(Sym { name: f.name.clone(), kind: "function", detail: fn_sig(f), line: l, col: c, children: vec![] });
            }
            Item::Struct(s) => {
                let (l, c) = decl_pos(text, &["struct", "data"], &s.name);
                let children = s.fields.iter().map(|fname| Sym { name: fname.clone(), kind: "field", detail: format!("field {}", fname), line: l, col: c, children: vec![] }).collect();
                out.push(Sym { name: s.name.clone(), kind: "struct", detail: format!("struct {}", s.name), line: l, col: c, children });
            }
            Item::Enum(e) => {
                let (l, c) = decl_pos(text, &["enum", "union"], &e.name);
                let children = e.variants.iter().map(|v| Sym { name: v.name.clone(), kind: "variant", detail: if v.arity > 0 { format!("{}({} field(s))", v.name, v.arity) } else { v.name.clone() }, line: l, col: c, children: vec![] }).collect();
                out.push(Sym { name: e.name.clone(), kind: "enum", detail: format!("enum {}", e.name), line: l, col: c, children });
            }
            Item::Trait(t) => { let (l, c) = decl_pos(text, &["trait"], &t.name); out.push(Sym { name: t.name.clone(), kind: "trait", detail: format!("trait {}", t.name), line: l, col: c, children: vec![] }); }
            Item::Const { name, .. } => { let (l, c) = decl_pos(text, &["const", "static"], name); out.push(Sym { name: name.clone(), kind: "const", detail: format!("const {}", name), line: l, col: c, children: vec![] }); }
            Item::Machine(m) => { let (l, c) = decl_pos(text, &["machine"], &m.name); out.push(Sym { name: m.name.clone(), kind: "machine", detail: format!("machine {}", m.name), line: l, col: c, children: vec![] }); }
            _ => {}
        }
    }
    out
}

fn fn_sig(f: &Func) -> String {
    let ps: Vec<String> = f.params.iter().enumerate().map(|(i, p)| match f.param_types.get(i).and_then(|t| t.clone()) {
        Some(t) => format!("{}: {}", p, t), None => p.clone() }).collect();
    let mut s = format!("fn {}({})", f.name, ps.join(", "));
    if let Some(rt) = &f.ret_type { s.push_str(&format!(" -> {}", rt)); }
    s
}

// fallback when the file doesn't parse: scan text for declarations
fn text_symbols(text: &str) -> Vec<Sym> {
    let mut out = Vec::new();
    for (li, line) in text.lines().enumerate() {
        for kw in ["fn", "struct", "data", "enum", "union", "trait", "const", "machine"] {
            if let Some(name) = after_kw(line, kw) {
                let col = line.find(&name).unwrap_or(0) as u32;
                let kind = match kw { "fn" => "function", "struct" | "data" => "struct", "enum" | "union" => "enum", "trait" => "trait", "const" => "const", _ => "machine" };
                out.push(Sym { name, kind, detail: format!("{}", kw), line: li as u32, col, children: vec![] });
                break;
            }
        }
    }
    out
}

// the identifier following `<kw> ` on a line (declaration name), if any
fn after_kw(line: &str, kw: &str) -> Option<String> {
    let t = line.trim_start().strip_prefix("pub ").unwrap_or(line.trim_start());
    let t = t.strip_prefix("async ").unwrap_or(t);
    let rest = t.strip_prefix(kw)?;
    if !rest.starts_with(|c: char| c.is_whitespace()) { return None; }
    let name: String = rest.trim_start().chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
    if name.is_empty() { None } else { Some(name) }
}

// locate a declaration `<kw> name` in the text (0-indexed line/col of `name`)
fn decl_pos(text: &str, kws: &[&str], name: &str) -> (u32, u32) {
    for (li, line) in text.lines().enumerate() {
        for kw in kws {
            if after_kw(line, kw).as_deref() == Some(name) {
                let col = line.find(name).unwrap_or(0) as u32;
                return (li as u32, col);
            }
        }
    }
    (0, 0)
}

// ---- cursor helpers ---------------------------------------------------------

fn nth_line(text: &str, n: u32) -> &str { text.lines().nth(n as usize).unwrap_or("") }

// the identifier token at (line, ch): returns (word, start_col, end_col)
fn word_at(text: &str, line: u32, ch: u32) -> Option<(String, u32, u32)> {
    let l = nth_line(text, line);
    let bytes: Vec<char> = l.chars().collect();
    let ch = ch as usize;
    if ch > bytes.len() { return None; }
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    let mut s = ch;
    while s > 0 && is_id(bytes[s - 1]) { s -= 1; }
    let mut e = ch;
    while e < bytes.len() && is_id(bytes[e]) { e += 1; }
    if s == e { return None; }
    let w: String = bytes[s..e].iter().collect();
    if w.chars().next().map_or(true, |c| c.is_ascii_digit()) { return None; }
    Some((w, s as u32, e as u32))
}

// (line, character) from a request's params.position
fn pos(msg: &str) -> (u32, u32) {
    (int_field(msg, "line").unwrap_or(0), int_field(msg, "character").unwrap_or(0))
}

// ---- handlers ---------------------------------------------------------------

fn publish_diagnostics(uri: &str, text: &str) {
    let mut diags: Vec<String> = Vec::new();
    match crate::parser::parse_program(text) {
        Err(e) => { let (l, c) = find_lc(&e).unwrap_or((1, 1)); diags.push(diag(l, c, &first_line(&e), 1)); }
        Ok(prog) => {
            let (errors, warnings) = crate::types::Checker::new(&prog).check(&prog);
            for e in errors { let (l, c) = find_lc(&e).unwrap_or((1, 1)); diags.push(diag(l, c, &clean(&e), 1)); }
            for w in warnings { let (l, c) = find_lc(&w).unwrap_or((1, 1)); diags.push(diag(l, c, &clean(&w), 2)); }
        }
    }
    send(&format!("{{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{{\"uri\":{},\"diagnostics\":[{}]}}}}", json_str(uri), diags.join(",")));
}

fn diag(line: u32, col: u32, msg: &str, sev: u8) -> String {
    let (l, c) = (line.saturating_sub(1), col.saturating_sub(1));
    format!("{{\"range\":{{\"start\":{{\"line\":{},\"character\":{}}},\"end\":{{\"line\":{},\"character\":{}}}}},\"severity\":{},\"source\":\"nova\",\"message\":{}}}", l, c, l, c + 1, sev, json_str(msg))
}

fn hover(text: &Option<String>, (line, ch): (u32, u32)) -> String {
    let Some(t) = text else { return "null".into() };
    let Some((w, _, _)) = word_at(t, line, ch) else { return "null".into() };
    let info = symbols(t).into_iter().find(|s| s.name == w).map(|s| format!("```nova\n{}\n```", s.detail))
        .or_else(|| builtin_doc(&w).map(|d| d.to_string()))
        .or_else(|| KEYWORDS.contains(&w.as_str()).then(|| format!("keyword `{}`", w)));
    match info {
        Some(v) => format!("{{\"contents\":{{\"kind\":\"markdown\",\"value\":{}}}}}", json_str(&v)),
        None => "null".into(),
    }
}

fn completion(text: &Option<String>, (line, ch): (u32, u32)) -> String {
    let Some(t) = text else { return "{\"isIncomplete\":false,\"items\":[]}".into() };
    let cur = nth_line(t, line);
    let before: String = cur.chars().take(ch as usize).collect();
    let mut items: Vec<String> = Vec::new();
    let syms = symbols(t);
    if before.trim_end().ends_with('.') {
        // field/member completion: offer all struct fields (best-effort)
        for s in &syms { for f in &s.children { if f.kind == "field" { items.push(citem(&f.name, 5, &f.detail)); } } }
    } else {
        for k in KEYWORDS { items.push(citem(k, 14, "keyword")); }
        for s in &syms {
            let cik = match s.kind { "function" => 3, "struct" => 22, "enum" => 13, "trait" => 8, "const" => 21, _ => 6 };
            items.push(citem(&s.name, cik, &s.detail));
            if s.kind == "enum" { for v in &s.children { items.push(citem(&v.name, 20, &v.detail)); } }
        }
        for (l, p) in local_names(t) { items.push(citem(&l, if p { 6 } else { 6 }, if p { "parameter" } else { "local" })); }
        for b in BUILTINS { items.push(citem(b, 3, "builtin")); }
    }
    format!("{{\"isIncomplete\":false,\"items\":[{}]}}", items.join(","))
}

fn citem(label: &str, kind: u8, detail: &str) -> String {
    format!("{{\"label\":{},\"kind\":{},\"detail\":{}}}", json_str(label), kind, json_str(detail))
}

fn signature_help(text: &Option<String>, (line, ch): (u32, u32)) -> String {
    let Some(t) = text else { return "null".into() };
    let cur = nth_line(t, line);
    let before: Vec<char> = cur.chars().take(ch as usize).collect();
    // find the nearest unclosed `name(` before the cursor
    let mut depth = 0i32; let mut i = before.len();
    while i > 0 { i -= 1; match before[i] { ')' => depth += 1, '(' => { if depth == 0 { break; } depth -= 1; }, _ => {} } }
    if i == 0 && before.get(0) != Some(&'(') { if !before.contains(&'(') { return "null".into(); } }
    let mut j = i; while j > 0 && (before[j - 1].is_alphanumeric() || before[j - 1] == '_') { j -= 1; }
    let name: String = before[j..i].iter().collect();
    if name.is_empty() { return "null".into(); }
    let Some(sig) = symbols(t).into_iter().find(|s| s.name == name && s.kind == "function") else { return "null".into() };
    let active = before[i..].iter().filter(|&&c| c == ',').count();
    format!("{{\"signatures\":[{{\"label\":{},\"parameters\":[]}}],\"activeSignature\":0,\"activeParameter\":{}}}", json_str(&sig.detail), active)
}

fn definition(uri: Option<&str>, text: &Option<String>, (line, ch): (u32, u32)) -> String {
    let (Some(t), Some(u)) = (text, uri) else { return "null".into() };
    let Some((w, _, _)) = word_at(t, line, ch) else { return "null".into() };
    match symbols(t).into_iter().find(|s| s.name == w) {
        Some(s) => location(u, s.line, s.col, s.col + w.chars().count() as u32),
        None => "null".into(),
    }
}

fn references(uri: Option<&str>, text: &Option<String>, (line, ch): (u32, u32)) -> String {
    let (Some(t), Some(u)) = (text, uri) else { return "[]".into() };
    let Some((w, _, _)) = word_at(t, line, ch) else { return "[]".into() };
    let locs: Vec<String> = token_occurrences(t, &w).into_iter()
        .map(|(l, s, e)| location(u, l, s, e)).collect();
    format!("[{}]", locs.join(","))
}

fn highlights(text: &Option<String>, (line, ch): (u32, u32)) -> String {
    let Some(t) = text else { return "[]".into() };
    let Some((w, _, _)) = word_at(t, line, ch) else { return "[]".into() };
    let hs: Vec<String> = token_occurrences(t, &w).into_iter()
        .map(|(l, s, e)| format!("{{\"range\":{}}}", range(l, s, l, e))).collect();
    format!("[{}]", hs.join(","))
}

fn document_symbols(text: &Option<String>) -> String {
    let Some(t) = text else { return "[]".into() };
    let items: Vec<String> = symbols(t).iter().map(|s| doc_sym(s)).collect();
    format!("[{}]", items.join(","))
}

fn doc_sym(s: &Sym) -> String {
    let children: Vec<String> = s.children.iter().map(|c| doc_sym(c)).collect();
    let r = range(s.line, s.col, s.line, s.col + s.name.chars().count() as u32);
    format!("{{\"name\":{},\"detail\":{},\"kind\":{},\"range\":{},\"selectionRange\":{},\"children\":[{}]}}",
        json_str(&s.name), json_str(&s.detail), sym_kind_num(s.kind), r, r, children.join(","))
}

fn workspace_symbols(docs: &HashMap<String, String>, query: &str) -> String {
    let q = query.to_lowercase();
    let mut out = Vec::new();
    for (uri, text) in docs {
        for s in symbols(text) {
            if q.is_empty() || s.name.to_lowercase().contains(&q) {
                out.push(format!("{{\"name\":{},\"kind\":{},\"location\":{}}}",
                    json_str(&s.name), sym_kind_num(s.kind),
                    format!("{{\"uri\":{},\"range\":{}}}", json_str(uri), range(s.line, s.col, s.line, s.col + s.name.chars().count() as u32))));
            }
        }
    }
    format!("[{}]", out.join(","))
}

fn rename(uri: Option<&str>, text: &Option<String>, (line, ch): (u32, u32), new: &str) -> String {
    let (Some(t), Some(u)) = (text, uri) else { return "null".into() };
    let Some((w, _, _)) = word_at(t, line, ch) else { return "null".into() };
    let edits: Vec<String> = token_occurrences(t, &w).into_iter()
        .map(|(l, s, e)| format!("{{\"range\":{},\"newText\":{}}}", range(l, s, l, e), json_str(new))).collect();
    format!("{{\"changes\":{{{}:[{}]}}}}", json_str(u), edits.join(","))
}

fn formatting(text: &Option<String>) -> String {
    let Some(t) = text else { return "[]".into() };
    match crate::parser::parse_program(t) {
        Ok(p) => {
            let out = crate::fmt::format_program(&p);
            let last = t.lines().count() as u32 + 1;
            format!("[{{\"range\":{},\"newText\":{}}}]", range(0, 0, last, 0), json_str(&out))
        }
        Err(_) => "[]".into(),
    }
}

fn folding_ranges(text: &Option<String>) -> String {
    let Some(t) = text else { return "[]".into() };
    let lines: Vec<&str> = t.lines().collect();
    let mut stack: Vec<u32> = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for (li, line) in lines.iter().enumerate() {
        for c in line.chars() {
            match c { '{' => stack.push(li as u32), '}' => { if let Some(start) = stack.pop() { if li as u32 > start { out.push(format!("{{\"startLine\":{},\"endLine\":{}}}", start, li)); } } }, _ => {} }
        }
    }
    format!("[{}]", out.join(","))
}

// ---- semantic tokens --------------------------------------------------------

fn semantic_tokens(text: &Option<String>) -> String {
    let Some(t) = text else { return "{\"data\":[]}".into() };
    // token type indices per the legend advertised in INIT_RESULT
    let (mut data, mut pl, mut pc) = (Vec::<u32>::new(), 0u32, 0u32);
    let fns: std::collections::HashSet<String> = symbols(t).iter().filter(|s| s.kind == "function").map(|s| s.name.clone()).collect();
    for (li, line) in t.lines().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '/' && i + 1 < chars.len() && chars[i + 1] == '/' {
                push_tok(&mut data, &mut pl, &mut pc, li as u32, i as u32, (chars.len() - i) as u32, 7); break;
            } else if c == '"' {
                let start = i; i += 1; while i < chars.len() && chars[i] != '"' { if chars[i] == '\\' { i += 1; } i += 1; } i += 1;
                push_tok(&mut data, &mut pl, &mut pc, li as u32, start as u32, (i - start) as u32, 5);
            } else if c.is_ascii_digit() {
                let start = i; while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '.' || chars[i] == '_') { i += 1; }
                push_tok(&mut data, &mut pl, &mut pc, li as u32, start as u32, (i - start) as u32, 6);
            } else if c.is_alphabetic() || c == '_' {
                let start = i; while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') { i += 1; }
                let w: String = chars[start..i].iter().collect();
                let tt = if KEYWORDS.contains(&w.as_str()) { 0 }
                    else if w.chars().next().map_or(false, |c| c.is_uppercase()) { 2 }
                    else if fns.contains(&w) || (i < chars.len() && chars[i] == '(') { 1 }
                    else { 3 };
                push_tok(&mut data, &mut pl, &mut pc, li as u32, start as u32, (i - start) as u32, tt);
            } else { i += 1; }
        }
    }
    let body: Vec<String> = data.iter().map(|n| n.to_string()).collect();
    format!("{{\"data\":[{}]}}", body.join(","))
}

fn push_tok(data: &mut Vec<u32>, pl: &mut u32, pc: &mut u32, line: u32, col: u32, len: u32, tt: u32) {
    let dl = line - *pl;
    let dc = if dl == 0 { col - *pc } else { col };
    data.extend_from_slice(&[dl, dc, len, tt, 0]);
    *pl = line; *pc = col;
}

// ---- shared: token scan, ranges, locations ----------------------------------

// every whole-word occurrence of `w` as an identifier token: (line, start, end)
fn token_occurrences(text: &str, w: &str) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    let wl = w.chars().count() as u32;
    for (li, line) in text.lines().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_alphabetic() || chars[i] == '_' {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') { i += 1; }
                let tok: String = chars[start..i].iter().collect();
                if tok == w { out.push((li as u32, start as u32, start as u32 + wl)); }
            } else { i += 1; }
        }
    }
    out
}

fn range(sl: u32, sc: u32, el: u32, ec: u32) -> String {
    format!("{{\"start\":{{\"line\":{},\"character\":{}}},\"end\":{{\"line\":{},\"character\":{}}}}}", sl, sc, el, ec)
}
fn location(uri: &str, line: u32, sc: u32, ec: u32) -> String {
    format!("{{\"uri\":{},\"range\":{}}}", json_str(uri), range(line, sc, line, ec))
}

// all local (let / param / loop) names declared in the file, with is_param flag
fn local_names(text: &str) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    if let Ok(prog) = crate::parser::parse_program(text) {
        for it in &prog.items {
            if let Item::Func(f) = it {
                for p in &f.params { out.push((p.clone(), true)); }
                collect_lets(&f.body, &mut out);
            }
        }
    }
    out
}
fn collect_lets(body: &[Stmt], out: &mut Vec<(String, bool)>) {
    for s in body {
        match s {
            Stmt::Let { name, .. } => out.push((name.clone(), false)),
            Stmt::ForRange { var, body, .. } | Stmt::ForEach { var, body, .. } => { out.push((var.clone(), false)); collect_lets(body, out); }
            Stmt::If { then, els, .. } => { collect_lets(then, out); if let Some(e) = els { collect_lets(e, out); } }
            Stmt::While { body, .. } => collect_lets(body, out),
            _ => {}
        }
    }
}

// ---- framing + tiny JSON ----------------------------------------------------

fn reply(msg: &str, result: &str) {
    send(&format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{}}}", raw_id(msg), result));
}

fn read_message(stdin: &mut io::Stdin, buf: &mut Vec<u8>, tmp: &mut [u8; 8192]) -> Option<String> {
    loop {
        if let Some(hend) = find(buf, b"\r\n\r\n") {
            let header = String::from_utf8_lossy(&buf[..hend]).to_string();
            let len: usize = header.lines().find_map(|l| l.strip_prefix("Content-Length:").map(|v| v.trim().parse().unwrap_or(0))).unwrap_or(0);
            let body_start = hend + 4;
            if buf.len() >= body_start + len {
                let body = String::from_utf8_lossy(&buf[body_start..body_start + len]).to_string();
                buf.drain(..body_start + len);
                return Some(body);
            }
        }
        let n = stdin.read(tmp).ok()?;
        if n == 0 { return None; }
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> { hay.windows(needle.len()).position(|w| w == needle) }

fn send(body: &str) {
    let out = io::stdout(); let mut o = out.lock();
    let _ = write!(o, "Content-Length: {}\r\n\r\n{}", body.len(), body);
    let _ = o.flush();
}

fn raw_id(msg: &str) -> String {
    if let Some(p) = msg.find("\"id\"") {
        let rest = &msg[p + 4..];
        if let Some(colon) = rest.find(':') {
            let after = rest[colon + 1..].trim_start();
            let end = after.find(|c| c == ',' || c == '}').unwrap_or(after.len());
            return after[..end].trim().to_string();
        }
    }
    "null".into()
}

fn str_field(msg: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let mut from = 0;
    while let Some(p) = msg[from..].find(&pat) {
        let i = from + p + pat.len();
        let after = msg[i..].trim_start().strip_prefix(':').map(|s| s.trim_start());
        if let Some(a) = after { if a.starts_with('"') { return Some(unescape(&a[1..])); } }
        from = i;
    }
    None
}

fn int_field(msg: &str, key: &str) -> Option<u32> {
    let pat = format!("\"{}\"", key);
    let p = msg.find(&pat)?;
    let after = msg[p + pat.len()..].trim_start().strip_prefix(':')?.trim_start();
    let num: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

fn unescape(s: &str) -> String {
    let mut out = String::new(); let mut it = s.chars();
    while let Some(c) = it.next() {
        match c {
            '"' => break,
            '\\' => match it.next() {
                Some('n') => out.push('\n'), Some('t') => out.push('\t'), Some('r') => out.push('\r'),
                Some('"') => out.push('"'), Some('\\') => out.push('\\'), Some('/') => out.push('/'),
                Some('u') => { let h: String = (0..4).filter_map(|_| it.next()).collect(); if let Ok(n) = u32::from_str_radix(&h, 16) { if let Some(ch) = char::from_u32(n) { out.push(ch); } } }
                Some(o) => out.push(o), None => break,
            },
            _ => out.push(c),
        }
    }
    out
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2); out.push('"');
    for c in s.chars() { match c {
        '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"), '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)), c => out.push(c),
    } }
    out.push('"'); out
}

fn first_line(s: &str) -> String { s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string() }
fn clean(s: &str) -> String { s.lines().next().unwrap_or(s).trim_start_matches("error:").trim_start_matches("warning:").trim().to_string() }

fn find_lc(s: &str) -> Option<(u32, u32)> {
    if let Some(p) = s.find("line ") {
        let rest = &s[p + 5..];
        let ln: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let (Ok(l), Some(cp)) = (ln.parse::<u32>(), rest.find("col ")) {
            let cn: String = rest[cp + 4..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(c) = cn.parse::<u32>() { return Some((l, c)); }
        }
    }
    let b = s.as_bytes(); let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() { let st = i; while i < b.len() && b[i].is_ascii_digit() { i += 1; }
            if i < b.len() && b[i] == b':' { let mid = i; i += 1; let cs = i; while i < b.len() && b[i].is_ascii_digit() { i += 1; }
                if i > cs { return Some((s[st..mid].parse().ok()?, s[cs..i].parse().ok()?)); } } } else { i += 1; }
    }
    None
}

// ---- vocab ------------------------------------------------------------------

const KEYWORDS: &[&str] = &["fn","let","mut","if","else","while","for","in","return","match","struct","data","enum","union","trait","impl","const","static","machine","initial","async","await","spawn","yield","break","continue","defer","try","catch","finally","throw","use","import","type","migrate","where","true","false","null","and","or","not"];

const BUILTINS: &[&str] = &["print","println","len","push","pop","str","int","float","to_int","to_float","upper","lower","split","slice","abs","min","max","sqrt","floor","ceil","round","keys","values","contains","range","sort","map","filter","reduce","sum","chr","ord","input","read_file","write_file"];

fn builtin_doc(w: &str) -> Option<&'static str> {
    if BUILTINS.contains(&w) { Some("Nova builtin function") } else { None }
}
