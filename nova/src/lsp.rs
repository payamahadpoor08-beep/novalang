// A minimal but real Language Server for Nova (`nova lsp`), speaking LSP over
// stdio (Content-Length framed JSON-RPC). It implements the core editor loop:
//   initialize -> capabilities
//   textDocument/didOpen + didChange -> publishDiagnostics (parse + type errors)
//   textDocument/hover -> a short info string
//   shutdown / exit
// JSON is hand-rolled (no external crate): we only need to read a few string
// fields and emit small objects, so a tiny extractor + escaper suffice.

use std::collections::HashMap;
use std::io::{self, Read, Write};

pub fn run() {
    let mut stdin = io::stdin();
    let mut docs: HashMap<String, String> = HashMap::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        // read one framed message
        let msg = match read_message(&mut stdin, &mut buf, &mut tmp) {
            Some(m) => m,
            None => break, // EOF
        };
        let method = str_field(&msg, "method").unwrap_or_default();
        match method.as_str() {
            "initialize" => {
                let id = raw_id(&msg);
                send(&format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"capabilities\":{{\
                     \"textDocumentSync\":1,\"hoverProvider\":true}},\
                     \"serverInfo\":{{\"name\":\"nova-lsp\",\"version\":\"1\"}}}}}}", id));
            }
            "initialized" => {}
            "textDocument/didOpen" | "textDocument/didChange" => {
                if let Some(uri) = str_field(&msg, "uri") {
                    // didOpen has textDocument.text; didChange has contentChanges[].text
                    if let Some(text) = str_field(&msg, "text") {
                        docs.insert(uri.clone(), text.clone());
                        publish_diagnostics(&uri, &text);
                    }
                }
            }
            "textDocument/hover" => {
                let id = raw_id(&msg);
                let n = str_field(&msg, "uri").and_then(|u| docs.get(&u).map(|t| t.lines().count())).unwrap_or(0);
                send(&format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":{{\"contents\":{{\"kind\":\"markdown\",\
                     \"value\":{}}}}}}}", id, json_str(&format!("Nova — {} line(s). Run `nova check` for full diagnostics.", n))));
            }
            "shutdown" => { let id = raw_id(&msg); send(&format!("{{\"jsonrpc\":\"2.0\",\"id\":{},\"result\":null}}", id)); }
            "exit" => break,
            _ => {}
        }
    }
}

// Parse + type-check `text` and publish diagnostics for `uri`.
fn publish_diagnostics(uri: &str, text: &str) {
    let mut diags: Vec<String> = Vec::new();
    match crate::parser::parse_program(text) {
        Err(e) => {
            let (line, col) = parse_err_pos(&e);
            diags.push(diagnostic(line, col, &first_line(&e)));
        }
        Ok(prog) => {
            let (errors, warnings) = crate::types::Checker::new(&prog).check(&prog);
            for e in errors { let (l, c) = msg_pos(&e); diags.push(diagnostic(l, c, &clean(&e))); }
            for w in warnings { let (l, c) = msg_pos(&w); diags.push(diagnostic_warn(l, c, &clean(&w))); }
        }
    }
    send(&format!(
        "{{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/publishDiagnostics\",\"params\":{{\
         \"uri\":{},\"diagnostics\":[{}]}}}}", json_str(uri), diags.join(",")));
}

fn diagnostic(line: u32, col: u32, msg: &str) -> String { diag_sev(line, col, msg, 1) }
fn diagnostic_warn(line: u32, col: u32, msg: &str) -> String { diag_sev(line, col, msg, 2) }
fn diag_sev(line: u32, col: u32, msg: &str, sev: u8) -> String {
    let l = line.saturating_sub(1);
    let c = col.saturating_sub(1);
    format!("{{\"range\":{{\"start\":{{\"line\":{},\"character\":{}}},\"end\":{{\"line\":{},\"character\":{}}}}},\
             \"severity\":{},\"source\":\"nova\",\"message\":{}}}", l, c, l, c + 1, sev, json_str(msg))
}

// ---- framing ----------------------------------------------------------------

fn read_message(stdin: &mut io::Stdin, buf: &mut Vec<u8>, tmp: &mut [u8; 4096]) -> Option<String> {
    loop {
        // do we have a full header + body in buf?
        if let Some(hend) = find(buf, b"\r\n\r\n") {
            let header = String::from_utf8_lossy(&buf[..hend]).to_string();
            let len: usize = header.lines()
                .find_map(|l| l.strip_prefix("Content-Length:").map(|v| v.trim().parse().unwrap_or(0)))
                .unwrap_or(0);
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

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn send(body: &str) {
    let out = io::stdout();
    let mut o = out.lock();
    let _ = write!(o, "Content-Length: {}\r\n\r\n{}", body.len(), body);
    let _ = o.flush();
}

// ---- tiny JSON helpers ------------------------------------------------------

// the raw text of the "id" value (number or string) for a request, or "null"
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

// value of the first `"key": "..."` string field found (with unescaping)
fn str_field(msg: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let mut from = 0;
    while let Some(p) = msg[from..].find(&pat) {
        let i = from + p + pat.len();
        let rest = &msg[i..];
        if let Some(colon) = rest.find(':') {
            let after = rest[colon + 1..].trim_start();
            if after.starts_with('"') {
                return Some(unescape(&after[1..]));
            }
        }
        from = i;
    }
    None
}

// read a JSON string body (without the opening quote) up to the closing quote
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        match c {
            '"' => break,
            '\\' => match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('u') => {
                    let hex: String = (0..4).filter_map(|_| it.next()).collect();
                    if let Ok(n) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(n) { out.push(ch); }
                    }
                }
                Some(other) => out.push(other),
                None => break,
            },
            _ => out.push(c),
        }
    }
    out
}

// emit a JSON string literal (with quotes) from a Rust string
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---- position extraction ----------------------------------------------------

fn first_line(s: &str) -> String { s.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string() }
fn clean(s: &str) -> String { s.lines().next().unwrap_or(s).trim_start_matches("error:").trim_start_matches("warning:").trim().to_string() }

// pest parse errors contain "--> L:C" or " L:C"
fn parse_err_pos(s: &str) -> (u32, u32) { find_lc(s).unwrap_or((1, 1)) }
fn msg_pos(s: &str) -> (u32, u32) { find_lc(s).unwrap_or((1, 1)) }

fn find_lc(s: &str) -> Option<(u32, u32)> {
    // the checker phrases positions as "line N, col M"
    if let Some(p) = s.find("line ") {
        let rest = &s[p + 5..];
        let lnum: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let (Ok(l), Some(cp)) = (lnum.parse::<u32>(), rest.find("col ")) {
            let cnum: String = rest[cp + 4..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(c) = cnum.parse::<u32>() { return Some((l, c)); }
        }
    }
    // pest parse errors use "<digits>:<digits>" (line:col)
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() { i += 1; }
            if i < b.len() && b[i] == b':' {
                let mid = i; i += 1; let cs = i;
                while i < b.len() && b[i].is_ascii_digit() { i += 1; }
                if i > cs {
                    let line = s[start..mid].parse().ok()?;
                    let col = s[cs..i].parse().ok()?;
                    return Some((line, col));
                }
            }
        } else { i += 1; }
    }
    None
}
