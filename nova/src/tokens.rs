// The canonical Nova token dump — the reference for self-hosting stage 1.
//
// `nova tokens <file>` prints one token per line as `kind<TAB>text` (with
// `\\`, `\n`, `\t`, `\r` escaped in the text) and a final `eof` line. The
// spec mirrors the lexical layer of src/nova.pest exactly:
//   * trivia (skipped): spaces/tabs/CR/LF; `//`+`///`+`//!` line comments;
//     `/* */` block comments WITH nesting.
//   * kw    — only the pest `hard_keyword` set (28); `union`/`type`/`machine`…
//             are contextual and lex as `ident`, exactly as the grammar treats
//             them.
//   * ident — [A-Za-z_][A-Za-z0-9_]* not a hard keyword.
//   * int   — dec (with `_`), 0x/0b/0o, optional (i|u)(8|16|32|64|128|size).
//   * float — dec.dec[exp][fsfx] | dec exp [fsfx]; a `.` is consumed only when
//             a digit follows (so `1..2` lexes int `..` int).
//   * str/char/raw(r#"…"# matched hashes)/tag(json|sql|re-prefixed) literals.
//   * fstr  — one token for the whole f-string; `{`/`}` track interpolation
//             depth, `{{`/`}}` are literal, a `"` at depth>0 toggles an
//             inner-string mode.
//   * life  — 'ident where it isn't a char literal (a closing quote wins).
//   * punct — maximal munch: 3-char ops before 2-char before single chars.
//
// `selfhost/lexer.nova` implements this same spec in 100% Nova; the
// differential gate `tests/selfhost_smoke.sh` requires byte-identical dumps
// over the whole corpus + std + examples.

pub const HARD_KEYWORDS: [&str; 29] = [
    "fn", "let", "mut", "if", "else", "for", "while", "loop", "break",
    "continue", "return", "struct", "enum", "trait", "impl", "use", "mod",
    "match", "as", "pub", "self", "super", "crate", "spawn", "select",
    "await", "true", "false", "null",
];

// 3-char operators, maximal-munch before 2-char. `->>` (stream) and `>>>`
// (unsigned shift) are in the grammar's `stream_op`/`shift_op` rules.
const PUNCT3: [&str; 9] = ["<<=", ">>=", "..=", "===", "!==", "<=>", "??=", "->>", ">>>"];
const PUNCT2: [&str; 22] = [
    "<<", ">>", "=>", "==", "!=", ">=", "<=", "&&", "||", "**", "??", "..",
    "->", ":=", "+=", "-=", "*=", "/=", "%=", "^=", "<-", "|>",
];
// note: `&=` and `|=` are 2-char too but must lose to `&&`/`||` prefixes never
// arising ambiguously; keep them in the table after the logical ops.
const PUNCT2B: [&str; 2] = ["&=", "|="];

pub fn lex(src: &str) -> Vec<(&'static str, String)> {
    let cs: Vec<char> = src.chars().collect();
    let n = cs.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    let at = |i: usize| -> char { if i < n { cs[i] } else { '\0' } };
    let is_ident_start = |c: char| c.is_ascii_alphabetic() || c == '_';
    let is_ident_cont = |c: char| c.is_ascii_alphanumeric() || c == '_';

    while i < n {
        let c = cs[i];
        // trivia
        if c == ' ' || c == '\t' || c == '\r' || c == '\n' { i += 1; continue; }
        if c == '/' && at(i + 1) == '/' {
            while i < n && cs[i] != '\n' { i += 1; }
            continue;
        }
        if c == '/' && at(i + 1) == '*' {
            let mut depth = 1;
            i += 2;
            while i < n && depth > 0 {
                if cs[i] == '/' && at(i + 1) == '*' { depth += 1; i += 2; }
                else if cs[i] == '*' && at(i + 1) == '/' { depth -= 1; i += 2; }
                else { i += 1; }
            }
            continue;
        }
        // identifiers / keywords / string-prefix forms
        if is_ident_start(c) {
            let start = i;
            while i < n && is_ident_cont(cs[i]) { i += 1; }
            let word: String = cs[start..i].iter().collect();
            // f"…" — one token, interpolation-aware
            if word == "f" && at(i) == '"' {
                i += 1; // opening quote
                let (mut depth, mut inner) = (0i32, false);
                while i < n {
                    let ch = cs[i];
                    if inner {
                        if ch == '\\' { i += 2; continue; }
                        if ch == '"' { inner = false; }
                        i += 1;
                        continue;
                    }
                    if ch == '\\' { i += 2; continue; }
                    if ch == '{' {
                        if at(i + 1) == '{' { i += 2; continue; }
                        depth += 1; i += 1; continue;
                    }
                    if ch == '}' {
                        if depth == 0 && at(i + 1) == '}' { i += 2; continue; }
                        if depth > 0 { depth -= 1; }
                        i += 1; continue;
                    }
                    if ch == '"' {
                        if depth == 0 { i += 1; break; }
                        inner = true; i += 1; continue;
                    }
                    i += 1;
                }
                out.push(("fstr", collect(&cs, start, i)));
                continue;
            }
            // json"…" / sql"…" / re"…"
            if (word == "json" || word == "sql" || word == "re") && at(i) == '"' {
                i += 1;
                while i < n && cs[i] != '"' {
                    if cs[i] == '\\' { i += 1; }
                    i += 1;
                }
                i += 1; // closing quote
                out.push(("tag", collect(&cs, start, i.min(n))));
                continue;
            }
            // r"…" / r#"…"# with matched hashes
            if word == "r" && (at(i) == '"' || at(i) == '#') {
                let mut hashes = 0usize;
                let mut j = i;
                while at(j) == '#' { hashes += 1; j += 1; }
                if at(j) == '"' {
                    j += 1;
                    'raw: while j < n {
                        if cs[j] == '"' {
                            let mut k = 0usize;
                            while k < hashes && at(j + 1 + k) == '#' { k += 1; }
                            if k == hashes { j += 1 + hashes; break 'raw; }
                        }
                        j += 1;
                    }
                    out.push(("raw", collect(&cs, start, j.min(n))));
                    i = j;
                    continue;
                }
                // `r` followed by `#` but no quote: plain ident, fall through
            }
            let kind = if HARD_KEYWORDS.contains(&word.as_str()) { "kw" } else { "ident" };
            out.push((kind, word));
            continue;
        }
        // numbers
        if c.is_ascii_digit() {
            let start = i;
            if c == '0' && (at(i + 1) == 'x' || at(i + 1) == 'b' || at(i + 1) == 'o') {
                let base = at(i + 1);
                i += 2;
                let ok = |ch: char| match base {
                    'x' => ch.is_ascii_hexdigit() || ch == '_',
                    'b' => ch == '0' || ch == '1' || ch == '_',
                    _ => ('0'..='7').contains(&ch) || ch == '_',
                };
                while i < n && ok(cs[i]) { i += 1; }
                eat_int_suffix(&cs, &mut i);
                out.push(("int", collect(&cs, start, i)));
                continue;
            }
            while i < n && (cs[i].is_ascii_digit() || cs[i] == '_') { i += 1; }
            let mut is_float = false;
            if at(i) == '.' && at(i + 1).is_ascii_digit() {
                is_float = true;
                i += 1;
                while i < n && (cs[i].is_ascii_digit() || cs[i] == '_') { i += 1; }
            }
            // exponent: e/E [+/-] digits (only when digits actually follow)
            if at(i) == 'e' || at(i) == 'E' {
                let mut j = i + 1;
                if at(j) == '+' || at(j) == '-' { j += 1; }
                if at(j).is_ascii_digit() {
                    is_float = true;
                    i = j;
                    while i < n && cs[i].is_ascii_digit() { i += 1; }
                }
            }
            if is_float {
                // float suffix f32/f64
                if at(i) == 'f' && (peek2(&cs, i + 1) == "32" || peek2(&cs, i + 1) == "64") { i += 3; }
                out.push(("float", collect(&cs, start, i)));
            } else {
                eat_int_suffix(&cs, &mut i);
                out.push(("int", collect(&cs, start, i)));
            }
            continue;
        }
        // plain string
        if c == '"' {
            let start = i;
            i += 1;
            while i < n && cs[i] != '"' {
                if cs[i] == '\\' { i += 1; }
                i += 1;
            }
            i += 1;
            out.push(("str", collect(&cs, start, i.min(n))));
            continue;
        }
        // char literal vs lifetime
        if c == '\'' {
            if at(i + 1) == '\\' {
                // escaped char: '\x' (escape may be u{…})
                let start = i;
                i += 2; // ' and backslash
                if at(i) == 'u' && at(i + 1) == '{' {
                    while i < n && cs[i] != '}' { i += 1; }
                    i += 1;
                } else {
                    i += 1;
                }
                if at(i) == '\'' { i += 1; }
                out.push(("char", collect(&cs, start, i.min(n))));
                continue;
            }
            if at(i + 2) == '\'' && at(i + 1) != '\'' {
                out.push(("char", collect(&cs, i, i + 3)));
                i += 3;
                continue;
            }
            if is_ident_start(at(i + 1)) {
                let start = i;
                i += 2;
                while i < n && is_ident_cont(cs[i]) { i += 1; }
                out.push(("life", collect(&cs, start, i)));
                continue;
            }
            out.push(("punct", "'".to_string()));
            i += 1;
            continue;
        }
        // punctuation: maximal munch 3 → 2 → 1
        let p3: String = cs[i..(i + 3).min(n)].iter().collect();
        if p3.len() == 3 && PUNCT3.contains(&p3.as_str()) {
            out.push(("punct", p3));
            i += 3;
            continue;
        }
        let p2: String = cs[i..(i + 2).min(n)].iter().collect();
        if p2.len() == 2 && (PUNCT2.contains(&p2.as_str()) || PUNCT2B.contains(&p2.as_str())) {
            out.push(("punct", p2));
            i += 2;
            continue;
        }
        out.push(("punct", c.to_string()));
        i += 1;
    }
    out.push(("eof", String::new()));
    out
}

fn collect(cs: &[char], a: usize, b: usize) -> String { cs[a..b.min(cs.len())].iter().collect() }

fn peek2(cs: &[char], i: usize) -> String { collect(cs, i, i + 2) }

fn eat_int_suffix(cs: &[char], i: &mut usize) {
    let n = cs.len();
    let at = |j: usize| -> char { if j < n { cs[j] } else { '\0' } };
    if at(*i) == 'i' || at(*i) == 'u' {
        let rest: String = cs[(*i + 1)..(*i + 5).min(n)].iter().collect();
        for s in ["128", "64", "32", "16", "8"] {
            if rest.starts_with(s) { *i += 1 + s.len(); return; }
        }
        if rest.starts_with("size") { *i += 5; }
    }
}

// escape token text for the one-line dump: `\` `\n` `\t` `\r`
pub fn escape(text: &str) -> String {
    let mut s = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\t' => s.push_str("\\t"),
            '\r' => s.push_str("\\r"),
            _ => s.push(c),
        }
    }
    s
}

pub fn dump(src: &str) -> String {
    let mut out = String::new();
    for (kind, text) in lex(src) {
        if kind == "eof" { out.push_str("eof\n"); }
        else { out.push_str(&format!("{}\t{}\n", kind, escape(&text))); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kinds(src: &str) -> Vec<String> {
        lex(src).into_iter().map(|(k, t)| if t.is_empty() { k.to_string() } else { format!("{}:{}", k, t) }).collect()
    }
    #[test] fn keywords_vs_contextual() {
        assert_eq!(kinds("fn union type"), ["kw:fn", "ident:union", "ident:type", "eof"]);
        assert_eq!(kinds("truely true"), ["ident:truely", "kw:true", "eof"]);
    }
    #[test] fn numbers() {
        assert_eq!(kinds("1..2"), ["int:1", "punct:..", "int:2", "eof"]);
        assert_eq!(kinds("1.5e3 0xFF_i64 0b1_0 0o77 2i32 3usize"),
                   ["float:1.5e3", "int:0xFF_i64", "int:0b1_0", "int:0o77", "int:2i32", "int:3usize", "eof"]);
        assert_eq!(kinds("1e9 1e x2"), ["float:1e9", "int:1", "ident:e", "ident:x2", "eof"]);
    }
    #[test] fn strings_and_raw() {
        assert_eq!(kinds(r##""a\"b" r"c" r#"d"e"# 'x' '\n'"##),
                   ["str:\"a\\\"b\"", "raw:r\"c\"", "raw:r#\"d\"e\"#", "char:'x'", "char:'\\n'", "eof"]);
    }
    #[test] fn fstring_one_token() {
        assert_eq!(kinds(r#"f"x={a+1} {{lit}}""#), [r#"fstr:f"x={a+1} {{lit}}""#, "eof"]);
        // interpolation containing a string literal
        assert_eq!(kinds(r#"f"v={get("k")}!""#), [r#"fstr:f"v={get("k")}!""#, "eof"]);
    }
    #[test] fn tagged_and_lifetime() {
        assert_eq!(kinds(r#"json"{}" ret 're"#), ["tag:json\"{}\"", "ident:ret", "life:'re", "eof"]);
    }
    #[test] fn nested_block_comment() {
        assert_eq!(kinds("a /* x /* y */ z */ b"), ["ident:a", "ident:b", "eof"]);
    }
    #[test] fn punct_maximal_munch() {
        assert_eq!(kinds("a<<=b ..= <=> x??=y a**b"),
                   ["ident:a", "punct:<<=", "ident:b", "punct:..=", "punct:<=>", "ident:x", "punct:??=", "ident:y", "ident:a", "punct:**", "ident:b", "eof"]);
    }
    #[test] fn punct_stream_shift_pipeline_ops() {
        // grammar operators that must munch as single tokens: `->>` stream,
        // `>>>` unsigned shift, `<-` channel send, `|>` pipeline.
        assert_eq!(kinds("a->>b a>>>b a<-b a|>b"),
                   ["ident:a", "punct:->>", "ident:b", "ident:a", "punct:>>>", "ident:b",
                    "ident:a", "punct:<-", "ident:b", "ident:a", "punct:|>", "ident:b", "eof"]);
        // still split correctly around them
        assert_eq!(kinds("a-> b"), ["ident:a", "punct:->", "ident:b", "eof"]);
        assert_eq!(kinds("a>> b"), ["ident:a", "punct:>>", "ident:b", "eof"]);
    }
}
