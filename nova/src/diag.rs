// Rich diagnostic rendering.
//
// The parser, checker, and interpreter all report errors as plain strings, and
// the located ones carry a `line L, col C:` (or `line L:`) prefix — inserted by
// the parser's `At` position markers. This module turns such a message into the
// familiar caret frame that modern compilers print:
//
//     error: index 10 out of bounds (len 3)
//       --> prog.nova:3:9
//        |
//      3 |     print(xs[10])
//        |         ^
//
// When no locator is present (or its line is outside `src`, e.g. a position that
// came from an inlined import), it degrades gracefully to `error: <message>`.

// Parse a leading `line L[, col C]:` locator, returning ((line, col?), rest).
fn split_locator(msg: &str) -> Option<((usize, Option<usize>), &str)> {
    let rest = msg.strip_prefix("line ")?;
    let (line_str, after) = rest.split_once(|c: char| c == ',' || c == ':')?;
    let line: usize = line_str.trim().parse().ok()?;
    // either `, col C: rest`  or  `: rest`
    if let Some(after_col) = after.trim_start().strip_prefix("col ") {
        let (col_str, tail) = after_col.split_once(':')?;
        let col: usize = col_str.trim().parse().ok()?;
        Some(((line, Some(col)), tail.trim_start()))
    } else {
        // `after` began right after the ':' that split_once consumed
        Some(((line, None), after.trim_start()))
    }
}

// Render an error message with source context. `label` is the leading word
// (e.g. "error", "runtime error"). `path` names the file for the `-->` line.
pub fn render(label: &str, path: &str, src: &str, msg: &str) -> String {
    match split_locator(msg) {
        Some(((line, col), body)) => {
            let src_line = src.lines().nth(line.saturating_sub(1));
            match src_line {
                Some(text) => {
                    let col = col.unwrap_or(1).max(1);
                    let gutter = line.to_string();
                    let pad = " ".repeat(gutter.len());
                    // caret sits under column `col`; tabs in the prefix are kept
                    // so the caret aligns with a tab-rendered source line
                    let mut caret = String::new();
                    for ch in text.chars().take(col.saturating_sub(1)) {
                        caret.push(if ch == '\t' { '\t' } else { ' ' });
                    }
                    caret.push('^');
                    format!(
                        "{label}: {body}\n {pad}--> {path}:{line}:{col}\n \
                         {pad}|\n {gutter} | {text}\n {pad}| {caret}",
                    )
                }
                // line out of range for this file (e.g. inlined import position)
                None => format!("{label}: {body}\n {}--> {path}:{line}", " "),
            }
        }
        None => format!("{label}: {msg}"),
    }
}
