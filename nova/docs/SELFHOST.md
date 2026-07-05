# Self-hosting Nova — the bootstrap log

The goal: the Nova compiler written 100% in Nova. The bootstrap is **staged**,
and every stage ships only when it is byte-identical to the Rust reference over
the whole corpus + std + examples — the same honesty rule as everything else.

| stage | artifact | reference | gate | status |
|---|---|---|---|---|
| 1. lexer | `selfhost/lexer.nova` | `nova tokens <file>` (src/tokens.rs) | `tests/selfhost_smoke.sh` — 94 files byte-identical, incl. the lexer lexing **itself**; 4-tier identical | ✅ done |
| 2. parser | `selfhost/parser.nova` → canonical AST dump | a Rust AST dumper | every corpus+std file equal | ⏳ next |
| 3. checker | `selfhost/checker.nova` | `nova check` diagnostics | same messages | planned |
| 4. eval | `selfhost/eval.nova` (tree-walk) | `nova run` | same program output | planned |
| 5. fixpoint | Nova builds `selfhost/*` with itself | the Rust-built binary | outputs converge | planned |

## Stage 1 — the lexer (done)

### The canonical token dump
`nova tokens <file>` prints one token per line as `kind<TAB>text` (with `\`,
newline, tab, CR escaped) and a final `eof` line. The spec mirrors the lexical
layer of `src/nova.pest` exactly:

- **trivia** (skipped): whitespace; `//`-family line comments; `/* */` block
  comments **with nesting**.
- **kw** — only the 29 pest `hard_keyword`s. Contextual keywords (`union`,
  `type`, `machine`, `data`, …) lex as `ident`, exactly as the grammar treats
  them.
- **int** — decimal with `_`, `0x`/`0b`/`0o`, optional `i/u 8|16|32|64|128|size`
  suffix. **float** — `dec.dec[exp][f32|f64]` or `dec exp …`; a `.` is consumed
  only when a digit follows, so `1..2` lexes as int, `..`, int.
- **str / char / raw** (`r#"…"#` with matched hashes) **/ tag**
  (`json"…"`/`sql"…"`/`re"…"`) literals per the grammar's escape rules.
- **fstr** — a whole f-string is ONE token; `{`/`}` track interpolation depth,
  `{{`/`}}` are literal, and a `"` inside an interpolation toggles an
  inner-string mode (so `f"v={get("k")}!"` is one token).
- **life** — `'ident` when it isn't a char literal (a closing `'` wins).
- **punct** — maximal munch, 3-char (`<<= >>= ..= === !== <=> ??=`) before
  2-char (`<< >> => == != >= <= && || ** ?? .. -> := += -= *= /= %= &= |= ^=`)
  before single characters.

### The Nova implementation
`selfhost/lexer.nova` implements the same spec in pure Nova: `read_file` +
an O(n) char-array precompute (string `for` iteration), `ord` for character
classes, `substring` for token text, and the same maximal-munch tables. It
uses Nova's own attribute system on itself — `#[comptime]` builds the keyword
map and punctuation tables once before `main`, `#[intent]` documents the
helpers.

### The gate
`tests/selfhost_smoke.sh` (wired into CI):
1. For **every** file in `tests/corpus/`, `std/`, `examples/` **plus the lexer
   itself**: `nova tokens f` vs `nova run selfhost/lexer.nova f` must be
   byte-identical (94/94 at landing).
2. The lexer is a Nova program, so the 4-tier discipline applies to it too:
   `run` / `vm` / `vm --no-jit` / `vm --jit` outputs must all agree.

### Honest scope
Stage 1 is the lexer only. The parser/checker/eval stages are real, separate
efforts (tracked above) — nothing beyond the lexer is claimed self-hosted yet.
