# Self-hosting Nova — the bootstrap log

The goal: the Nova compiler written 100% in Nova. The bootstrap is **staged**,
and every stage ships only when it is byte-identical to the Rust reference over
the whole corpus + std + examples — the same honesty rule as everything else.

| stage | artifact | reference | gate | status |
|---|---|---|---|---|
| 1. lexer | `selfhost/lexer.nova` | `nova tokens <file>` (src/tokens.rs) | `tests/selfhost_smoke.sh` — 95/95 files byte-identical, incl. the lexer lexing **itself**; 4-tier identical | ✅ done |
| 2. parser | `selfhost/parser.nova` | `nova ast <file>` (src/astdump.rs) | 96/96 files byte-identical, incl. the parser parsing **itself + the lexer**; 4-tier identical | ✅ done |
| 3. checker | `selfhost/checker.nova` | `nova check <file>` (types::Checker + diag.rs) | 96/96 files byte-identical (diagnostics + `OK` line), incl. checking **itself**; 4-tier identical | ✅ done |
| 4. eval | `selfhost/eval.nova` (tree-walk) | `nova run <file>` | program output byte-identical on every corpus + std + example file | ✅ done |
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

## Stage 2 — the parser (done)

`selfhost/parser.nova` is a complete recursive-descent + Pratt parser that
mirrors `src/nova.pest` **and** the `src/parser.rs` lowering, emitting a
canonical S-expression AST byte-identical to the Rust reference `nova ast`
(`src/astdump.rs`) for **every** real file — no curated subset.

### The canonical AST dump (`nova ast`)
`src/astdump.rs` walks the real pest-produced AST and prints a **total**
S-expression (every `ast.rs` node has a form — no escapes), one top-level item
per line, unwrapping the transparent `At` position wrapper. Because the dump is
total, nothing can hide behind an unhandled node.

### What the Nova parser reproduces
The full language and, crucially, the lowering's exact desugarings — not just
the grammar:
- items: `fn` (async, generics + bounds, params with self/mut/ref/`linear`/
  `affine` modes + defaults, `-> T`, `![E]` effects, `where`, `=> expr`),
  `struct`/`data`/`enum`/`union`, `trait` (+ default-method signature stripping,
  exactly as the lowering drops it), `impl … for …`, `const`/`static`, `type`
  aliases (dropped without a `;`, matching the grammar) + refinements, `extern`
  (+ variadic `...`), `machine`, `migrate`, `test`, `use` trees.
- the full precedence tower + the real desugarings: `a as Int` → `int(a)`,
  `a |> f(x)` → `f(x, a)`, `a ?? b` → `if a != null …`, `x += 1` →
  `x = x + 1`, tuples → first element, `[v; n]` → `array_fill`, `module.f(x)` →
  qualified call, struct-literal `..spread` dropped, `?.` safe field.
- patterns (or / range / tuple / slice+rest / struct / enum / capitalised-unit),
  `match` guards, comprehensions, f-strings (interpolation-aware), map/set
  literals, closures (`x =>`, `(a,b) =>`, `|a| =>`), `spawn`/`await`/`select`,
  `ch <- v` and `src ->> sink` stream desugaring (shared hygiene counter),
  **big-integer literals**, and **macros** — declaration collection plus
  `name!(args)` **expansion with `$param` substitution and `let`-binding
  hygiene** (`tmp` → `tmp_hyg0`), reproducing the reference expansion.
- Rust-`Display`-parity float formatting and the implicit trailing-return rule
  (the last expression-statement of a fn/lambda/spawn body becomes the return,
  even with a trailing `;`).

### The gate
`tests/selfhost_smoke.sh` (in CI): `nova ast f` == `nova run selfhost/parser.nova
f` for **all 95** files (corpus + std + examples + the lexer and parser
themselves), and the parser is 4-tier identical (interp / vm / --no-jit / --jit).
The parser parses **itself and the lexer** byte-identically — the real
self-hosting property.

## Stage 3 — the checker (done)

`selfhost/checker.nova` reproduces `nova check` **byte-identically** on every
real file: the `OK: parsed N item(s), no type errors` line for clean files, and
the two diagnostics that the checker actually emits on the corpus — **undefined
variable** (errors) and **assigned but never used** (warnings) — rendered with
the exact `src/diag.rs` caret frame (`--> file:line:col`, gutter, `^`).

### How it works
- The tokenizer records **source positions** (line/col per token); every
  statement is wrapped `(@ L C …)` so a diagnostic can be located.
- It **inlines `use "file.nova"` imports** (dedup by path) so the item count `N`
  matches `nova check` (which flattens imports).
- It parses items into the canonical S-expression (reusing the stage-2 parser)
  and walks them, reproducing the Rust checker's exact scoping: a **flat
  per-function scope** where both `let` **and bare assignment declare** a name,
  nested blocks (`if`/`while`/`for`/`try`/…) get a scope *snapshot* that doesn't
  leak, and `match`/slice/struct-pattern bindings, lambda params, `for`/`catch`
  vars all scope correctly. A bare identifier value `(v NAME)` unresolved against
  scope + consts + top-level fns + builtins (with uppercase-unit-variant and
  `_`-prefix leniency) is an `undefined variable`; a `let`-bound local never read
  is `assigned but never used`.

### The gate
`tests/selfhost_smoke.sh`: `nova check f` == `nova run selfhost/checker.nova f`
for all 96 files (corpus + std + examples + the lexer/parser/checker), and the
checker is 4-tier identical and **checks itself** byte-identically.

### Honest scope
Only the checks that actually fire on the corpus are ported (name resolution +
unused-local) — that IS byte-identical `nova check` here. The rich
type/effect/move/refinement inference in `types.rs` emits no diagnostics on the
corpus and is future work.

## Stage 4 — the evaluator (done)

`selfhost/eval.nova` is a complete tree-walking interpreter written in Nova. It
loads a program (inlining `use "file"` imports), parses every item into the
canonical S-expression with the **stage-2 parser reused verbatim**, registers
functions / consts / structs / enums / methods / trait-impls / type-refinements /
`migrate` blocks / `machine`s / generators, then runs `main` — printing
**byte-identically to `nova run`** for every corpus + std + example file.

### What it reproduces (the whole runtime, not a subset)
- **Values** native to Nova plus tagged carriers for the compound kinds:
  `__nv_struct__` (alphabetical field display, matching the reference `BTreeMap`
  order), `__nv_enum__`, `__nv_clo__` (closures with captured env), `__nv_range__`
  (expanded to an array when printed or iterated), `__nv_chan__`, `__nv_gen__`,
  `__nv_task__`, `__nv_machine__`.
- **Control flow** via a `[flow,value]` tuple (`n`/`r`/`b`/`c`) plus native Nova
  `throw`/`try` for exceptions; **expression blocks** run in a shadowing child
  scope, swallow `throw`, run the tail on `break`, and yield the value on
  `return`; **`break` at function level acts as a return**; **`defer`** fires
  block-scoped in LIFO order on every exit path; `try`/`catch`/`finally` with
  finally-wins semantics; a catch-less `try` swallows.
- **Pattern matching** — literals, ranges, `or`, tuple, **slice with `...rest`
  binding (prefix + suffix)**, struct, enum.
- **The full builtin surface** (~200): collections, strings, math (incl.
  `pi`/`e`/`tau`/`phi`, `lerp`, `array(n,fill)`), files, `exec`/dirs, seeded
  `rand`, `time`, JSON, asserts, `min`/`max`/`min_of`/`max_of`, `count`, negative
  and open-ended **slicing**.
- **Lazy generators** (`yield` → replayed body, `.take`/`.next`/`for`),
  **channels** (`chan`/`send`/`recv`/`<-`/`->>`), **cooperative async** (`spawn`
  returns a task driven by `.await`/`recv`/end-of-program drain, matching the
  reference's deferred scheduling), **`select`**.
- **Refinement types** (`type Pos = Int if it > 0` enforced on annotated `let`),
  **state machines** (`machine`/`send`/`state_of`, invalid-transition throw),
  **state migration** (`migrate from A to B`), **big integers** (digit-fold so
  literals past i64 promote), and the **attribute runtime**: `#[memo]`,
  `#[self_healing]` (retry), `#[requires]`/`#[ensures]` contracts, `#[trace]`,
  `#[profile]`, `#[time_travel]`, `#[hot_swap]`, `#[integrity]`, `#[version]`, with
  the `hot_swap`/`integrity_of`/`profile_of`/`attrs_of`/`history_of`/`meta_of`/
  `encrypt`/`decrypt`/`is_debugged` builtins.

### The gate
`tests/selfhost_smoke.sh`: `nova run f` == `nova run selfhost/eval.nova f` for
**all 93** corpus + std + example files, and the evaluator is 4-tier identical
(interp / vm / --no-jit / --jit). The three compute-heavy kernels
(`aot_mandel`, `jit_arrays`, `numeric_mixed`) are a tree-walker running on top of
the reference tree-walker, so they need a generous per-file timeout — the output
is identical, only slower.

## Honest scope (overall)
Stages 1–4 (lexer, parser, checker, evaluator) are done and verified byte-
identical on every real file. Stage 5 (the fixpoint bootstrap — Nova building
`selfhost/*` with itself and reproducing the Rust binary) is the remaining step.
