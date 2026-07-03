# Nova — Handoff v3.25 (for the new chat)

Nova is a programming language implemented in Rust. This file is the single
source of truth for a fresh session continuing the work.

## Where the code lives — READ THIS FIRST
All work is now committed to git on branch **`claude/nova-v3-3-continuation-5xl8hk`**
(repo `payamahadpoor08-beep/novalang`, code under `nova/`). Earlier sessions kept
everything in an ephemeral scratchpad; it is now version-tracked so a new chat can
`git fetch` it.

⚠️ **Pushing is currently blocked.** This session's GitHub integration is
read-only: `git push` returns *"Permission denied"* and the GitHub API returns
*"403 Resource not accessible by integration"*. The commits below exist **locally
only**. To get them onto GitHub, the repo owner must grant the Claude GitHub App
**write/contents** permission for `novalang`, then a single `git push -u origin
claude/nova-v3-3-continuation-5xl8hk` ships everything. Until then, the migration
vehicle is the tarball `nova-lang-v3.25.tar.gz` (download → re-upload to the new
chat → `tar xzf`).

Local commits (newest first): LLVM boxed backend · systems layer · lambda/leak
fixes · diagnostics overhaul · v3.24 baseline import.

## Execution tiers (all byte-identical on every program, verified)
1. **Interpreter** (`src/interp.rs`, the oracle) — `nova run`
2. **Bytecode VM** (`src/bytecode.rs`) — optimizer, pooled frames
3. **Tiered Cranelift JIT** (`src/jit.rs`) — default in `nova vm`; i64+f64 tracks,
   deopt-to-VM, threshold 100, `--jit-stats`
4. **Embed builds** (`src/build.rs`) — `nova build` = standalone binary; works for
   every program
5. **AOT native** (`src/aot.rs` + `runtime/nova_rt.c`) — `nova build --aot[=c|llvm]`.
   Two tiers: `typed` (pure i64/f64 unboxed) and `boxed` (strings/arrays/slices/
   f-strings/for-each via a from-scratch refcounted runtime). **Both the C and the
   LLVM backends now do both tiers.** C #includes the runtime; LLVM emits textual
   IR calling it through clang's `NV` ABI (value = `(i8 tag, i64 payload)`, return
   = `{i8,i64}`) and compiles the `.ll` + `nova_rt.c` together (`-Dstatic=`). Every
   AOT binary is byte-diffed vs `nova run` before shipping; anything non-AOT-able
   or divergent falls back to embed automatically.

## What changed this session (v3.25)
- **Diagnostics overhaul** (`src/diag.rs`, `main.rs`, `types.rs`, `parser.rs`):
  runtime + checker errors now render a modern caret frame (source line + `^` +
  `--> file:line:col`); the gradual checker now checks **argument types** against
  declared concrete parameter types (Str→Int is an error), staying gradual
  (Unknown/numeric-widening allowed); pest syntax errors use human names
  ("an expression") not internal rule ids.
- **Systems layer** (`src/interp.rs`, `src/types.rs`): `exec(cmd,[args]) ->
  {code,stdout,stderr}`, `list_dir` (sorted), `mkdir`, `cwd`, `chdir`, `now_ms`,
  `sleep_ms`, `setenv` — alongside the existing args/env/read_file/write_file/
  file_exists/remove_file/read_line/input/exit/to_int/to_float/chr/ord. Effectful/
  non-deterministic ⇒ excluded from JIT/AOT eligibility ⇒ ship via embed.
- **Quirk fixes**: block-bodied lambdas now return their trailing expression
  implicitly (parser); boxed-AOT `main()` now releases its locals (valgrind-clean).
- **LLVM boxed backend** (see tier 5) — parity with C on the corpus.

## Written in Nova itself
- `std/` — 7 tested modules (list, sort, mathx, strx, ds, func, json), 34 test
  blocks, all green via `nova test std/<m>.nova`.
- `examples/apps/` — `wc.nova` (verified vs coreutils), `todo.nova` (JSON-backed
  CLI). `tests/corpus/` — 19 differential programs incl. `sys_layer.nova`.

## Verification status (all green)
78 Rust tests · 0 warnings · corpus 19/19 ×4 VM modes · AOT tiers (both backends):
1 typed + 4 boxed native + 14 embed-fallback, 0 byte-diffs · LLVM boxed binaries
valgrind-clean (0 errors/leaks) · stdlib 34/34 · example sweep 43 run==vm + 6
documented interp-only VM fallbacks · fmt idempotent.

## Build & verify
```bash
cd nova
cargo build --release && cargo test --release      # expect 0 warnings, 78 pass
bash tests/run_corpus.sh                            # 19/19 x4; prints AOT tiers
for m in list sort mathx strx ds func json; do ./target/release/nova test std/$m.nova; done
./target/release/nova build --aot=llvm tests/corpus/aot_strings.nova   # boxed LLVM native
```

## Known limits (documented, consistent across tiers)
- The VM/JIT don't compile try/catch, state machines, refinements, generators,
  async — those `main`s fall back to the interpreter (a clear message, not a bug).
- AOT float printing of extreme magnitudes (e.g. `1e301`) differs from Rust's
  Display in `nova_rt.c`'s `fmt_f64`; such programs fall back to embed on BOTH
  AOT backends (byte-diff gate). A future runtime `fmt_f64` improvement would let
  them go native — a good next task.

## Standing constraints
No toy/stub code; user functions > line counts; compile+test+sweep before
claiming success; honest reporting. Minimal comments. The old `sk-...` key that
leaked in an early paste should be rotated.
