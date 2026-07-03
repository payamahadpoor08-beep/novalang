# Changelog

## v3.26.0 — 2026-07-03

The "no tier left behind" release: the VM and JIT close most of their coverage
gap with the interpreter, and float printing reaches exact Rust parity.

### Bytecode VM
- **Native exceptions and defers**: `try/catch/finally`, `throw`, `defer`, and
  `break`-with-value now compile to bytecode (`Op::Try` sub-runs + a runtime
  defer stack) instead of forcing the whole function onto the interpreter.
  Runtime errors are catchable exactly as in `nova run`; `finally` beats
  `return`; defers unwind LIFO per block, including during throws.
- **Six real VM/interpreter divergences fixed**, found by differential probing:
  flat statement-block scoping, expression-block flow consumption and write
  shadowing, `break`/`continue` outside loops, and an uncaught-throw sentinel
  leak.
- **Byte-identical located errors** via `Op::Pos` position tracking.
- **Peephole superinstructions** (`IncLocal`, fused compare-branch): the pure
  VM is ~10% faster than v3.25 despite the added position tracking.

### JIT (Cranelift, tiered)
- Native integer `**` (transcription of `i64::checked_pow`; overflow deopts to
  BigInt promotion, negative/huge exponents deopt to the float path).
- **Local integer arrays** on the i64 track: literals that never escape a pure
  function (index, assign, `len`/`push`/`pop`, aliasing) lower to a
  thread-local arena with deopt-guarded bounds. Hot sieve: 465ms → 170ms.
- f64 track: `%` and `**` (bit-identical via Rust libcalls), integer `for`
  ranges, and **mixed int/float arithmetic** via a static kind analysis.

### Runtime / AOT
- `fmt_f64` rewritten to match Rust's `Display` exactly (full decimal
  expansion, shortest round-trip digits, ryū tie rounding). Verified by a
  2,004,442-value differential fuzz: **0 mismatches**. Float-printing programs
  now ship native on both AOT backends, valgrind-clean.

### Docs & project
- `nova/docs/ROADMAP.md` (bilingual EN/فارسی problem/solution inventory),
  `nova/docs/BUILD.md`, `HANDOFF4.md`, root README, installer, CI workflow,
  CONTRIBUTING, issue templates.
- Three new stdlib modules: `std/setx.nova`, `std/fmtx.nova`, `std/datex.nova`.

**Verification:** 117+ Rust tests · corpus 23/23 × 4 VM modes · AOT census
typed=1 boxed=5 (byte-diff gated, both backends) · stdlib all green ·
52-file example sweep × 4 modes, 0 diffs · 0 compiler warnings.

## v3.25 — 2026-07-03 (pre-release state, imported from tarball)

- Diagnostics overhaul: caret frames for runtime/checker errors, argument-type
  checking in the gradual checker, human-readable parse errors.
- Systems layer: `exec`, `list_dir`, `mkdir`, `cwd`, `chdir`, `now_ms`,
  `sleep_ms`, `setenv`, and friends.
- LLVM boxed AOT backend reaches parity with the C backend; boxed `main`
  releases its locals (valgrind-clean).
- Block-bodied lambdas return their trailing expression.

## v3.3 and earlier

Interpreter, Pest grammar, gradual type checker, effects/ownership analyses,
bytecode VM phases 1–4, Cranelift JIT with tiering and f64 track, embed
builds, typed/boxed AOT tiers, stdlib, REPL, formatter, doc extractor.
