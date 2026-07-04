# Nova — Handoff v3.26 (for the next session)

Nova is a programming language implemented in Rust. This file supersedes
HANDOFF3.md as the single source of truth for a fresh session.

## Where the code lives
Branch **`claude/interpreter-compiler-review-t59e7u`** of
`payamahadpoor08-beep/novalang`, code under `nova/` — **pushed to GitHub**
(the v3.25 push blockage is resolved; the old tarballs are gone from the repo).
Commits this session: v3.25 import · VM exceptions/defers/flow+peephole ·
JIT pow/f64-%/**\/mixed/local-arrays · fmt_f64 Rust-parity · docs.

## Execution tiers (all byte-identical on every program, verified)
1. **Interpreter** (`src/interp.rs`, the oracle) — `nova run`
2. **Bytecode VM** (`src/bytecode.rs`) — now also natively runs try/catch/
   finally/throw/defer/break-with-value (Op::Try sub-runs + runtime defer
   stack), tracks source positions (Op::Pos) for byte-identical located
   errors, and fuses superinstructions (IncLocal, LocalsBinJf). Only `yield`
   and refinement-typed `let` fall back (documented message).
3. **Tiered Cranelift JIT** (`src/jit.rs`) — i64 track (+ native `**`, + local
   integer arrays over a thread-local arena with deopt on OOB/empty-pop),
   f64 track (+ `%`, `**` via Rust libcalls, integer for-ranges, mixed
   int/float via static FKind analysis). Deopt-to-VM unchanged; arena resets
   per top-level `raw_call`, keeping re-runs pure.
4. **Embed builds** (`src/build.rs`) — works for every program.
5. **AOT native** (`src/aot.rs` + `runtime/nova_rt.c`) — typed/boxed tiers on
   BOTH C and LLVM backends; `fmt_f64` now matches Rust `Display` exactly
   (2,004,442-value differential fuzz, 0 mismatches), so float-printing
   programs ship native (census: typed=1 boxed=5 embed=17 on the corpus).

## What changed this session (v3.25 → v3.26)
- **VM**: native exceptions/defers/flow; fixed 6 real VM≠interp divergences
  (flat block scoping, expression-block flow/write semantics, break/continue
  outside loops, sentinel leak on uncaught throw); position tracking; peephole
  fusion (~10% faster pure VM); example sweep is now 52 files × 4 modes,
  0 diffs, single documented fallback (refinements).
- **JIT**: int pow (checked_pow transcription), f64 %/**, mixed int/float
  track, local-array track (sieve 465ms → 170ms tiered).
- **Runtime**: fmt_f64 rewritten (Rust never prints e-notation; shortest
  round-trip digits in plain decimal; ryū tie-rounding via exact expansion);
  fuzz harness in scratchpad proved 0/2M mismatches; valgrind still clean.
- **Docs**: `docs/ROADMAP.md` (bilingual problem/solution inventory —
  READ THIS for what's next), `docs/BUILD.md`, README Phase 10 section.

## Verification status (all green)
117 Rust tests · 0 warnings · corpus 23/23 × 4 VM modes · AOT census
typed=1 boxed=5 embed=17, byte-diff gate on both backends · boxed binaries
valgrind-clean · stdlib 34/34 · 52-file example sweep × 4 modes, 0 diffs.

## Build & verify
```bash
cd nova
cargo build --release && cargo test --release   # 0 warnings, 117 pass
bash tests/run_corpus.sh                        # 23/23 x4; AOT tier census
for m in list sort mathx strx ds func json; do ./target/release/nova test std/$m.nova; done
./target/release/nova vm --jit-stats --jit-threshold=3 tests/corpus/jit_arrays.nova
```

## Known limits (documented, consistent across tiers)
- VM: `yield` (generators) and refinement-typed `let` are interp-only.
- JIT arrays must be local and non-escaping; escaping arrays/structs/strings
  are the designed next step (docs/ROADMAP.md §3).
- AOT: try/catch, generators, closures beyond the boxed subset fall back to
  embed (design for setjmp/longjmp handlers in ROADMAP §6).
- VM-native runtime errors are located via Op::Pos at statement granularity;
  exotic cases relying on sub-expression cursor drift may differ (none seen
  in corpus/sweep).

## Suggested next steps (from docs/ROADMAP.md, in order of value)
1. Interpreter environment overhaul: slot resolution + free-var closure
   capture + Map as a real hash map (§1).
2. Generics: `Ty::Array(Box<Ty>)`/`Map(k,v)`/`Var` + small gradual unifier (§2).
3. JIT: escaping arrays, scalar-replaced structs, f64 arrays (§3).
4. AOT try/catch via setjmp/longjmp in nova_rt.c (§6); WASM via
   `clang --target=wasm32-wasi` when a wasi-sysroot is available (§4).
5. stdlib: setx/re/datetime/fmt/csv/pathx/heap with test blocks (§8).

## Standing constraints
No toy/stub code; compile+test+sweep before claiming success; honest
reporting; the interpreter is the oracle and every tier ships byte-identical
or falls back. The old `sk-...` key that leaked in an early paste should be
rotated.
