# Nova — Handoff v3.27 (for the next session)

Supersedes HANDOFF4.md. Branch `claude/interpreter-compiler-review-t59e7u` of
`payamahadpoor08-beep/novalang`, code under `nova/`, pushed.

## Headline this session: kernels are now C-competitive
A cross-language benchmark (`nova/bench/`, correctness-gated) exposed that Nova's
compute kernels ran slower than Python. Root causes found and fixed:

1. **Once-called kernels never JIT'd.** `TieredJit::warm_loops` now compiles every
   eligible loop-containing function up-front. sieve (<2M) `nova vm`: **858ms → 49ms**.
2. **No track covered mixed int/float functions.** New unified numeric track
   (`src/jit.rs::NumGen`, `numeric_eligible_set`, `NumCheck`) compiles functions that
   mix int counters/accumulators with float math and return int or float. mandelbrot
   (600²×200) `nova vm`: **4934ms → 65ms (76×)** — beats Java/Node/Lua/Python, ~1.3× C.

Benchmark standing (CI machine, ms, lower better):
| workload | C | Nova aot | Nova vm | Java | Node | Lua | Python |
|---|--:|--:|--:|--:|--:|--:|--:|
| fib(32) | 6 | 7 | 19 | 46 | 54 | 129 | 250 |
| sieve <2M | 7 | 54 | 52 | 71 | 45 | 124 | 34 |
| mandel 600²×200 | 50 | 65 | 65 | 94 | 89 | 538 | 3816 |

Also this session: bytecode VM operand-fused superinstructions (BinLL/BinLC);
interpreter Vec-scopes + frame/arg pools + Int/Float fast-paths + FNV maps + LTO
(~1.7× on call-heavy code — a tree-walker's honest ceiling; the fast path is `nova vm`).

## Architecture note — the numeric track
`NumGen` shares the i64 track's ABI (`deopt_ptr` + i64 args → i64; f64 results as raw
bits). All params are ints, so the VM dispatches it like the i64 track (all-Int args)
and reads the result back as Int or Float via `TieredJit::num_ret_is_float`. It only
claims functions that neither the i64 nor f64 tracks do — additive, existing tracks
untouched. Deopt guards (overflow/div0) keep re-runs byte-identical.

## Verification (all green)
122 Rust tests (5 new numeric) · corpus 24/24 × 4 VM modes (new `numeric_mixed.nova`) ·
52-file sweep × 4 modes, 0 diffs · stdlib 47/47 · 0 warnings · `bash bench/run.sh`
correctness gate green.

## Honest open gaps (designs in docs/ROADMAP.md)
- **Interpreter tree-walker floor**: `nova run` won't beat CPython on tight loops; it is
  the semantic oracle by design. Real speed = `nova vm` / `--aot`.
- **AOT typed tier** doesn't yet emit the numeric/mixed kernels natively — they ship via
  the embed build, which runs the tiered VM (so they still get the 65ms JIT speed), not
  a standalone typed binary. Extending `aot.rs::emit_typed` to mixed int/float is next.
- Generics element-types in `types.rs` (§2), Rc→arena (§1), WASM/ARM/x32/mobile (§4),
  FFI, self-hosting — all still design-only.

## Build & verify
```bash
cd nova
cargo build --release && cargo test --release     # 0 warnings, 122 pass
bash tests/run_corpus.sh                           # 24/24 x4 + AOT census
bash bench/run.sh                                  # cross-language table
```
