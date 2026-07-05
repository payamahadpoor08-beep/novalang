# Nova — feature reality audit (v3.28)

An honest, source-verified answer to "which grammar features are actually
implemented, and which only parse?" Status is derived from the code, not the
marketing table. Legend:

- **Run** = executes correctly on `nova run` (and, where noted, VM/JIT/AOT).
- **Check** = a real static analysis in `types.rs` (parsed + enforced).
- **Parse** = the grammar accepts it and builds an AST node, but nothing
  executes or enforces it — a no-op today.
- **Absent** = not in the grammar.

## Core language — Run ✅ (implemented and executed)
| feature | status | where |
|---|---|---|
| structs + methods (`impl`) | Run | interp.rs `make_struct`, `call_method_vals` |
| tuple structs / `data C(...)` | Run | parser + interp |
| enums with payloads + `match`/`=>` + guards | Run (VM-native) | interp `match_pattern`, bytecode `Op::MatchTest` |
| closures / lambdas `x => ...` (+ block body) | Run (VM-native) | interp `call_closure`, bytecode lambda chunks |
| generics `[T]` + trait bounds `[T: Trait]` + `where` | Check | types.rs `FnSig`, bound checks, return substitution |
| traits + `impl Trait for T` (default/required methods) | Run | interp `methods`, trait defaults |
| type aliases `type X = Y` | Run | parser/interp |
| refinement types `type Pos = int if it > 0` | Run | interp `refinements` — predicate enforced on `let` |
| effect system `-> T ![io]` | Check | types.rs `check_effects` (declared-vs-used) |
| ownership: linear/affine/move | Check | types.rs `check_moves` (use-after-move, move-in-loop) |
| async/await/spawn, channels, `select` | Run | interp scheduler, `Future`/`Task`/`Channel` |
| state machines (`machine`, `initial`, transitions, `send`/`state_of`) | Run | interp `machines` |
| macros (hygienic `$param`, `*`/`+`/`?`) | Run (partial) | parser macro expander |
| defer / try / catch / finally / throw | Run (VM-native) | interp + bytecode `Op::Try`/defers |
| loops (`while`/`loop`/`for range`/`for each`), break/continue | Run (VM-native) | interp + bytecode |
| `??` null-coalescing, ranges, slices, comprehensions | Run | interp + bytecode |
| f-strings, tagged strings (`json"..."`), raw strings | Run | parser/interp |
| collections: array/map/set literals, `[0; n]` fill | Run | interp + bytecode |
| lazy streams: generators (`yield`) + `stream[T]` return type, pulled by `for x in` | Run | interp `gen_produce`, `Value::Generator`; `tests/corpus/stream_lazy.nova` |
| stdlib (math/strings/arrays/random/time/json + list/sort/mathx/strx/ds/func/json/setx/fmtx/datex) | Run | `std/*.nova`, builtins |

## Numeric performance (v3.27) — Run ✅
| feature | status |
|---|---|
| tiered Cranelift JIT: i64, f64, **mixed int/float** tracks | Run — mandelbrot 65ms (was 4934ms) |
| local integer arrays JIT'd (arena) | Run — sieve 49ms (was 858ms) |
| **local int-field structs JIT'd (arena slots)** | Run — struct kernel 0.25s vs 2.84s pure-VM (~11×); aliases share the handle; escapes stay interp/VM |
| AOT native (C + LLVM), **pure-int/float** kernels | Run — fib native 7ms ≈ C |
| AOT native for **mixed/array** kernels | **Parse→embed** — see gap #1 below |

## Attributes — now real (v3.28, Phase 1) ✅
Attributes are no longer discarded; these carry tested semantics on every tier
(see `docs/ATTRIBUTES.md`, `tests/corpus/attributes.nova`):
| attribute | status |
|---|---|
| `#[zero_alloc]` | Check — `nova check` errors if the function allocates |
| `#[self_healing(attempts: N)]` / `#[retry(attempts: N)]` | Run — retries the call on runtime error |
| `#[hot_swap]` + `hot_swap(name, closure)` | Run — runtime body replacement |
| `#[integrity]` + `integrity_of(name)` | Run — stable tamper-detection hash |
| `#[memo]` / `#[memoize]` | Run — result cache keyed by args |
| `#[requires]` / `#[assumes]` / `#[ensures]` | Run — design-by-contract checks (real predicate exprs) |
| `#[trace]` / `#[log]` / `#[audit]` | Run — prints `name(args) -> result` per call |
| `#[profile]` + `profile_of(name)` | Run — call counting |
| `#[deprecate]` / `#[deprecated]` | Run — one-time warning on use |
| `#[time_travel(depth: N)]` + `history_of(name)` | Run — bounded ring buffer of past results (snapshot/rollback) |
| `#[encrypt]` + `encrypt(s,key)`/`decrypt(s,key)` | Run — keyed XOR cipher (obfuscation-grade, documented) |
| `#[anti_debug]` + `is_debugged()` | Run — best-effort Linux TracerPid debugger detection |
| `#[anti_tamper]` | Run — verifies the function body hash hasn't changed since first call |
| `#[hot]` / `#[cold]` | Run — `hot` warms the JIT up-front, `cold` prevents warming |
| `#[simd]` | Run — JIT hint: eagerly compiles the numeric/array kernel up-front (like `#[hot]`). True Cranelift SIMD-type vectorization is a documented future deepening — the attribute honestly means "compile this kernel now", not "it is vectorized". |
| `#[obfuscate]` + `nova obfuscate <file>` | Run — alpha-renames a function's local identifiers to opaque names; behaviour byte-identical (`tests/obfuscate_smoke.sh`). Source obfuscation, not encryption. |
| `#[comptime]` (no-arg fn) | Run — const-evaluated once before `main`; every call returns the cached value |
| metadata (`#[version]`, `#[since]`, `#[throws]`, `#[intent]`, `#[deps]`, …) + `meta_of(name,key)` | Run — captured + queryable |
| **any attribute** + `attrs_of(name)` | Run — all attributes captured + introspectable |

## Parse-only ⚠️ (accepted syntax, NOT yet executed/enforced)
These build AST nodes but currently do nothing at runtime — the honest truth:
| feature | status |
|---|---|
| Higher-Kinded Types `[T[_]]` | Parse only (checker erases to Unknown) |
| associated types (`type Item;` in traits) | Parse only |
| effect polymorphism `![E]` | Parse only (monomorphic effects only) |
| AST quasiquotation `ast!{...}` / procedural macros | Parse only |
| `#[polymorph]` | **Parse only — no-op.** In a tree-walker, random dispatch among semantically-identical clones is a no-op by construction; it is properly an AOT-codegen concern (emit N equivalent C variants) and is deferred to that phase — deliberately not faked. |

## Absent ❌ (not in grammar, despite the table)
`union` types are not in the grammar and not implemented.

## Tooling status
| tool | status |
|---|---|
| REPL, `nova run/vm/check/test/doc/fmt/obfuscate`, disasm | Run ✅ |
| **memory safety** — AOT binaries valgrind-clean (0 definite leaks / 0 errors) across boxed + embed tiers | Verified ✅ — `tests/valgrind_smoke.sh` |
| **daemon mode** (`nova daemon`) — persistent service, `load`/`reload`/`run`/`funcs`/`stats` | Run ✅ |
| **incremental compilation** — `reload` re-parses and reuses unchanged functions, reporting exactly what changed | Run ✅ |
| **hot reload** — `run` after `reload` executes new code without restarting the daemon | Run ✅ |
| predictive compilation — the tiered JIT warms a hot function's whole callee closure ahead of need | Run (heuristic) |
| **state migration** (`migrate from Old to New { ... }` + `migrate(value)`) | Run ✅ — see `docs/MIGRATION.md` |
| LSP, package manager | Not implemented (design only) |
| **WASM target** (`nova build --aot=wasm`) — typed + boxed | Run ✅ — compiles the portable AOT C (incl. `nova_rt.c`) to `wasm32-wasi` via clang + a wasi-libc sysroot, shipped only if byte-identical to `nova run` under node's WASI (`tests/wasm_smoke.sh`). Strings/arrays included; only embed-tier programs are excluded. |
| **ARM target** (`nova build --aot=arm`) — typed + boxed | Run ✅ — cross-compiles the portable AOT C (incl. `nova_rt.c`) to a static aarch64 binary via `aarch64-linux-gnu-gcc`, shipped only if byte-identical to `nova run` under `qemu-aarch64` (`tests/arm_smoke.sh`). For Raspberry Pi / aarch64 mobile. |
| 32-bit / other mobile targets | Not implemented (same cross-compile pattern, add on request) |

## The three real gaps that matter for "AOT/speed"
1. **AOT native for mixed/array kernels.** `nova build --aot` compiles pure-int
   functions to a standalone native binary (fib), but sieve (arrays) and mandel
   (mixed int/float) fall back to the **embed** build — the interpreter+VM
   wrapped in a binary. They still run at JIT speed (65ms), but they are not
   yet true standalone native code. Fix: extend `aot.rs::emit_typed` (the C
   text emitter) to per-variable `int64_t`/`double` + arrays, mirroring the
   JIT's `NumGen`/array track, plus a Rust-`Display` float printer.
2. **Full generics in `types.rs`**: element types (`[T]`) are erased to Unknown
   (design in ROADMAP §2).
3. **Interpreter `Rc`/tree-walking floor**: `nova run` is the oracle, not a
   speed tier; use `nova vm`/`--aot`.

---

خلاصه فارسی: این سند صادقانه می‌گوید کدام فیچرها واقعاً کار می‌کنند و کدام فقط
parse می‌شوند. هسته‌ی زبان (struct/enum/match/closure/async/channel/machine/
refinement/effect-check/move-check/macro/generic-با-bound/JIT/AOT) واقعاً پیاده و
اجرا می‌شود. اما تمام attrهای «امنیتی/پیشرفته» (`#[simd]`، `#[encrypt]`،
`#[self_healing]` و...) فقط parse می‌شوند و **هیچ کاری نمی‌کنند** — no-op. همچنین
اکنون `#[comptime]` (ارزیابی زمان‌کامپایل، یک‌بار پیش از main) و state migration
(`migrate from Old to New`) هم واقعاً کار می‌کنند و byte-identical در همه‌ی tierها هستند.
اما HKT، union، associated types، stream، LSP/package-manager و
WASM/ARM هنوز پیاده نشده‌اند. AOT برای کرنل‌های mixed/array فعلاً native نمی‌سازد و
به embed برمی‌گردد (با سرعت JIT اجرا می‌شود ولی باینری مستقل native نیست) — این
گام بعدی واقعی است.
