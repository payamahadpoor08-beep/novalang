# Nova — feature reality audit (v3.31)

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
| enums + **`union`** with payloads + `match`/`=>` + guards | Run (VM-native) | interp `match_pattern`; `union` lowered to enum runtime |
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
| associated types (`type Item;` + impl `type Item = X`), effect-poly `![E]`, HKT `[F[_]]` | Run (gradual) | parse + resolve to Unknown where not concrete; `tests/corpus/type_system_advanced.nova` |
| stdlib (math/strings/arrays/random/time/json + list/sort/mathx/strx/ds/func/json/setx/fmtx/datex) | Run | `std/*.nova`, builtins |

## Numeric performance (v3.27) — Run ✅
| feature | status |
|---|---|
| tiered Cranelift JIT: i64, f64, **mixed int/float** tracks | Run — mandelbrot 65ms (was 4934ms) |
| local integer arrays JIT'd (arena) | Run — sieve 49ms (was 858ms) |
| **local int-field structs JIT'd (arena slots)** | Run — struct kernel 0.25s vs 2.84s pure-VM (~11×); aliases share the handle; escapes stay interp/VM |
| AOT native (C + LLVM), **pure-int/float** kernels | Run — fib native 7ms ≈ C |
| AOT native for **local-int-array** kernels (sieve) | Run — true standalone native binary (typed tier), not embed |
| AOT native for **mixed int/float** kernels (mandel) | Run — true standalone native binary (typed tier, per-variable i64/double typing); 52ms |

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
| AST quasiquotation `ast!{...}` / procedural macros | Parse only |
| `#[polymorph]` | **Parse only — no-op.** In a tree-walker, random dispatch among semantically-identical clones is a no-op by construction; it is properly an AOT-codegen concern (emit N equivalent C variants) and is deferred to that phase — deliberately not faked. |

## Absent ❌
(none outstanding from the marketing table — `union` now implemented as a tagged union, lowered to the enum runtime; see `tests/corpus/union_types.nova`.)

## Tooling status
| tool | status |
|---|---|
| REPL, `nova run/vm/check/test/doc/fmt/obfuscate`, disasm | Run ✅ |
| **memory safety** — AOT binaries valgrind-clean (0 definite leaks / 0 errors) across boxed + embed tiers | Verified ✅ — `tests/valgrind_smoke.sh` |
| **daemon mode** (`nova daemon`) — persistent service, `load`/`reload`/`run`/`funcs`/`stats` | Run ✅ |
| **incremental compilation** — `reload` re-parses and reuses unchanged functions, reporting exactly what changed | Run ✅ |
| **hot reload** — `run` after `reload` executes new code without restarting the daemon | Run ✅ |
| **demon compiler** (`nova demon <file>` + `#[demon(mode/watch/cache/hot_reload/incremental/diagnostics/optimize)]`) — background daemon that file-watches (portable mtime poll), incrementally reparses only changed functions (intelligent per-function cache), streams live checker diagnostics, and hot-reloads `main` in-process | Run ✅ — `tests/demon_smoke.sh` |
| predictive compilation — the tiered JIT warms a hot function's whole callee closure ahead of need | Run (heuristic) |
| **state migration** (`migrate from Old to New { ... }` + `migrate(value)`) | Run ✅ — see `docs/MIGRATION.md` |
| **LSP** (`nova lsp`) — full IDE server: diagnostics, hover, completion, signatureHelp, goto-definition, references, documentHighlight, document/workspace symbols, rename, formatting, semanticTokens, foldingRange | Run ✅ — 11-capability suite in `tests/lsp_smoke.sh` |
| **package registry** (`nova add`/`remove`/`install`/`update`/`tree`/`publish`/`registry`) — real registry on `nova.hgx` `[dependencies]`: SemVer resolver (caret/tilde/wildcard/range) + reproducible `nova.lock`, `.nvpkg` archives, SHA-256 verification, an HTTP index server (`nova registry`) + client, registry/path/git sources, and **two-mode `[abilities]`** (attributes declarable in the manifest, merged onto code) | Run ✅ — full loop (publish→serve→resolve→fetch→verify→vendor→run byte-identical) in `tests/registry_smoke.sh`; local vendor path in `tests/pkg_smoke.sh`; see `docs/REGISTRY.md` |
| **WASM target** (`nova build --aot=wasm`) — typed + boxed | Run ✅ — compiles the portable AOT C (incl. `nova_rt.c`) to `wasm32-wasi` via clang + a wasi-libc sysroot, shipped only if byte-identical to `nova run` under node's WASI (`tests/wasm_smoke.sh`). Strings/arrays included; only embed-tier programs are excluded. |
| **ARM64 target** (`nova build --aot=arm`) — ARMv8/aarch64, typed + boxed | Run ✅ — cross-compiles the portable AOT C (incl. `nova_rt.c`) to a static aarch64 binary via `aarch64-linux-gnu-gcc`, byte-identical under `qemu-aarch64`. Modern phones / Raspberry Pi. |
| **ARM32 target** (`nova build --aot=arm32`) — ARMv7/armhf, typed + boxed | Run ✅ — static 32-bit ARM binary via `arm-linux-gnueabihf-gcc -marm`, byte-identical under `qemu-arm`. Older / weaker phones. Both arches gated by `tests/arm_smoke.sh`. |
| **self-hosting, stage 1: the Nova lexer in Nova** (`selfhost/lexer.nova` + `nova tokens` reference) | Run ✅ — byte-identical token dumps vs the Rust reference on all 95 files (including lexing itself), 4-tier identical; `tests/selfhost_smoke.sh`, `docs/SELFHOST.md`. |
| **self-hosting, stage 2: the Nova parser in Nova** (`selfhost/parser.nova` + `nova ast` reference) | Run ✅ — a complete recursive-descent + Pratt parser mirroring `nova.pest` **and** the `parser.rs` lowering (all desugarings: cast/pipeline/`??`/compound-assign/stream/macro-expansion-with-hygiene/implicit-return/…); canonical S-expression AST byte-identical to `nova ast` on all 96 files, **including parsing itself and the lexer**, 4-tier identical. |
| **self-hosting, stage 3: the Nova checker in Nova** (`selfhost/checker.nova` + `nova check` reference) | Run ✅ — name resolution (undefined-variable) + unused-local warnings with position-accurate `diag.rs` caret frames; byte-identical to `nova check` on all 96 files, **including checking itself**, 4-tier identical (`tests/selfhost_smoke.sh`, `docs/SELFHOST.md`). Only the checks that fire on the corpus are ported; full type inference is future. Stage 5 (fixpoint) honestly *not* claimed yet. |
| **self-hosting, stage 4: the Nova evaluator in Nova** (`selfhost/eval.nova` + `nova run` reference) | Run ✅ — a complete tree-walking interpreter in Nova (values, control flow + `defer`/`try`/`finally`, all patterns incl. slice-rest, ~200 builtins, lazy generators, channels, cooperative `spawn`/`select`/`await`, refinement types, state machines, migrations, big integers, and the full attribute runtime `#[memo]`/`#[self_healing]`/`#[requires]`/`#[trace]`/`#[time_travel]`/`#[hot_swap]`/…). Program output **byte-identical to `nova run` on all 93 corpus+std+example files**, 4-tier identical, and eval.nova is itself lexed/parsed/checked byte-identically by stages 1–3 (`tests/selfhost_smoke.sh`, `docs/SELFHOST.md`). |

## Remaining AOT notes
1. **(resolved) AOT native for mixed int/float kernels.** fib, sieve AND mandel
   now compile to true standalone native binaries. Historical note: `nova build --aot` now compiles
   pure-int functions (fib) AND **local-int-array kernels (sieve)** to standalone
   native binaries — the C emitter has a `NIA` int-array (the C twin of the JIT
   arena), so sieve is a true native `aot_sieve` (typed tier, byte-identical, also
   cross-compiles to arm/wasm). Still embed-only: **mixed int/float** in one
   function (mandel) — needs per-variable `int64_t`/`double` typing mirroring the
   JIT's `NumGen`. Mandel still runs at JIT speed via embed.
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
