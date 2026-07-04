# Nova — Problems, Solutions, and Roadmap (v3.26)

This document is the single honest inventory of Nova's known weaknesses, what was
**fixed in v3.26**, and the concrete design for everything still open. Each section
ends with a Persian summary (خلاصه فارسی).

Reference study: sections below cite design lessons taken from reading the Zig
compiler sources (`src/InternPool.zig`, `lib/std/heap/arena_allocator.zig`) and the
Rust compiler (`compiler/rustc_middle/src/mir`, `compiler/rustc_trait_selection`).

---

## 1. `Rc<RefCell>` in the interpreter → arena / better environments

**Real finding:** profiling and code reading show the dominant interpreter cost is
NOT the heap `Rc<RefCell>` values (those implement Nova's reference semantics and
are shared correctly with the VM) — it is the environment model:

- `Scope = HashMap<String, Value>` (`src/interp.rs:13`) is **cloned wholesale** on
  every call (`:1354`), every closure invocation (`:1553`), every expression block
  (`:2191`), and every match arm (`:2175`).
- Closures capture the **entire scope** (`:2100`), not their free variables.
- `Map` is an association list `Rc<RefCell<Vec<(Value,Value)>>>` (`:71`) — O(n) lookup.

**Improved in v3.26.1 (measured, byte-identical):**
- `Scope` is now an insertion-ordered `Vec<(Rc<str>, Value)>` instead of a
  `HashMap<String, Value>`: cloning a scope (per call/block/match-arm/closure) is a
  `Vec` copy with cheap `Rc` key bumps instead of rehashing and re-allocating every
  key string, and small scopes resolve variables by a short scan instead of hashing.
- Call frames and argument vectors are borrowed from free-list pools.
- The `call` dispatcher resolves a user function in a single map lookup, skipping the
  ~150-arm builtin match; `eval_binop` and the `Binary` arm gained an Int-op-Int fast
  path (which the VM's `Op::Bin` also benefits from); name tables use an FNV hasher;
  release builds use fat LTO + one codegen unit.
- Result: **fib(32) 2898→1707ms (≈1.7× on call-heavy code)**, loop-sum 704→566ms,
  closures 377→297ms; VM loop-sum 512→432ms.

**Honest ceiling:** a tree-walker's cost per node (recursive `eval` dispatch, `Value`
enum moves, `Result` propagation, `Rc` refcount traffic) is irreducible without
compiling to slots — which is precisely what the bytecode VM does. The 10×+ path is
therefore `nova vm`, not `nova run`: on fib(35) the interpreter is ~7.2s while the
tiered JIT is ~81ms (~89×). Squeezing the tree-walker further has diminishing returns;
the real next lever below (slot resolution) essentially rebuilds the VM's frame model
inside the interpreter.

**Design (next, for a larger jump):**
1. Compile-time slot resolution for the interpreter too: a resolver pass that maps
   every `Ident` to (depth, slot), turning `Scope` into `Vec<Value>` frames — this
   is exactly what `bytecode.rs`'s `FnCompiler::define/resolve` already does; reuse
   that resolver for both tiers.
2. Free-variable analysis for closures (capture only what the lambda mentions).
3. `Map` → insertion-ordered hash map (index-map: `HashMap<KeyHash, usize>` + a
   `Vec<(Value,Value)>`), keeping iteration order and the current equality rules.
4. *Zig lesson (`InternPool.zig`):* intern strings/identifiers once and address
   them by u32 index; `bytecode.rs`'s name pool already does this — extend it to
   the interpreter so scope keys become integers, not `String`s.
5. *Zig lesson (`arena_allocator.zig`):* the JIT's new thread-local array arena
   (v3.26, `src/jit.rs::JIT_ARENA`) shows the pattern works; AST nodes are the next
   candidate (allocate per-run, free all at once).

خلاصه فارسی: هزینه اصلی مفسر clone کامل HashMap محیط در هر فراخوانی است، نه
`Rc<RefCell>`. راه‌حل: اسلات‌های عددی به‌جای نام (همان کاری که VM می‌کند)، capture
فقط متغیرهای آزاد در closure، و تبدیل Map به hash-map واقعی. در v3.26 مسیر فرار
(VM) به‌قدری کامل شد که برنامه‌های خیلی کمتری اسیر مفسر کند می‌مانند.

---

## 2. Generics in `types.rs` → `[T]`, `where`, element types

**Current state:** `Ty` (`src/types.rs:13`) is `Int/Float/Str/Bool/Null/Array/Map/
Struct(name)/Func/Unknown` — `Array` and `Map` carry **no element types**, and
there is no type variable. `FnSig` already tracks `type_params` and `where_bounds`;
the checker validates trait bounds (`:580-599`) and substitutes generic returns
(`:601-607`). So `[T: Trait]` and `where` *exist* but are erased to `Unknown` at
every collection boundary.

**Design:**
1. `Ty::Array(Box<Ty>)`, `Ty::Map(Box<Ty>, Box<Ty>)`, `Ty::Var(u32)`.
2. A small unifier (`unify(a, b) -> Result<Subst>`): `Unknown` stays the gradual
   top; `Var` binds; mismatch at a concrete pair is an error. No HM inference —
   only propagate literals and declared annotations (gradual stays gradual).
3. Array literals infer `Array(join of element types)`; indexing yields the
   element type; `push(a, v)` checks `v` against it.
4. Generic instantiation: on call, seed `Var` for each `type_param`, unify
   argument types against parameter annotations, apply the substitution to the
   return type — the current `:601` special case becomes the general rule.
5. *Rust lesson (`rustc_trait_selection`):* keep trait solving as a plain
   obligation list checked per call site (Nova's bounds are just name sets — no
   need for a full solver); resist adding subtyping.

خلاصه فارسی: `Ty` فعلاً نوع عنصر آرایه/مپ را ندارد و متغیر نوعی وجود ندارد. طرح:
افزودن `Array(Box<Ty>)`، `Map(k,v)` و `Var`، یک unifier کوچک gradual، و تعمیم
جایگذاری نوع بازگشتی به قانون عمومی. bounds ها همین حالا چک می‌شوند.

---

## 3. "JIT only for numbers" → arrays, mixed int/float kernels, structs

**Fixed in v3.27 (major — the mandelbrot-class win):**
- **Unified numeric track** (`src/jit.rs::NumGen` / `numeric_eligible_set`): functions
  that mix integer loop counters/accumulators with float math and return an int **or**
  a float now compile to native code. Per-variable I64/F64 kinds, `as_f` promotion,
  `to_float`/`to_int`, mixed comparisons, overflow/div0 deopt, all on the i64 ABI
  (f64 results carried as bits). **mandelbrot 600²×200: `nova vm` 4934ms → 65ms (76×)**,
  now on par with C and ahead of Java/Node/Lua/Python.
- **Eager loop-kernel warming** (`TieredJit::warm_loops`): a compute kernel called once
  from `main` is compiled up-front instead of never crossing the call threshold.
  **sieve <2M on `nova vm`: 858ms → 49ms (10×)**.
- **Local integer arrays on the i64 track** (v3.26): arrays built from literals that
  never escape a pure function (indexing, `a[i]=v`, `len`/`push`/`pop`, aliasing) compile
  against a thread-local arena (`src/jit.rs::nova_arr_*`), with OOB/empty-pop deopt.
- **Native integer `**`** (transcription of `i64::checked_pow` with per-multiply
  overflow deopts; negative/huge exponents deopt to the interpreter's Float path).
- **f64 track**: `%` and `**` (bit-identical via Rust libcalls), integer `for`
  ranges with i64 counters, and **mixed int/float arithmetic** via a static FKind
  analysis (ints promote through the interpreter's `as_f` rules; anything whose
  int-ness is observable — Int÷Int, Int returns, Int==Int — stays off the track).

**Still open (design):**
1. Escaping arrays: return `Vec<i64>` through the ABI as (ptr,len) and box into
   `Value::Array` at the call site; needs ownership transfer out of the arena.
2. Simple structs: flatten fix-shaped, int-field structs into scalars (scalar
   replacement) when they never escape — same eligibility pattern as arrays.
3. Float arrays: a second arena of `Vec<f64>` mirroring the i64 one.
4. String track: intern read-only strings (Zig InternPool pattern) and support
   `len`/compare/concat-into-arena.

خلاصه فارسی: در v3.26 آرایه‌های محلی صحیح، توان صحیح native، باقیمانده/توان float
و ریاضیات مختلط int/float به JIT اضافه شد (sieve داغ ~۲.۷ برابر سریع‌تر). گام بعد:
آرایه‌های فرارکننده، structهای ساده (تخت‌سازی به اسکالر) و آرایه‌های float.

---

## 4. AOT only C/LLVM → WASM and ARM

**Current state:** two backends (portable C via `cc`, textual LLVM IR via `clang`)
share `runtime/nova_rt.c` and the byte-diff oracle gate (`src/build.rs:71`).

**Design (not implemented — needs toolchains):**
1. **WASM**: the C backend is the vehicle — `clang --target=wasm32-wasi` over the
   generated `.c` + `nova_rt.c`. The runtime is already freestanding C (malloc/
   printf only), so the WASI libc covers it. Gate: run the byte-diff oracle under
   `wasmtime`. Blockers: a `wasi-sysroot` and wasmtime in the build environment.
2. **ARM (aarch64)**: pure cross-compilation — `cc --target=aarch64-linux-gnu`
   (or zig cc, which bundles the sysroot — the pragmatic choice). Gate via qemu.
3. The `Tier`/oracle machinery needs no change: add a `--target=` flag threaded to
   the backend command and skip the gate (with a warning) when no emulator exists.

خلاصه فارسی: مسیر WASM از بک‌اند C با `--target=wasm32-wasi` می‌گذرد و ARM با کراس
کامپایل (ترجیحاً `zig cc` که sysroot همراه دارد). ساختار Tier/oracle تغییر نمی‌خواهد؛
فقط toolchain و emulator برای گیت بایت‌به‌بایت لازم است.

---

## 5. VM coverage (try/catch, defer, …) — FIXED in v3.26

The statement gate at `bytecode.rs` that forced whole functions onto the
interpreter for `try/catch/throw/defer/break-value` is gone:

- `Op::Try` transcribes the tree-walker's TryCatch logic over sub-runs of the same
  frame (body/catch/finally ranges, handler stubs for flows that continue outward).
- Defers register at runtime (`PushDefer`) and unwind LIFO per block, exactly like
  `exec_block`'s deferred list — including during throw unwinding.
- Runtime errors become catchable `Str` values only when a handler exists;
  `finally` wins over `return`; YIELD unwinding passes through — all transcribed
  from the interpreter arm by arm.
- Differential probing found and fixed **six real VM≠interp divergences**:
  statement blocks now share the flat function scope; expression blocks consume
  every `Flow` (return yields the block's value; break/continue/throw fall through
  to the tail) and shadow writes; break/continue outside loops behave like the
  call boundary; uncaught throws no longer leak the internal sentinel.
- `Op::Pos` mirrors `cur_pos`, so located runtime errors print byte-identically.
- New peephole superinstructions (`IncLocal`, `LocalsBinJf`) make the pure VM
  ~10% faster than v3.25 despite position tracking.

**Still interp-only:** `yield` (generators replay through the tree-walker), and
`let` with refinement-type annotations (predicates run in the interpreter; such
functions fall back with the documented message).

خلاصه فارسی: try/catch/finally/throw/defer/break-با-مقدار حالا کاملاً در VM اجرا
می‌شوند؛ شش واگرایی واقعی VM/مفسر هم پیدا و رفع شد؛ خطاهای مکان‌دار بایت‌به‌بایت
یکسان چاپ می‌شوند و VM خالص ~۱۰٪ سریع‌تر شد. فقط `yield` و refinementها مفسری ماندند.

---

## 6. AOT float-extreme fallback — FIXED in v3.26

Rust's f64 `Display` never prints e-notation; the old `%g`-based `fmt_f64` could
never match `1e301`. Rewritten (`runtime/nova_rt.c`): integral values print via
`%.1f` at any magnitude; others find the shortest round-tripping digits with a
`%.*e` probe and lay them out in plain decimal; exact-midpoint ties (glibc rounds
even, Rust's ryū rounds up) are detected from the full exact expansion. Verified
by a 2,004,442-value differential fuzz (extremes, denormals, bit-neighbors of
every power of ten, 2M random bit patterns): **0 mismatches**. `float_edges` and
`aot_float_print` now build native on BOTH backends, byte-identical, valgrind-clean.

**Still open for AOT:** try/catch (design: `setjmp/longjmp` handler stack in
`nova_rt.c` + a `nv_throw` that unwinds to the innermost handler; finally lowered
as code duplication on both edges), generators, closures beyond the current boxed
subset. The embed fallback keeps all of these correct today.

خلاصه فارسی: چاپ float در C بازنویسی شد و با ۲ میلیون مقدار فازتست، صفر اختلاف با
Rust دارد؛ برنامه‌های float-extreme حالا در هر دو بک‌اند native و valgrind-تمیز
هستند. برای try/catch در AOT طرح setjmp/longjmp مستند شد.

---

## 7. C runtime polish (memory safety, leaks)

**Current state:** refcounted `NV` values; boxed binaries valgrind-clean on the
corpus (0 errors, 0 leaks) — re-verified after the fmt_f64 rewrite in v3.26.

**Design (hardening):**
1. A `NOVA_RT_DEBUG` build with refcount audit (counts of alloc/release per type,
   abort on double-release / use-after-free via canaries).
2. An ASan CI leg for the AOT corpus (`-fsanitize=address,undefined`).
3. Overflow-checked `sb_put` growth and allocation-failure paths (today `malloc`
   results are unchecked — fine for scripts, not for a real runtime).

خلاصه فارسی: باینری‌های boxed در valgrind تمیزند؛ گام بعد: بیلد دیباگ با ممیزی
refcount، اجرای ASan/UBSan روی corpus و چک‌کردن نتیجه mallocها.

---

## 8. stdlib gaps

**Current state:** 7 tested modules (list, sort, mathx, strx, ds, func, json),
34 test blocks, all green.

**Plan (each lands with `test` blocks, pure Nova):**
`setx` (set ops over maps) · `re` (literal/star matcher, ~100 lines) · `datetime`
(civil date arithmetic over `now_ms`) · `fmt` (pad/align/thousands) · `csv`
(parse/emit with quoting) · `pathx` (join/dirname/ext) · `rand` helpers (shuffle/
sample over the builtin RNG) · `heap` (binary heap; also a good JIT-array benchmark).

خلاصه فارسی: هفت ماژول سبز داریم؛ فهرست بعدی: set، regex سبک، تاریخ/زمان، fmt،
csv، path، ابزارهای rand و heap — همگی با بلوک‌های test و به زبان خود Nova.

---

## 9. async / complex match / ownership / FFI / self-hosting

- **async**: cooperative scheduler exists (spawn/await/channels/select as
  delegated expressions). Design: state-machine lowering per async fn (the
  `machine` item is the natural IR), then VM-native resume ops. Until then they
  run correctly through delegation.
- **complex match**: `match` is VM-native (pattern tests delegated per-arm).
  Design: compile literal/enum-tag arms to a jump table; keep slice/struct
  patterns delegated.
- **ownership**: `check_moves` (`types.rs:890`) covers use-after-move and
  move-in-loop; borrows/lifetimes are out of scope by design (gradual language).
- **FFI**: `extern fn` parses but errors at call time. Design: `libloading`-based
  `extern "C"` for i64/f64/str signatures only, behind an `--allow-ffi` flag
  (deterministic corpus stays FFI-free).
- **self-hosting**: distance is large (needs FFI or a bytecode serializer first).
  Realistic next step: the `nova fmt` formatter rewritten in Nova as a milestone.

خلاصه فارسی: async از مسیر delegation درست کار می‌کند و طرح lowering به ماشین حالت
دارد؛ match عمدتاً native است؛ ownership در حد move-checking باقی می‌ماند؛ FFI با
libloading برای امضاهای ساده طراحی شد؛ self-hosting فعلاً دور است و اولین قدم
واقع‌بینانه بازنویسی fmt به خود Nova است.

---

## 10. Documentation debt

v3.26 adds this ROADMAP, `docs/BUILD.md` (AOT pipeline), HANDOFF4, and inline
comments on every new complex mechanism (VM exception machinery, defer stack,
JIT tracks/arena, fmt_f64 algorithm). Still to do: a language reference manual
(one page per construct, generated from `nova doc`), and comment passes over
`parser.rs`/`interp.rs` hot paths.

خلاصه فارسی: در این نسخه ROADMAP، سند BUILD، HANDOFF4 و کامنت‌های مکانیزم‌های
پیچیده اضافه شد؛ مرجع کامل زبان و کامنت‌گذاری parser/interp باقی مانده است.

---

## Verification discipline (unchanged)

The interpreter is the oracle. Every tier must match it byte-for-byte:
`cargo test --release` (117) · `tests/run_corpus.sh` (23 programs × 4 VM modes +
AOT census with byte-diff gate) · stdlib 34/34 · 52-file example sweep × 4 modes ·
valgrind on boxed AOT binaries · `fmt` idempotence · 0 compiler warnings.
