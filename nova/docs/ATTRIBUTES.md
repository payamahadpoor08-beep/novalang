# Nova attributes — real semantics

Attributes (`#[...]` before a function) used to be discarded by the parser. As of
v3.28 they carry **real, tested behaviour** and behave identically on every tier
(`nova run`, `nova vm`, JIT) — attributed functions run through the interpreter's
call path so their semantics are uniform. Each attribute below has a corpus test
(`tests/corpus/attributes.nova`) and Rust unit tests.

## Implemented

### `#[zero_alloc]`
A **static guarantee**, enforced by `nova check`: the function must not allocate.
Any array/map/set/struct literal, comprehension, f-string, closure, or string
concatenation is reported as an error. Integer/float arithmetic is fine.

```nova
#[zero_alloc]
fn dot(a, b, c, d) { a*c + b*d }   // ok

#[zero_alloc]
fn bad(n) { xs = [n]; xs }          // error: allocates an array/set literal
```

### `#[self_healing(attempts: N)]`
Runtime fault tolerance: if the call raises a runtime error (or `throw`), it is
retried up to **N** times before the error propagates. Useful for flaky I/O.
State shared through a reference (an array/struct) persists across retries.

```nova
#[self_healing(attempts: 5)]
fn connect(state) {
  state[0] = state[0] + 1
  if state[0] < 3 { throw "transient" }   // fails twice, then succeeds
  state[0]
}
```

### `#[hot_swap(scope: function)]`
The function's body can be **replaced at runtime** via `hot_swap("name", closure)`;
subsequent calls use the new body. Errors if the target isn't marked `#[hot_swap]`.

```nova
#[hot_swap(scope: function)]
fn formatter(x) { "v1:" + str(x) }
// ... later ...
hot_swap("formatter", (x) => "v2:" + str(x * 10))
```

### `#[integrity]`
`integrity_of("name")` returns a stable content hash (FNV-1a over the function's
AST) — the same for identical code, different when the code changes. A program can
verify its own critical functions haven't been altered.

```nova
#[integrity]
fn important() { 1 + 2 + 3 }
// integrity_of("important") is stable across runs, distinct per function
```

### `#[memo]` / `#[memoize]`
Caches results by argument values — a pure function is computed once per distinct
input. (`memo`'d `fib` runs in linear time.)

### `#[requires(expr, …)]` / `#[assumes(expr, …)]` / `#[ensures(expr, …)]`
**Design by contract.** `requires`/`assumes` predicates are checked at entry with
the parameters in scope; `ensures` is checked at exit with `result` bound to the
return value. A violation throws a catchable contract error.

```nova
#[requires(x >= 0)]
#[ensures(result >= x)]
fn double_up(x) { x + x }
```

### `#[retry(attempts: N)]`
Alias of `#[self_healing]`: retry the call on a runtime error up to N times.

### `#[trace]` / `#[log]` / `#[audit]`
Prints a deterministic `trace: name(args) -> result` line on every call.

### `#[profile]`
Counts calls; `profile_of("name")` returns the count.

### `#[deprecate(...)]` / `#[deprecated]`
Prints a one-time warning to stderr on first call (with the optional note).

### `#[comptime]`
Marks a **no-argument** function for compile-time evaluation. Its body is
evaluated exactly once, before `main` runs, and the result is cached; every call
to the function returns that precomputed constant instead of re-running the body
(e.g. build a lookup table or fold a constant once at startup). Because the value
is computed at init, a `#[comptime]` function must be self-contained — it runs
before global constants exist, so referencing runtime-only state is an error.
Interp-only (uniform across tiers). See `tests/corpus/comptime_eval.nova`.

### Introspection — `attrs_of("name")`
Returns the array of every attribute name on a function. Because **all** attributes
are captured (not just the behavioural ones), even attributes whose full behaviour
is still on the roadmap are visible and usable via this builtin.

## Roadmap (parse-only today — being implemented in later phases)
`#[encrypt]`, `#[time_travel]`, `#[obfuscate]`, `#[simd]`, `#[anti_debug]`,
`#[anti_tamper]`, `#[polymorph]`, plus optimisation hints (`#[inline_cache]`,
`#[tail_call]`, `#[cold]`/`#[hot]`) and metadata tags (`#[version]`, `#[since]`,
`#[intent]`, `#[example]`, `#[throws]`, `#[budget]`, …). All are **captured and
introspectable via `attrs_of`** today; each gains full behaviour in later phases and
is marked done in FEATURES.md only when a corpus test proves it.

---

خلاصه فارسی: attributeها دیگر دور ریخته نمی‌شوند و رفتار واقعی دارند (یکسان روی همه‌ی
تیرها). `#[zero_alloc]` تضمین ایستای عدم تخصیص حافظه (با `nova check`)؛
`#[self_healing(attempts: N)]` تلاش مجدد روی خطا؛ `#[hot_swap]` جایگزینی بدنه در زمان
اجرا با `hot_swap(name, closure)`؛ `#[integrity]` هش محتوایی پایدار برای تشخیص دستکاری.
بقیه‌ی attributeها در فازهای بعدی پیاده می‌شوند و تا آن زمان no-op و صادقانه مستند‌اند.
