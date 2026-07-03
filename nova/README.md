# Nova

A real, **batteries-included programming language** with a hand-written
tree-walking interpreter in Rust, built on a fuzz-tested Pest grammar — and a
serious type system underneath.

Nova programs actually run and produce output — this is a working language, not
a paper design. It reads clean like Python, but carries the machinery of a
modern systems-grade language: structs, closures, enums with pattern matching,
generics with trait bounds, lazy generators, file modules, an effect system,
refinement types, and static ownership (linear/affine) checking — all behind a
gradual type checker with located errors, plus an interactive REPL.

```nova
fn fib(n) {
  if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
  print(fib(10))        // 55
}
```

---

## Quick start

```bash
cargo build --release

./target/release/nova run program.nova     # run a program
./target/release/nova check program.nova    # parse + type-check, report errors
./target/release/nova test  suite.nova      # run all test "..." { } blocks
./target/release/nova doc   program.nova     # extract /// docs as Markdown
./target/release/nova fmt   program.nova     # print canonically formatted source
./target/release/nova repl                   # interactive REPL
./target/release/nova version                # print version
```

Running `nova` with no arguments drops you straight into the REPL.

---

## Language tour

### Values and variables

`let`, `return`, and semicolons are all **optional**. Write Nova clean like
Python, or explicit like Rust — both styles work and mix freely.

```nova
x = 10                 // no `let` needed
name = "Nova"          // no semicolon needed
greeting = "hello " + name
```

Types: integers, floats, strings, booleans, `null`, arrays, structs, closures,
enums, and maps.

### Operators

```nova
a + b   a - b   a * b   a / b   a % b   a ** b      // arithmetic (** is power)
a == b  a != b  a < b   a <= b  a > b   a >= b      // comparison
a && b  a || b  !a                                   // logic
a ?? b                                               // null-coalescing
```

### Control flow

```nova
if x > 0 { "positive" } else if x < 0 { "negative" } else { "zero" }

while i < 10 { i = i + 1 }

for i in 0..10 { print(i) }      // ranges
for x in array { print(x) }       // iteration

// `if` is an expression — it yields a value
sign = if n < 0 { -1 } else { 1 }
```

### Functions and recursion

```nova
fn factorial(n) {
  if n <= 1 { 1 } else { n * factorial(n - 1) }   // last expression is returned
}
```

### Arrays

```nova
nums = [5, 2, 8, 1, 9]
nums[0] = 99                      // indexed assignment
print(len(nums))
print(sort(nums))                 // [1, 2, 8, 9, 99]
push(nums, 7)
last = pop(nums)
```

### Structs and methods

```nova
struct Account { owner: String, balance: i32 }

impl Account {
  fn deposit(mut self, amount) { self.balance = self.balance + amount }
  fn report(self) { print(self.owner + ": " + str(self.balance)) }
}

fn main() {
  acc = Account { owner: "Payam", balance: 100 }
  acc.deposit(50)
  acc.report()                    // Payam: 150
}
```

Methods take `self` or `mut self`; mutation through `mut self` is visible to the
caller (reference semantics).

### Closures and functional programming

```nova
fn make_adder(n) { x => x + n }   // returns a closure that captures n

nums = [1, 2, 3, 4, 5]
print(map(nums, x => x * 2))             // [2, 4, 6, 8, 10]
print(filter(nums, x => x % 2 == 0))     // [2, 4]
print(reduce(nums, (a, b) => a + b, 0))  // 15

add5 = make_adder(5)
print(add5(10))                          // 15
```

Closures are first-class values with lexical capture: lambdas remember the
environment in which they were created.

### Enums and pattern matching

```nova
enum Option { None, Some(i32) }
enum List   { Nil, Cons(i32, List) }

fn list_sum(lst) {
  match lst {
    Nil => 0,
    Cons(head, tail) => head + list_sum(tail),
  }
}

fn classify(n) {
  match n {
    0 => "zero",
    1 | 2 | 3 => "small",       // or-patterns
    4..=10 => "medium",         // range patterns
    x if x > 100 => "huge",     // guards
    _ => "large",               // wildcard
  }
}
```

Patterns support literals, bindings, enum destructuring, or-patterns, ranges,
guards, wildcards, and slice patterns (`[head, ...tail]`). Self-referential
enums give recursive data types.

### Dictionaries / maps

```nova
freq = dict()
for w in ["apple", "banana", "apple"] {
  if map_has(freq, w) { map_set(freq, w, map_get(freq, w) + 1) }
  else { map_set(freq, w, 1) }
}
print(freq)                       // {"apple": 2, "banana": 1}

ages = dict()
ages["Payam"] = 24                // index syntax works too
print(ages["Payam"])
```

### Error handling

```nova
struct AppError { code: i32, message: String }

fn risky(n) {
  if n < 0 { throw AppError { code: 400, message: "negative" } }
  n * 2
}

fn main() {
  try {
    print(risky(-1))
  } catch err {
    print("error " + str(err.code) + ": " + err.message)
  } finally {
    print("cleanup always runs")
  }
}
```

`throw` raises any value; `try`/`catch`/`finally` handle it. Throws propagate
across function calls to the nearest `try`. Genuine runtime errors and explicit
throws are kept distinct.

**Diagnostics.** Errors are reported like a modern compiler — the offending
source line with a caret and a `--> file:line:col` locator:

```
runtime error: index 10 out of bounds (len 3)
  --> prog.nova:3:3
   |
 3 |   print(xs[10])
   |   ^
```

This frame is shared by runtime errors, type-checker errors, and warnings. The
gradual type checker (`nova check`) also verifies **argument types** against
declared parameter types — passing a `Str` where an `Int` is required is a
compile-time error — while staying gradual: an unknown/dynamic type on either
side, and `Int`/`Float` widening, are accepted. Syntax errors name what was
expected in human terms ("an expression"), not internal grammar rules.

### Modules and the standard library

```nova
use math.{sqrt, pow, gcd}
use strings.{upper, split, join}
use arrays.{sort, sum}
use math as m

print(sqrt(2))                    // bare call
print(math.max(5, 9))             // qualified call
print(m.pow(3, 3))                // alias call
```

Standard modules:

- **math**: sqrt, cbrt, pow, abs, floor, ceil, round, trunc, fract, sign, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, log, log2, log10, ln, exp, min, max, clamp, lerp, gcd, lcm, factorial, hypot, deg2rad, rad2deg, pi, e, tau, phi
- **strings**: upper, lower, title, capitalize, trim, trim_start, trim_end, split, join, contains, starts_with, ends_with, replace, repeat, pad_left, pad_right, find, substring, levenshtein
- **arrays**: sort, reverse, sum, contains_elem
- **collections**: first, last, min_of, max_of, unique, slice, take, skip, flatten, zip, enumerate, chunk, concat
- **iter**: count, position, all_true, any_true
- **rand**: seed, random, rand_int, rand_float, rand_bool, choice, shuffle
- **time**: now, now_millis, now_nanos
- **json**: json_parse, json_stringify, json_pretty

### Built-in tests

A `test "name" { ... }` block is a first-class language construct. Run them all
with `nova test file.nova`:

```nova
fn factorial(n) { if n <= 1 { 1 } else { n * factorial(n - 1) } }

test "factorial works" {
  assert_eq(factorial(5), 120)
  assert_eq(factorial(0), 1)
}

test "math holds" {
  assert(2 + 2 == 4, "arithmetic broke")
}
```

```
running 2 test(s)

  PASS  factorial works
  PASS  math holds

2 passed, 0 failed
```

### Comprehensions, f-strings, and compound assignment

```nova
// list comprehension: [expr for x in iterable if condition]
squares = [x * x for x in nums]
evens   = [x for x in nums if x % 2 == 0]
both    = [x * 2 for x in nums if x % 2 == 0]

// f-strings: interpolate any expression with { }
name = "Payam"
print(f"Hello {name}, next year you are {age + 1}")

// compound assignment
x += 5    // x = x + 5
x -= 3
x *= 2
x /= 4
x %= 3
```

- comprehensions iterate arrays (and the keys of maps), filter with `if`
- f-strings (`f"...{expr}..."`) interpolate arbitrary expressions
- compound assignment `+= -= *= /= %=` desugars to the binary op

### Infinite loops

```nova
loop {
  if done { break }      // loop runs until an explicit break
  if skip { continue }
}
```

`loop { ... }` is an unconditional loop, equivalent to `while true`. Exit it with
`break` and skip iterations with `continue`.

### Safe navigation and null coalescing

```nova
u?.name              // Null if u is Null, else u.name
u?.profile?.bio      // chains safely through Nulls
config ?? "default"  // value if non-Null, else the fallback
empty?.name ?? "anon"
```

`?.` short-circuits to `null` instead of erroring when the base is `null`.
`??` returns its left side unless it is `null`, in which case it returns the right.

### Bitwise operators and loop control

```nova
print(12 & 10)   // 8   AND
print(12 | 10)   // 14  OR
print(12 ^ 10)   // 6   XOR
print(1 << 4)    // 16  shift left
print(256 >> 2)  // 64  shift right

while true {
  if done { break }       // break exits the loop
  if skip { continue }    // continue skips to the next iteration
}
```

Bitwise `& | ^ << >>` and unary `~` (NOT) work on integers with the usual precedence. Modulo `%` works on both integers and floats (`5.5 % 2.0` is `1.5`). `break` and
`continue` control loop flow in `while`, `for n in a..b`, and `for x in list`.

### Pipelines, maps, sets, and defer

```nova
// pipeline: a |> f desugars to f(a)
result = 5 |> double |> inc       // inc(double(5)) = 11
total  = nums |> filter(even) |> sum

// map literal #{}  and set literal #()
ages = #{"Ali": 30, "Sara": 25}
print(ages["Sara"])                // 25
unique = #(1, 2, 2, 3)             // 3 distinct elements

// defer: runs on the way out, in reverse order, even on throw/return
fn process() {
  defer cleanup()                  // always runs last
  defer print("second")            // runs before cleanup
  work()
}
```

- `|>` threads a value into the next call as its final argument
- `#{k: v, ...}` builds a map; `#(a, b, ...)` builds a set (deduplicated)
- `defer expr` / `defer { ... }` queue work to run when the block exits (LIFO)
- defers run on normal exit, early `return`, and `throw` alike

### Constants and data types

```nova
const MAX_USERS = 100
const APP_NAME = "Nova"
const LIMITS = [10, 20, 30]

data Point(x, y)              // compact struct declaration
data Color(r, g, b)

fn main() {
  print(MAX_USERS * 2)        // 200
  p = Point { x: 3, y: 4 }
  print(p.x)                  // 3
}
```

- `const NAME = value` defines a read-only global, evaluated once at load
- `data Name(a, b, ...)` is a concise struct; construct with `Name { a: .., b: .. }`

### Macros

```nova
macro square { ($x:expr) => { $x * $x } }

fn main() {
  print(square!(5))       // 25
  print(square!(3 + 1))   // 16 — arguments are parenthesised, so precedence holds
}
```

Macros expand at parse time by substituting each argument for its `$param`
placeholder, then re-parsing the result. Arguments are wrapped in parentheses to
preserve evaluation order.

### Generics

Functions and structs can be generic over type parameters. Because Nova's
interpreter is dynamic, generics run with zero overhead; the type checker
understands them too and resolves type variables at each call site.

```nova
fn id[T](x: T) -> T { x }

fn main() {
  print(id(42))        // works with Int
  print(id("hello"))   // and Str, and anything else
}
```

- type parameters go in `[T, U, ...]` after the function or struct name
- the checker substitutes them at call sites: `id[T](x: T) -> T` applied to an
  `Int` argument has return type `Int`, so a later misuse is still caught
- explicit parameter and return types (`n: Int`, `-> Str`) sharpen the checker

### Traits

Traits define a contract of methods that types can implement, with optional
default implementations — like Rust traits or Java interfaces.

```nova
trait Describe {
  fn describe(self) -> Str;          // required (note the semicolon)
  fn loud(self) -> Str { "..." }      // default implementation
}

struct Dog { name: Str }

impl Describe for Dog {
  fn describe(self) -> Str { "a dog named " + self.name }
  fn loud(self) -> Str { "WOOF" }     // overrides the default
}
```

- required methods end with `;`; default methods have a body
- `impl Trait for Type` must provide every required method (checked at load time)
- types inherit default methods they don't override
- calling an unimplemented requirement or an unknown trait is a load-time error

### State machines

A `machine` block is a first-class declarative state machine. It compiles to a
value with a current state plus `send`/`state_of` operations.

```nova
machine TrafficLight {
  initial Red
  Red    -> Green  on "go"
  Green  -> Yellow on "slow"
  Yellow -> Red    on "stop"
}

fn main() {
  light = TrafficLight()
  print(state_of(light))      // Red
  print(send(light, "go"))    // Green
  print(send(light, "slow"))  // Yellow
}
```

- `machine Name { initial S  A -> B on "event" ... }`
- construct with `Name()`, inspect with `state_of(m)`, transition with `send(m, "event")`
- an invalid transition throws, so it can be caught with `try`/`catch`
- `machine`, `initial`, and `on` are contextual — still usable as identifiers

### Interactive REPL

```
$ nova repl
Nova 1.0 — interactive REPL

nova> 1 + 2 * 3
7
nova> x = 10
nova> x * x
100
nova> fn double(n) { n * 2 }
ok
nova> double(21)
42
nova> :quit
```

Definitions and variables persist across lines.

---

## Type checking

`nova check` runs a gradual static type checker before execution. It catches
real mistakes without rejecting valid dynamic code:

```
$ nova check buggy.nova
found 4 type error(s):
  - function `add` expects 2 argument(s), got 1
  - operator `-` requires numbers, found Str and Int
  - if condition must be Bool, found Int
  - undefined variable: unknown_thing
```

It detects undefined variables, wrong function arity, operators on incompatible
types (`"x" - 5`), non-boolean conditions, and unknown struct fields.

The checker infers each function's **return type** and propagates it across call
boundaries, so a bug like `if get_count() { ... }` (where `get_count` returns an
`Int`) is caught even though the error is one call away. It also emits **warnings**
for variables that are assigned but never read (prefix with `_` to silence).

Types that can't be determined become `Unknown`, which unifies with everything —
so the checker complements Nova's dynamic nature instead of fighting it.

## Architecture

```
src/
  nova.pest    the official grammar (v13.1, 539 lines, dot-paths only, no `::`)
               compile-verified against pest 2.7.8; 350k inputs, 0 parser panics
  ast.rs       typed syntax tree
  parser.rs    lowers pest's parse tree into the AST
  types.rs     gradual static type checker (nova check)
  interp.rs    tree-walking interpreter + standard library + test runner
  main.rs      CLI: run / check / test / repl / version
```

The interpreter walks the AST directly, maintaining a call stack of scopes.
Arrays, structs, closures, enums, and maps are reference-counted so mutation
through methods and shared references behaves correctly.

---

## Built-in functions

**Core**: print, len, push, pop, array, str, int, float, abs, sqrt
**Functional**: map, filter, reduce, range
**Maps**: dict, map_set, map_get, map_has, map_del, map_len, map_keys, map_values
**Testing**: assert, assert_eq
**State machines**: send, state_of

(Plus everything in the `math`, `strings`, and `arrays` modules.)

---

### Async, spawn, and channels

Nova runs concurrent code on a cooperative single-threaded scheduler — real
interleaving, no OS threads, no data races.

```nova
async fn compute(n) { n * n }      // calling an async fn yields a Future

fn main() {
  print(compute(8).await)          // 64 — .await drives the Future to a value

  h = spawn { 1 + 2 + 3 }          // spawn queues a task, returns a handle
  print(h.await)                   // 6

  ch = chan()                      // a buffered channel
  spawn {
    ch <- 10                       // send with `ch <- value`
    ch <- 20
    close(ch)
  }
  print(recv(ch) + recv(ch))       // 30 — recv pulls from the channel
}
```

- `async fn f(...)` returns a `Future`; `f(...).await` (or `await f(...)`) runs it
- `spawn { ... }` queues a task and returns a handle; `handle.await` waits for it
- a spawned task's trailing expression is its result value
- channels: `chan()` creates one, `ch <- v` sends, `recv(ch)` receives,
  `close(ch)` closes, `chan_len(ch)` reports buffered count
- tasks that are never awaited still run after `main` returns (fire-and-forget)
- the scheduler advances queued tasks whenever a `recv` would otherwise block,
  so producers and consumers make progress without threads

See `examples/async.nova` for await, spawn, channels, and fan-in together.

## Status

Nova is a working language covering the core of modern programming: functions
and recursion, arrays, structs and methods, closures with capture, enums and
pattern matching, dictionaries, modules with a standard library, error handling,
async/await with spawn and channels, a test runner, and an interactive REPL.

The official grammar (`src/nova.pest`, v13.1) is broader than the interpreter:
it specifies effect polymorphism, linear/affine types, refinement types,
macros 2.0, streams, and `select`. These parse today and are on the roadmap for
execution. Paths use dots only — Nova never uses `::`.


## Slice ranges and static globals (v3.3)

```nova
a = [10, 20, 30, 40, 50]
a[1..3]    // [20, 30]      half-open
a[1..=3]   // [20, 30, 40]  inclusive
a[..2]     // [10, 20]      open start
a[3..]     // [40, 50]      open end
a[-2..]    // [40, 50]      negative index from the end
1..5       // [1, 2, 3, 4]  a range is also a standalone value

static LIMIT = 5             // a global binding (semicolon optional)
```

Slicing works on arrays and strings. A bare `lo..hi` / `lo..=hi` range
materializes into an array. `static` declares a global, like `const`.


## Type aliases and slice patterns (v3.4)

```nova
type UserId = Int          // a friendly name for an existing type
type Row    = Array

fn next_id(id: UserId) -> UserId { id + 1 }   // checked exactly as Int

fn classify(xs) {
  match xs {
    []              => "empty",
    [only]          => f"single: {only}",       // exact length
    [head, ...tail] => f"{head} then {tail}",   // `...` captures the rest
  }
}

match [1, 2, 3, 4] {
  [first, ...middle, last] => ...,  // bind both ends, soak up the middle
}
```

A `type` alias is resolved by the type checker, so `UserId` catches the same
mistakes as `Int`. Slice patterns match arrays by shape: each element gets a
sub-pattern, and a single `...` rest soaks up the middle, optionally binding the
skipped elements to a name. See `examples/aliases_patterns.nova`.


## Trait bounds and extern declarations (v3.5)

```nova
trait Greet { fn greet(self) -> Str; }
struct Dog { name: Str }
impl Greet for Dog { fn greet(self) -> Str { "woof" } }

fn announce[T](x: T) -> Str where T: Greet { x.greet() }  // where clause
fn shout[T: Greet](x: T) -> Str { x.greet() }             // inline bound

extern "C" {
  fn c_clock();
  fn c_printf(fmt, ...);   // variadic
}
```

Trait bounds — written `where T: Trait` or inline `[T: Trait]` — are checked:
when a call's argument has a known concrete type, that type must have a matching
`impl Trait for Type`, or `nova check` reports an error. `extern` blocks declare
foreign functions; Nova has no FFI yet, so the checker knows their name and
arity but calling one is a clear runtime error. See `examples/bounds_extern.nova`.


## Catchable runtime errors, constant folding, and `nova doc` (v3.6)

```nova
fn safe_div(a, b) {
  result = 0
  try { result = a / b } catch e { result = -1 }   // division by zero is catchable
  result
}

print(2 + 3 * 4)   // constant-folded to 14 before execution
```

- **Catchable runtime errors.** Faults like division by zero or indexing past
  the end of an array are now ordinary exceptions: a `try`/`catch` handles them
  (the message arrives as the catch variable) instead of aborting the program.
  Uncaught, they still stop the program with a clear `runtime error:` message.
- **Constant folding.** A pre-execution pass folds fully-constant
  `Unary`/`Binary` subexpressions to their result. It reuses the interpreter's
  own arithmetic, so results are identical; errors (`1/0`) and big-integer
  results are deliberately left for runtime.
- **`nova doc <file>`.** Generates Markdown API docs from `///` doc comments
  (and a `//!` module header). Try `nova doc examples/documented.nova`.

See `examples/safe_errors.nova` and `examples/documented.nova`.


## Lazy generators (v3.7)

```nova
fn naturals() {            // any function with `yield` is a generator
  i = 0
  loop { yield i; i = i + 1 }
}

for n in naturals() {      // pulled lazily, one value at a time
  if n >= 5 { break }
  print(n)                 // 0 1 2 3 4
}

print(naturals().take(5))  // [0, 1, 2, 3, 4]  — works on infinite generators
```

A function whose body contains `yield` is a generator: calling it returns a lazy
sequence instead of running the body. Values are produced on demand, so infinite
generators are fine — `for`/`break` and `.take(n)` only pull what they need, and
`.next()` advances one step (null once exhausted). Laziness is implemented by
replaying the body up to the requested `yield`, so generator bodies should be
pure (free of side effects). See `examples/generators.nova`.


## File modules (v3.8)

```nova
// app.nova
use "geometry.nova"        // import every item from another file
use "sub/helpers.nova"     // path is relative to this file

fn main() { print(area(3, 4)) }   // area() comes from geometry.nova
```

`use "path.nova"` imports all items from another Nova file (previously `use`
only reached the built-in stdlib). Imports are resolved before type-checking and
execution: paths are relative to the importing file, each file is loaded exactly
once (so diamonds and cycles are safe), and a duplicate function name across
files is a clear error. See `examples/modules/app.nova`.


## Located error messages (v3.9)

```text
$ nova run prog.nova
runtime error: line 3, col 3: division by zero

$ nova check prog.nova
found 1 type error(s):
  - line 2, col 15: operator `-` requires numbers, found Str and Int
```

Runtime, type, and lowering errors now report `line, col`. The parser tags each
statement's expressions with their source position (an `Expr::At` marker); the
interpreter and type checker remember the position they are working on and prefix
it onto any error. Parse errors already point at the offending token via pest.
See `examples/errors_located.nova`.


## Effect annotations (v3.10)

```nova
fn announce(m) ![IO] { print(m) }   // declares the IO effect
fn add(a, b)   ![]   { a + b }       // pure: no effects allowed

fn oops() ![] { print("x") }         // nova check: performs `IO` not in ![]
```

A function may declare the effects it is allowed to perform with `![..]`
(`![IO]`, `![IO, Net]`, or `![]` for pure). `nova check` enforces it: performing
an effect outside the declared set — via an effectful builtin (`print` → `IO`,
`rand*` → `Rand`, …) or by calling a function that declares that effect — is a
located compile-time error. The system is gradual: unannotated functions are not
checked, and the interpreter ignores effects entirely. See `examples/effects.nova`.


## Refinement types and richer assertions (v3.11)

```nova
type Pos = Int if it > 0;        // a value of type Pos must satisfy `it > 0`

let n: Pos = 7                   // ok
let b: Pos = 0                   // runtime error: refinement `Pos` violated by value 0
```

A refined type alias attaches a predicate (over the value, written `it`). When a
value is bound to that type via an annotated `let`, the predicate is checked at
runtime; a violation is a located, catchable error. Test assertions also grew
beyond `assert`/`assert_eq`: `assert_ne`, `assert_true`, `assert_false`,
`assert_gt`, `assert_lt`, and `assert_contains`. See `examples/refinements.nova`.


## Linear & affine types — ownership checking (v3.12)

```nova
fn redeem(token: linear Int) -> Int { token * 10 }   // consumed exactly once

fn dup(x: linear Int) { print(x); print(x) }   // error: use of moved value `x`
fn lose(x: linear Int) { }                      // error: linear `x` never used
```

A `linear` parameter must be consumed exactly once; an `affine` parameter at most
once. The type checker performs a real flow analysis: using a value after it is
moved, never consuming a `linear` value, or using such a value inside a loop (it
could run more than once) are located compile-time errors. Branches are handled
soundly (using a value in both arms of an `if` is fine). Gradual and static —
unannotated parameters are unchecked and the interpreter ignores the modifiers.
See `examples/ownership.nova`.


## The formatter — `nova fmt` (v3.13)

```bash
nova fmt program.nova        # prints canonical, re-formatted Nova to stdout
```

`nova fmt` is an AST pretty-printer that emits clean, canonical Nova. It is
verified semantics-preserving: for every bundled example, running the formatted
output produces byte-identical results, and formatting is idempotent
(`fmt(fmt(x)) == fmt(x)`). It canonicalizes records to `data` form and drops the
redundant trailing `return` (Nova is expression-oriented). Current limitations:
comments are not preserved and macro calls print in expanded form.

> While building this, the round-trip check uncovered and fixed a latent
> interpreter bug: value blocks and `if`/`else` used as expressions
> (`let m = if a > b { a } else { b }`) now correctly yield their last value
> instead of `null`.


## Hygienic macros, multi-line REPL, `fmt -w` (v3.14)

```nova
macro double { ($x:expr) => { let tmp = $x; tmp + tmp } }
let tmp = 100
double!(tmp)   // 200 — the macro's own `tmp` never captures the caller's
```

- **Hygienic macros.** Macro bodies may now contain statements and `let`
  bindings; any binding the macro introduces is renamed to a unique name per
  expansion, so it can neither capture nor be captured by call-site variables.
- **Multi-line REPL.** The REPL keeps reading (with a `....>` prompt) while
  braces/parens are open, so you can type a whole function definition.
- **`nova fmt -w <file>`** rewrites a file in place instead of printing.

See `examples/macro_hygiene.nova`.


## The stream operator `->>` (v3.15)

```nova
let ch = chan();
[1, 2, 3, 4] ->> ch;          // stream every element into the channel
recv(ch) + recv(ch) + recv(ch) + recv(ch)   // 10
```

`src ->> sink` feeds each element of `src` into channel `sink` (it is
`for x in src { sink <- x }`) and evaluates to the channel, so it composes with
`recv`, `select`, and spawned producers. This is distinct from `<-` (a single
send) and `|>` (the function pipe `x |> f` == `f(x)`). See `examples/streams.nova`.


## Bytecode VM + JIT — `nova vm` / `nova bench` (v3.22, Phase 5 complete)

```bash
nova run    compute.nova        # tree-walking interpreter (default, full language)
nova vm     compute.nova        # bytecode VM + tiered JIT (same output, fastest)
nova vm     compute.nova --jit  # eager JIT: compile every eligible function up front
nova vm     compute.nova --no-jit          # pure VM, no native code
nova vm     compute.nova --jit-threshold=N # tune tiering (default 100 calls)
nova vm     compute.nova --jit-stats       # prove what got compiled (stderr)
nova vm     compute.nova --dump # print the bytecode instead of running
nova vm     compute.nova --no-opt  # skip the optimizer (for A/B comparison)
nova bench  compute.nova        # time interpreter vs VM vs JIT
nova disasm compute.nova        # pretty-print the compiled chunks
nova jit --dump compute.nova    # print the Cranelift IR for JIT-eligible functions
```

A step toward a native/JIT/LLVM backend: an AST→bytecode compiler and a
slot-based stack VM. The **computational core** — scalars, arithmetic,
`if`/`while`/`for`, functions and recursion — is compiled to native bytecode and
runs ~3× faster than the tree-walker on `fib(32)`.

**Phase 2 extends the VM to the whole language.** Heap and control features
beyond the core — arrays, maps/sets, structs, enums, `match`, closures, methods,
f-strings, comprehensions, slices, index/field assignment, channels — run
correctly under `nova vm`: the compiler keeps the core native and *delegates*
each remaining expression to the interpreter's own `eval` (which takes its scope
by shared reference, so heap mutations propagate and VM locals are never
corrupted — correct by construction). `for…in` iterates lazily, so even infinite
generators work with `break`. Named calls reuse the interpreter's `call_named`,
so builtins, stdlib, and enum/struct constructors resolve identically while
their *arguments* still run on the fast native path.

**Phase 3A compiles the heap operations to native opcodes** instead of
delegating them: array/map/set literals, ranges, element/key indexing and
slicing, struct construction, field reads (`a.b`, `a?.b`), f-strings, and
index/field assignment all run directly on the VM now (`MakeArray`/`MakeMap`/
`MakeSet`/`MakeRange`/`Index`/`Slice`/`GetField`/`MakeStruct`/`Fmt`/`IndexSet`/
`SetField`). Each opcode calls the *same* helper the interpreter uses
(`index_get`, `field_set`, `make_struct`, …), so results stay byte-identical;
heap-heavy code is ~1.75× faster than the tree-walker. Features not yet native
(`match`, closures, methods, comprehensions, async) still delegate, so coverage
never regresses.

**Phase 3 adds an optimizer, faster call frames, and tooling.** After
compilation each chunk is run through a peephole/CFG optimizer
(jump-threading → reachability-based dead-code elimination → compaction with
target remapping); `--no-opt` disables it and the test suite asserts optimized
and unoptimized runs produce identical results. Calls reuse pooled operand-stack
and locals buffers instead of allocating per call, so recursion is ~3.15× faster
than the tree-walker on `fib(32)`. `nova disasm` (and `nova vm --dump`) print the
bytecode, and a verifier checks jump-target and slot bounds in the test suite.

**Phase 4 makes the VM fully native** — `match`, method calls, and closures no
longer delegate. `match` compiles to per-arm tests (reusing the interpreter's
`match_pattern`, with bindings written into VM slots and guards/bodies native);
method calls dispatch through a shared `call_method_vals`; and each lambda body
is compiled to its own chunk, so closures created **and called** run on the VM
(capturing their environment like the interpreter does, with a fallback to
`call_closure` for closures the interpreter built). Match- and closure-heavy
programs run ~2× faster than the tree-walker.

**Phase 5 adds a Cranelift JIT** (`nova vm --jit`). Functions that provably
compute and return integers with no side effects, calls, or non-integer values
(`jit_eligible` in `src/jit.rs`) are compiled to native machine code; everything
else keeps running on the VM. The JIT is correct *by construction*: every
operation that could leave the integer world — arithmetic overflow (Nova would
promote to BigInt), division/modulo by zero, `**`, out-of-range shifts, negating
`i64::MIN` — branches to a **deopt** path, and the VM then re-runs the whole call
(safe, since eligible functions are pure). So the JIT can only be faster, never
wrong, and a differential test asserts JIT == VM on every case including the
deopts. Integer loop/iterative code runs **~15–35× faster** than the
interpreter (e.g. a Collatz sweep: interp 13.5s → VM 5.9s → JIT 0.36s).

**Phase 5B compiles calls between eligible functions as native calls** (the
eligible set is a fixpoint over the call graph), so recursion runs entirely in
machine code and a callee's deopt propagates up through the native frames. On
`fib(35)` the JIT is **~59× faster than the VM** (4.0s → 0.068s) and
byte-identical. `nova jit --dump` prints the generated Cranelift IR.

**Phase 5C adds tiering and f64 specialization.** The JIT is now on by default
in `nova vm`: a function compiles only after its 100th call (threshold picked by
benchmark; override with `--jit-threshold=N` or `jit-threshold` in `nova.hgx`),
so cold functions are **never** compiled — `--jit-stats` proves it. A second,
disjoint eligibility track compiles **f64-only** functions natively (`fadd`/
`fsub`/`fmul`/`fdiv`, comparisons in conditions); floats never deopt because
IEEE inf/NaN semantics already match the interpreter bit-for-bit. `%`/`**` on
floats stay on the VM (no Cranelift instruction), as does anything int/float
mixed — correct first, fast second.

## Projects & native binaries — `nova.hgx` and `nova build` (Phase 6.1)

A Nova project keeps a `nova.hgx` (strict TOML subset) at its root:

```toml
[package]
name = "myapp"
version = "0.1.0"
entry = "src/main.nova"

[build]
opt-level = "release"
jit-threshold = 100

[target]
default = "pc"
```

With it, `nova run` / `nova vm` / `nova check` / `nova build` need no file
argument (they read `entry`); explicit paths keep working unchanged.

`nova build` produces a **standalone executable** in `./build/<name>`: the Nova
runtime with the program embedded in a trailer (`NOVA_EMBED_v1`). It needs no
Nova installation to run, works for **every** Nova program (tiered VM/JIT for
compilable `main`s, interpreter fallback otherwise), and its output is
byte-identical to `nova run` by construction — verified across interp-only,
generator, async, and compute examples. Single-file programs only for now
(file imports are rejected with a clear error). See `examples/hgx_app/`.

## AOT — true native binaries via C and LLVM (Phase 7)

```bash
nova build --aot      app.nova   # C backend:    emit C, compile with cc -O2
nova build --aot=llvm app.nova   # LLVM backend: emit .ll, compile with clang -O2
```

When every function is in the JIT's i64/f64 eligible sets and `main` is
integer statements plus `print(...)`, `nova build --aot` compiles the whole
program to a **pure native executable** — no runtime, no warm-up, full
gcc/clang `-O2` optimization. The build **verifies the binary against
`nova run` byte-for-byte before shipping it**; any program that isn't fully
AOT-able (or diverges, e.g. an overflow that Nova promotes to BigInt) falls
back to the embedded-runtime build automatically — never an error, never a
wrong answer.

**Phase 8 — the boxed tier.** Programs with strings, arrays, slices,
f-strings, for-each, and float printing also compile to pure native binaries
now: the generated C `#include`s a small **refcounted runtime**
(`runtime/nova_rt.c`, written from scratch — tagged values, ownership-
disciplined codegen, UTF-8 char-indexed strings, and shortest-round-trip float
formatting matching Rust's `Display` exactly) and the whole thing is one
translation unit under `cc -O3 -flto`, so LLVM/GCC inline the runtime's
refcount and accessor ops straight into program code — the same architecture
Nim and Swift use. `--aot-flags="-march=native"` passes extra codegen flags.
`nova build --aot` reports which tier the program landed in (`typed` /
`boxed` / embed fallback).

**Both backends do both tiers.** `--aot=llvm` emits textual LLVM IR that calls
the same `runtime/nova_rt.c` through the ABI clang uses for the `NV` value type
(an NV passed by value is `(i8 tag, i64 payload)`; an NV return is `{i8,i64}`);
the `.ll` and the runtime are compiled together (the runtime gets external
linkage via `-Dstatic=`). So Nova has two independent native backends — C and
LLVM — that must each reproduce `nova run` byte-for-byte, which is real
differential coverage, not decoration. Numeric code wins big from AOT (9–60×);
string-heavy code is allocation-bound and lands near VM speed — measured, not
guessed. Programs using BigInt promotion, maps, structs, closures, match,
generators, or async stay on the embed build (still standalone, still
byte-identical).

Execution tiers on `fib(35)` (same source, same output):

| tier | time | vs interpreter |
|---|---|---|
| `nova run` (interpreter) | 9.6 s | 1× |
| `nova vm --no-jit` (bytecode VM) | 3.9 s | 2.5× |
| `nova vm` (tiered Cranelift JIT) | 0.068 s | ~140× |
| `nova build --aot` (C / LLVM native) | **0.030 s** | **~320×** |

## Standard library, written in Nova (`std/`)

`use "std/<module>.nova";` resolves relative to the importing file, then
`$NOVA_STD`, then next to the `nova` executable. Every module carries its own
`test` blocks (`nova test std/<module>.nova`):

| module | contents |
|---|---|
| `std/list.nova` | map/filter/reduce, reverse, concat, zip, enumerate, unique, flatten, chunk, windows, take/drop(+while), all/any, sum/product/min/max |
| `std/sort.nova` | insertion/quick/merge sort, `sort_by`, `is_sorted`, binary search, kth-smallest |
| `std/mathx.nova` | gcd/lcm, primality + sieve, factorial, fib, integer sqrt/pow, clamp/sign/abs, digits, nCr |
| `std/strx.nova` | starts/ends_with, repeat, pad, reverse, palindrome, find/contains/replace, split/join, trim, levenshtein |
| `std/ds.nova` | Stack, Queue (amortized), MinHeap (+heap_sort), Counter |
| `std/func.nova` | compose/pipe, apply_n, flip, negate, fold_range, fixpoint, memoize |
| `std/setx.nova` | insertion-ordered sets: add/has/remove, union/intersect/difference, subset/equality |
| `std/fmtx.nova` | pad/center, thousands separators, integer-math fixed-point, percent, table rows |
| `std/datex.nova` | civil date arithmetic (Hinnant algorithms): leap years, weekday, add_days/diff, ISO formatting, now_ms bridge |

## Differential test corpus (`tests/corpus/`)

Focused programs covering integer edge cases (i64 boundaries, BigInt
promotion), floats (inf/NaN/-0.0), strings, arrays/maps/sets, match patterns,
closures, structs/impl, generators, try/catch, stdlib integration, and JIT
tiering. `bash tests/run_corpus.sh` asserts every program is byte-identical
across `run`, `vm`, `vm --jit`, `vm --no-jit`, and a low tiering threshold.

## Writing real programs (Phase 9)

Nova has a full system interface — every builtin works identically in the
interpreter, VM, JIT, and built binaries:

| builtin | behavior |
|---|---|
| `args()` | program argv (`nova run f.nova a b` → `["a","b"]`; built binary → its argv) |
| `env(name)` | environment variable or `null` |
| `read_file(p)` / `write_file(p, s)` / `append_file(p, s)` | file I/O; failures are catchable `throw`s |
| `file_exists(p)` / `remove_file(p)` | filesystem checks |
| `read_line()` / `input(prompt)` | stdin (EOF → `null`) |
| `eprint(v)` / `exit(code)` | stderr and exit status |
| `to_int(s)` / `to_float(s)` | parsing (`null` on failure) |
| `chr(n)` / `ord(s)` / `type_of(v)` | codepoints and runtime types |
| `exec(cmd, [args])` | run a process → `{code, stdout, stderr}` map |
| `list_dir(p)` / `mkdir(p)` | directory listing (sorted) and recursive create |
| `cwd()` / `chdir(p)` | working directory get/set |
| `now_ms()` / `sleep_ms(n)` / `setenv(k, v)` | wall-clock time, sleep, env set |

`std/json.nova` is a full JSON parser/serializer **written in Nova** (objects,
arrays, escapes, `\u` codepoints, floats/exponents, position-reporting errors).

Two real CLI applications live in `examples/apps/`, both shippable as
standalone binaries with `nova build`:
- **`wc.nova`** — line/word/char counts (output verified against coreutils).
- **`todo.nova`** — a JSON-file-backed task manager (`add`/`list`/`done`/
  `clear`, `TODO_FILE` env override, corrupt-file handling, exit codes).

`nova build` now inlines `use "file.nova"` imports at build time, so
multi-file projects (including `std/`) ship as one self-contained executable.
Programs doing I/O are excluded from the AOT tiers by design — the AOT
byte-diff gate requires determinism — and ship via the embed build instead.

Two language fixes that fell out of writing real code: **string ordering**
(`"a" < "b"`, lexicographic, in all engines including the C runtime), and
`type_of` (was checker-known but never implemented).

## Roadmap 2026–27

- **2026 Q3** — AOT beyond the numeric core: strings/arrays in the C backend
  via a small refcounted runtime; `nova build` emits multi-file projects.
- **2026 Q4** — self-hosted tooling: the formatter and doc generator rewritten
  in Nova itself on `std/`; package registry design for `nova.hgx`
  dependencies.
- **2027 H1** — heap JIT (arrays/structs in native code), escape analysis in
  the optimizer, incremental compilation for `nova build`.
- **2027 H2** — debugger protocol, LSP server, and the 1.0 language
  specification freeze.

The test suite and an example sweep assert the VM's (and JIT's) output is
**byte-identical** to `nova run`. A `main` that uses statement-level
`try`/`throw`/`defer` is still handed back to `nova run` with a clear message —
the VM never produces a wrong answer. See `examples/compute.nova` and
`examples/vm_showcase.nova`.
