# Nova build pipeline — `nova build` and the AOT tiers

## Prerequisites

Building Nova needs a working Rust toolchain (stable). If `cargo build --release`
fails with:

```
error: rustup could not choose a version of cargo to run, because one wasn't
specified explicitly, and no default is configured.
```

your rustup has no default toolchain — run **`rustup default stable`** once, then
build again. Build from inside the crate directory
(`cd novalang/nova && cargo build --release`). To update an existing clone use
`git pull` inside it rather than re-running `git clone` (which errors that the
destination already exists).

## The one rule

Every shipped binary must print **byte-identical** output to `nova run` (the
tree-walking interpreter, the semantic oracle). `src/build.rs` enforces this: it
compiles, runs both, byte-diffs stdout, and silently falls back to the embed
build on any difference. A Nova binary can therefore be slower than hoped, but
never wrong.

## Tiers, in the order they are tried

| tier | what it is | when it applies |
|---|---|---|
| **typed** | pure i64/f64 code straight to native (no runtime) | every function JIT-eligible; `main` prints ints/literals only |
| **boxed** | refcounted `NV` values over `runtime/nova_rt.c` | strings, arrays, slices, f-strings, for-each; whole-program fixpoint in `src/aot.rs::analyze_boxed` |
| **embed** | the full interpreter + your source in one binary | everything else — always works |

Two native backends produce identical results and are both gated by the oracle:

- `--aot` / `--aot=c`: generated C, `#include "nova_rt.c"`, built with `cc -O2`.
- `--aot=llvm`: textual LLVM IR calling the same runtime through clang's `NV`
  ABI (`{i8 tag, i64 payload}` by value, `{i8,i64}` returns), compiled together
  with `nova_rt.c` (`-Dstatic=`).

## WASM target (`--aot=wasm`)

`nova build --aot=wasm program.nova` compiles the **typed and boxed** tiers (the
same portable AOT C, including the refcounted `nova_rt.c`) to a `program.wasm`
targeting `wasm32-wasi` with `clang --target=wasm32-wasi --sysroot=<wasi>`. It
ships only if it passes the oracle gate: the `.wasm` is run under **node's WASI**
and its output must be byte-identical to `nova run`, else no artifact (honest
fallback). Strings, arrays and maps work; only embed-tier programs are excluded.
Requires `clang` (wasm32 target), a wasi-libc sysroot (`apt-get install
wasi-libc libclang-rt-*-dev-wasm32`, giving `/usr/lib/wasm32-wasi/libc.a`), and
`node` (>=18, for `node:wasi`). See `tests/wasm_smoke.sh`.

## ARM target (`--aot=arm`)

`nova build --aot=arm program.nova` cross-compiles the **same portable AOT C**
(typed *and* boxed — `nova_rt.c` is ordinary libc C) to a **static aarch64**
binary with `aarch64-linux-gnu-gcc -static`, for Raspberry Pi / aarch64 mobile.
It ships only if it passes the oracle gate: the binary is run under
`qemu-aarch64` and its output must be byte-identical to `nova run`. Requires
`aarch64-linux-gnu-gcc` and (to self-verify) `qemu-aarch64`
(`apt-get install gcc-aarch64-linux-gnu qemu-user`). **`--aot=arm32`** does the
same for **ARMv7 (32-bit hard-float)** — older / weaker phones — via
`arm-linux-gnueabihf-gcc -marm`, verified under `qemu-arm`
(`apt-get install gcc-arm-linux-gnueabihf`). Embed-tier programs aren't
ARM-AOT-able (the embed binary would be the host arch); use the typed/boxed
tiers. See `tests/arm_smoke.sh` (covers both arches).

## Commands

```bash
nova build program.nova            # embed build (always succeeds)
nova build --aot program.nova      # typed→boxed→embed via the C backend
nova build --aot=wasm program.nova # wasm32-wasi (typed+boxed), node-WASI-verified
nova build --aot=arm program.nova  # static aarch64/ARMv8 (typed+boxed), qemu-verified
nova build --aot=arm32 program.nova # static ARMv7/32-bit (typed+boxed), qemu-verified
nova build --aot=llvm program.nova # same tiers via the LLVM backend
```

The printed line tells you the truth about what you got:
`built build/x (aot-llvm, boxed tier, native)` vs `... using the embedded runtime build`.

## Float printing

`nova_rt.c::fmt_f64` reproduces Rust's `Display for f64` exactly — full decimal
expansion (never e-notation), shortest round-trip digits, ryū-style tie
rounding — verified by a 2M-value differential fuzz against Rust. This is what
lets float-heavy programs pass the byte-diff gate and ship native.

## Verifying a build

```bash
bash tests/run_corpus.sh     # 23 programs: 4 VM modes + AOT tier census
valgrind -q ./build/yourbin  # boxed binaries are expected 0-error, 0-leak
```

---

## خلاصه فارسی

قانون یگانه: هر باینری باید خروجی بایت‌به‌بایت یکسان با `nova run` بدهد؛ در غیر
این صورت خودکار به بیلد embed برمی‌گردد (کند شدن ممکن است، غلط شدن هرگز).
سه لایه به ترتیب امتحان می‌شوند: **typed** (فقط i64/f64 خالص)، **boxed** (رانتایم
refcount شده C برای رشته/آرایه/…)، و **embed** (مفسر کامل + سورس). دو بک‌اند C و
LLVM هر دو از همان `nova_rt.c` استفاده می‌کنند و هر دو با گیت oracle چک می‌شوند.
چاپ float اکنون دقیقاً مطابق Rust است (فازتست ۲ میلیون مقداری، صفر اختلاف).
