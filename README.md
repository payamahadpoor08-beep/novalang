# Nova

**A batteries-included programming language that reads like Python and compiles like a systems language** — hand-written interpreter, bytecode VM, tiered Cranelift JIT, and true native AOT (C **and** LLVM backends), all implemented from scratch in Rust and all verified **byte-identical** to each other on every program.

[![CI](https://github.com/payamahadpoor08-beep/novalang/actions/workflows/ci.yml/badge.svg)](https://github.com/payamahadpoor08-beep/novalang/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-3.26.0-brightgreen.svg)](CHANGELOG.md)
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg)](nova/Cargo.toml)

```nova
fn fib(n) {
  if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
  print(fib(30))          // same source, four execution tiers, identical output
}
```

## Why Nova is interesting

- **Four execution tiers, one semantics.** The tree-walking interpreter is the
  oracle; the slot-based bytecode VM, the tiered Cranelift JIT (hot functions
  compile after N calls, with deoptimization back to the VM), and the AOT
  native builds must all reproduce its output **byte-for-byte** — enforced by a
  differential corpus run on every change, and by a build gate that refuses to
  ship a native binary that differs.
- **A real language, not a calculator**: structs + methods, closures, enums with
  pattern matching, generics with trait bounds, lazy generators, async/await +
  channels, try/catch/finally + defer, refinement types, static ownership
  (move) checking, modules, a REPL, a formatter, a doc extractor, and a
  10-module standard library written in Nova itself with its own test blocks.
- **JIT that never lies**: integer overflow promotes to BigInt, so every JIT'd
  multiply is overflow-checked and deopts to the VM when the integer world is
  left. Local integer arrays JIT into a thread-local arena; out-of-bounds
  access deopts and re-raises the interpreter's exact error.
- **Two independent native backends** (portable C and textual LLVM IR) sharing
  one from-scratch refcounted runtime — each gated byte-for-byte against the
  interpreter, valgrind-clean, with float printing proven identical to Rust's
  `Display` by a 2-million-value differential fuzz.

| tier — `fib(35)` | time | speedup |
|---|---|---|
| `nova run` (interpreter) | 9.6 s | 1× |
| `nova vm --no-jit` (bytecode VM) | 3.9 s | 2.5× |
| `nova vm` (tiered Cranelift JIT) | 0.068 s | ~140× |
| `nova build --aot` (native, C or LLVM) | **0.030 s** | **~320×** |

## Installation

Requires the Rust toolchain (`rustup.rs`) and a C compiler (for AOT builds).

**Linux / macOS**

```bash
git clone https://github.com/payamahadpoor08-beep/novalang.git
cd novalang
./install.sh                 # builds release binary, installs to ~/.local/bin
nova version                 # -> Nova 3.26.0
```

**Any platform (cargo)**

```bash
cargo install --path nova    # puts `nova` on your cargo bin path
```

**Windows**

```powershell
git clone https://github.com/payamahadpoor08-beep/novalang.git
cd novalang\nova
cargo build --release        # binary at target\release\nova.exe
```

## Quick start

```bash
nova repl                    # interactive REPL (also: just `nova`)
nova run app.nova            # tree-walking interpreter (the oracle)
nova vm app.nova             # bytecode VM + tiered JIT (fastest way to run)
nova build --aot app.nova    # standalone native binary (C backend)
nova build --aot=llvm app.nova
nova test suite.nova         # run `test "..." { }` blocks
nova fmt -w app.nova         # canonical formatter (idempotent)
nova check app.nova          # gradual type checker with located errors
```

## Documentation

| doc | contents |
|---|---|
| [nova/README.md](nova/README.md) | full language tour + the development story of every phase |
| [nova/docs/ROADMAP.md](nova/docs/ROADMAP.md) | honest inventory: every known weakness, what v3.26 fixed, concrete designs for the rest (bilingual EN/فارسی) |
| [nova/docs/BUILD.md](nova/docs/BUILD.md) | the AOT pipeline: tiers, both backends, the byte-diff oracle gate |
| [CHANGELOG.md](CHANGELOG.md) | release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how to build, verify, and add to the differential corpus |

## Verification discipline

Every change must pass, warning-free:

```bash
cd nova
cargo test --release          # 117+ unit + differential tests
bash tests/run_corpus.sh      # 23 programs x 4 VM modes + AOT tier census, byte-identical
for m in list sort mathx strx ds func json setx fmtx datex; do
  ./target/release/nova test std/$m.nova
done
```

## License

MIT — see [LICENSE](LICENSE).
