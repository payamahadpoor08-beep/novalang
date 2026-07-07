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

## Benchmarks — Nova vs C, C++, Java, JS, TypeScript, Lua, Python

Same algorithm in every language, each verified to print the **same result**
before timing (correctness gate). Best-of-3 wall-clock ms on the CI machine —
reproduce with `bash nova/bench/run.sh`. Nova's default fast path is
**`nova vm`** (bytecode VM + tiered Cranelift JIT); `nova build --aot` produces a
native binary; `nova run` is the tree-walking interpreter kept as the semantic
*oracle* (every other tier is proven byte-identical to it), not a speed tier.

| workload | C -O2 | **Nova aot** | **Nova vm** | Java | Node JS | Lua 5.4 | Python 3 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **fib(32)** recursive | 6 | **7** | 19 | 46 | 54 | 129 | 250 |
| **sieve** (primes <2M) | 7 | 54 | **52** | 71 | 45 | 124 | 34 |
| **mandelbrot** 600²×200 | 50 | **65** | **65** | 94 | 89 | 538 | 3816 |

(ms; lower is better.) On numeric kernels Nova compiles to native code through
its JIT/AOT and **matches C on recursion (fib) and beats Java, Node, Lua and
Python on the float-heavy mandelbrot** — while staying a dynamically-typed
Python-like language. The interpreter (`nova run`) is intentionally the slow,
authoritative oracle; run real work with `nova vm` or `nova build --aot`.

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

**32-bit ARM (Termux / Android, Raspberry Pi in 32-bit mode)**

The JIT tier uses Cranelift, which can only target `x86-64`, `aarch64`, `riscv64`
and `s390x`. On a 32-bit ARM host (Termux usually reports `armv7l`) Cranelift
won't even compile, so build with the JIT turned off — everything else
(interpreter, bytecode VM, and the native **AOT** backend via `nova build`) works
unchanged:

```bash
cargo build --release --no-default-features   # JIT off; interp + VM + AOT still work
```

`./install.sh` detects a 32-bit ARM host automatically and does this for you (and
falls back to `--no-default-features` if a full build fails on any host). The only
thing you give up is the in-process JIT; you still get native-speed binaries from
`nova build`.

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

## Networking (TCP + HTTP, in the box)

Nova ships blocking TCP sockets as builtins — `tcp_listen`, `tcp_accept`,
`tcp_connect`, `tcp_read`, `tcp_write`, `tcp_close` — so servers, clients and
(on top of them) HTTP are written directly in Nova, no libraries to wire up. A
complete HTTP/1.1 server (see [`nova/demos/http_server.nova`](nova/demos/http_server.nova)):

```nova
fn respond(conn) {
  req = tcp_read(conn, 4096)
  body = "Hello from Nova over HTTP!\n"
  resp = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n" +
         "Content-Length: " + str(len(body)) + "\r\nConnection: close\r\n\r\n" + body
  tcp_write(conn, resp); tcp_close(conn)
}
fn main() {
  ln = tcp_listen("127.0.0.1:8080")
  loop { respond(tcp_accept(ln)) }
}
```

```bash
nova run nova/demos/http_server.nova &
curl http://127.0.0.1:8080          # -> Hello from Nova over HTTP!
```

On top of the primitives, `nova/demos/` also has, in pure Nova:
`web_app.nova` (routing + static-file hosting + a JSON API — front-end **and**
back-end), and `ws_server.nova` / `ws_client.nova` (a full **WebSocket** RFC 6455
echo, handshake and framing included, using `ws_accept` + binary `tcp_*_bytes`).
Host resolution (`resolve`, honouring `/etc/hosts`), `hostname`, `base64_*`,
`sha1_hex` and `ws_accept` are builtins too.

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
