# Nova cross-language benchmark

Identical algorithms in every language. Each must print the **same** result
(correctness gate) before its time is recorded. Times are best-of-3 wall-clock ms
on the CI machine — reproduce with `bash run.sh`.

Nova appears across its tiers:
- **`nova run`** — the tree-walking interpreter (the *oracle*: every other tier is
  proven byte-identical to it). Not a performance tier.
- **`nova vm`** — bytecode VM + tiered Cranelift JIT. The default fast path.
- **`nova build --aot`** — native binary (C backend).

Workloads: `fib` (recursive, call-bound), `sieve` (integer array), `mandel`
(float, nested loops). Run one with `bash run.sh fib`.

Toolchains: C (gcc -O2), C++ (g++ -O2), Java, Node, TypeScript (node
--experimental-strip-types), Lua 5.4, Python 3. Not installed here: Zig, Nim, R.
