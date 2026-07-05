#!/usr/bin/env bash
# ARM target: `nova build --aot=arm` cross-compiles the AOT C (typed + boxed,
# via the portable nova_rt.c) to a static aarch64 binary and ships it only if,
# run under qemu-aarch64, its output is byte-identical to `nova run`. Exercises a
# spread of typed + boxed corpus programs. Skips cleanly if the cross toolchain
# or qemu is absent. Run from repo root:  bash tests/arm_smoke.sh [path-to-nova]
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
if ! command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 || ! command -v qemu-aarch64 >/dev/null 2>&1; then
  echo "arm smoke: SKIP (aarch64 gcc or qemu-aarch64 absent)"; exit 0
fi
mkdir -p build
progs="int_edges bigint_math aot_arrays aot_float_print aot_strings aot_mixed_tiers float_edges numeric_mixed"
pass=0; fail=0; skip=0
for p in $progs; do
  f="tests/corpus/$p.nova"; [ -f "$f" ] || continue
  expect=$("$BIN" run "$f" 2>&1)
  "$BIN" build --aot=arm "$f" >/dev/null 2>&1
  b="build/$p"
  if [ ! -f "$b" ]; then echo "SKIP $p (not typed/boxed AOT-able)"; skip=$((skip+1)); continue; fi
  if ! file "$b" | grep -q aarch64; then echo "FAIL [$p] not an aarch64 binary"; fail=$((fail+1)); rm -f "$b"; continue; fi
  got=$(qemu-aarch64 "$b" 2>&1)
  if [ "$got" = "$expect" ]; then pass=$((pass+1)); else
    echo "FAIL [$p] arm output diverged:"; diff <(echo "$expect") <(echo "$got") | head -6; fail=$((fail+1))
  fi
  rm -f "$b"
done
echo "arm smoke: $pass passed, $fail failed, $skip skipped"
[ $fail -eq 0 ]
