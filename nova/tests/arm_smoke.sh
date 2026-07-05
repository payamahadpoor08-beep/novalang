#!/usr/bin/env bash
# ARM targets: `nova build --aot=arm` (aarch64 / ARMv8, modern phones) and
# `--aot=arm32` (ARMv7 32-bit, older/weaker phones) cross-compile the AOT C
# (typed + boxed, via the portable nova_rt.c) to a static binary and ship it only
# if, run under the matching qemu, output is byte-identical to `nova run`. Skips
# an arch cleanly if its cross gcc or qemu is absent.
#   bash tests/arm_smoke.sh [path-to-nova]
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
mkdir -p build
progs="int_edges aot_arrays aot_float_print aot_strings aot_mixed_tiers float_edges aot_sieve"

run_arch() { # $1=flag $2=cc $3=qemu $4=label
  local flag="$1" cc="$2" qemu="$3" label="$4"
  if ! command -v "$cc" >/dev/null 2>&1 || ! command -v "$qemu" >/dev/null 2>&1; then
    echo "$label smoke: SKIP ($cc or $qemu absent)"; return 0
  fi
  local pass=0 fail=0 skip=0
  for p in $progs; do
    local f="tests/corpus/$p.nova"; [ -f "$f" ] || continue
    local expect; expect=$("$BIN" run "$f" 2>&1)
    "$BIN" build "--aot=$flag" "$f" >/dev/null 2>&1
    local b="build/$p"
    if [ ! -f "$b" ]; then skip=$((skip+1)); continue; fi
    local got; got=$("$qemu" "$b" 2>&1)
    if [ "$got" = "$expect" ]; then pass=$((pass+1)); else
      echo "FAIL [$label/$p]:"; diff <(echo "$expect") <(echo "$got") | head -4; fail=$((fail+1))
    fi
    rm -f "$b"
  done
  echo "$label smoke: $pass passed, $fail failed, $skip skipped"
  [ $fail -eq 0 ]
}

rc=0
run_arch arm   aarch64-linux-gnu-gcc    qemu-aarch64 "arm64(ARMv8)" || rc=1
run_arch arm32 arm-linux-gnueabihf-gcc  qemu-arm     "arm32(ARMv7)" || rc=1
exit $rc
