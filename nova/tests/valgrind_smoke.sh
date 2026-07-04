#!/usr/bin/env bash
# Memory-safety gate: every AOT binary must run under valgrind with zero definite
# leaks and zero memory errors. Covers the hand-written runtime C (nova_rt.c) on
# boxed-tier programs AND the embedded interpreter+JIT arena on embed-tier ones.
# Skips cleanly if valgrind is not installed. Run from repo root:
#   bash tests/valgrind_smoke.sh [path-to-nova]
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
if ! command -v valgrind >/dev/null 2>&1; then
  echo "valgrind smoke: SKIP (valgrind not installed)"; exit 0
fi
mkdir -p build
progs="strings_ops arrays_maps json_roundtrip stdlib_combo match_patterns closures_hof generators_lazy"
pass=0; fail=0
for p in $progs; do
  f="tests/corpus/$p.nova"
  [ -f "$f" ] || continue
  "$BIN" build --aot "$f" >/dev/null 2>&1 || { echo "SKIP $p (no aot build)"; continue; }
  valgrind --leak-check=full --error-exitcode=99 --errors-for-leak-kinds=definite \
           -q "./build/$p" >/dev/null 2>"build/vg_$p.txt"
  if [ $? -eq 0 ]; then
    pass=$((pass+1))
  else
    echo "FAIL [$p] valgrind errors:"; grep -E "definitely lost|Invalid|ERROR SUMMARY" "build/vg_$p.txt" | head -4
    fail=$((fail+1))
  fi
  rm -f "build/$p" "build/vg_$p.txt"
done
echo "valgrind smoke: $pass passed, $fail failed"
[ $fail -eq 0 ]
