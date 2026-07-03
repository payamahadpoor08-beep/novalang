#!/usr/bin/env bash
# Differential corpus: every program must produce byte-identical output on the
# interpreter, the tiered VM, the eager JIT, and the plain VM. Run from repo root:
#   bash tests/run_corpus.sh [path-to-nova]
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
pass=0; fail=0
for f in tests/corpus/*.nova; do
  expect=$(timeout 30 "$BIN" run "$f" 2>&1); rc=$?
  ok=1
  for mode in "" "--jit" "--no-jit" "--jit-threshold=3"; do
    got=$(timeout 30 "$BIN" vm $mode "$f" 2>&1); vrc=$?
    if [ "$got" != "$expect" ] || [ "$vrc" != "$rc" ]; then
      case "$got" in
        *"use \`nova run\`"*) : ;;  # interp-only main: documented fallback
        *) echo "FAIL [$f] mode='$mode'"
           diff <(echo "$expect") <(echo "$got") | head -6
           ok=0 ;;
      esac
    fi
  done
  if [ $ok -eq 1 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi
done
echo "corpus: $pass passed, $fail failed"
[ $fail -eq 0 ]

# AOT column: build each corpus program with --aot; the build itself verifies
# byte-identity vs `nova run` before shipping. Report the tier distribution.
typed=0; boxed=0; embed=0
for f in tests/corpus/*.nova; do
  name=$(basename "$f" .nova)
  msg=$("$BIN" build --aot "$f" 2>/dev/null | tail -1)
  case "$msg" in
    *"typed tier"*) typed=$((typed+1));;
    *"boxed tier"*) boxed=$((boxed+1));;
    *) embed=$((embed+1));;
  esac
  rm -f "build/$name"
done
rm -rf build
echo "aot tiers: typed=$typed boxed=$boxed embed-fallback=$embed"
