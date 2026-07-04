#!/usr/bin/env bash
# Obfuscation must be behaviour-preserving: for every corpus program, running the
# `nova obfuscate` output must produce byte-identical results to running the
# original. Run from repo root:  bash tests/obfuscate_smoke.sh [path-to-nova]
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
pass=0; fail=0
for f in tests/corpus/*.nova; do
  name=$(basename "$f")
  expect=$(timeout 30 "$BIN" run "$f" 2>&1); rc=$?
  if ! "$BIN" obfuscate "$f" > "$tmp/$name" 2>"$tmp/err"; then
    echo "FAIL [$name] obfuscate errored:"; head -3 "$tmp/err"; fail=$((fail+1)); continue
  fi
  got=$(timeout 30 "$BIN" run "$tmp/$name" 2>&1); grc=$?
  if [ "$got" = "$expect" ] && [ "$grc" = "$rc" ]; then
    pass=$((pass+1))
  else
    echo "FAIL [$name] obfuscated output diverged:"
    diff <(echo "$expect") <(echo "$got") | head -6
    fail=$((fail+1))
  fi
done
echo "obfuscate smoke: $pass passed, $fail failed"
[ $fail -eq 0 ]
