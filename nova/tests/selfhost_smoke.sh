#!/usr/bin/env bash
# Self-hosting stage-1 gate: the Nova lexer written in Nova (selfhost/lexer.nova)
# must produce a token dump BYTE-IDENTICAL to the Rust reference (`nova tokens`)
# for every corpus, std, and example file — including lexing itself. The lexer
# is a Nova program, so the 4-tier discipline applies to it too: interp, vm,
# vm --no-jit, and vm --jit outputs must all agree.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/${1:-target/release/nova}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
t="$(mktemp -d)"; trap 'rm -rf "$t"' EXIT

pass=0; fail=0
for f in tests/corpus/*.nova std/*.nova examples/*.nova selfhost/lexer.nova; do
  "$BIN" tokens "$f" > "$t/rust.txt" 2>/dev/null
  "$BIN" run selfhost/lexer.nova "$f" > "$t/nova.txt" 2>/dev/null
  if cmp -s "$t/rust.txt" "$t/nova.txt"; then pass=$((pass+1))
  else fail=$((fail+1)); echo "  DIFF (rust vs nova lexer): $f"; fi
done
echo "selfhost lexer vs rust reference: $pass identical, $fail diffs"

# 4-tier discipline on the lexer itself (one representative input)
f=std/list.nova
"$BIN" run          selfhost/lexer.nova "$f" > "$t/a.txt" 2>&1
"$BIN" vm           selfhost/lexer.nova "$f" > "$t/b.txt" 2>&1
"$BIN" vm --no-jit  selfhost/lexer.nova "$f" > "$t/c.txt" 2>&1
"$BIN" vm --jit     selfhost/lexer.nova "$f" > "$t/d.txt" 2>&1
if cmp -s "$t/a.txt" "$t/b.txt" && cmp -s "$t/a.txt" "$t/c.txt" && cmp -s "$t/a.txt" "$t/d.txt"; then
  echo "selfhost lexer 4-tier: identical"
else
  echo "selfhost lexer 4-tier: DIVERGED"; fail=$((fail+1))
fi

[ "$fail" -eq 0 ] || exit 1
