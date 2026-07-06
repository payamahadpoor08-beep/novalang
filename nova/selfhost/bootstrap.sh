#!/usr/bin/env bash
# Stage 5 — the fixpoint bootstrap.
#
# Stages 1-4 prove each Nova front-end (lexer/parser/checker/eval, written in
# Nova) reproduces its Rust reference. Stage 5 closes the loop: run the Nova
# front-end *under the Nova evaluator* and confirm the doubly-interpreted result
# still equals the Rust reference — i.e. the Nova-implemented compiler, executed
# by the Nova-implemented evaluator, reproduces the Rust binary.
#
#   nova run eval.nova lexer.nova   F   ==  nova tokens F
#   nova run eval.nova parser.nova  F   ==  nova ast    F
#   nova run eval.nova checker.nova F   ==  nova check  F
#
# This is a tree-walker driving a tree-walker driving a tokenizer, so it runs on
# small inputs (the fixpoint is about convergence, not speed).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/${1:-target/release/nova}"
cd "$ROOT"
TO=180
fail=0

# a handful of small, representative real files (fast under double interpretation)
FILES=$(ls -S tests/corpus/*.nova std/*.nova 2>/dev/null | tail -4)

ck() { # ck <label> <reference-cmd...> -- <nova-under-eval-cmd...>
  local label="$1"; shift
  local ref=(); local sub=()
  while [ "$1" != "--" ]; do ref+=("$1"); shift; done; shift
  sub=("$@")
  if cmp -s <(timeout $TO "${ref[@]}" 2>&1) <(timeout $TO "${sub[@]}" 2>&1); then
    return 0
  else
    echo "  fixpoint DIFF ($label): ${sub[*]}"; return 1
  fi
}

lp=0; pp=0; cp=0; n=0
for f in $FILES; do
  n=$((n+1))
  ck "lexer"   "$BIN" tokens "$f" -- "$BIN" run selfhost/eval.nova selfhost/lexer.nova   "$f" && lp=$((lp+1)) || fail=1
  ck "parser"  "$BIN" ast    "$f" -- "$BIN" run selfhost/eval.nova selfhost/parser.nova  "$f" && pp=$((pp+1)) || fail=1
  ck "checker" "$BIN" check  "$f" -- "$BIN" run selfhost/eval.nova selfhost/checker.nova "$f" && cp=$((cp+1)) || fail=1
done

echo "bootstrap fixpoint (eval∘front-end vs Rust reference) over $n files:"
echo "  eval∘lexer   == nova tokens : $lp/$n"
echo "  eval∘parser  == nova ast    : $pp/$n"
echo "  eval∘checker == nova check  : $cp/$n"

[ "$fail" -eq 0 ] && echo "bootstrap: CONVERGED ✓" || { echo "bootstrap: DIVERGED"; exit 1; }
