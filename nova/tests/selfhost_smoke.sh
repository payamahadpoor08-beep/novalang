#!/usr/bin/env bash
# Self-hosting gate — the Nova compiler front-end written in Nova, byte-verified
# against the Rust reference over EVERY real file (tests/corpus + std + examples
# + the selfhost sources themselves). No curated subset.
#   stage 1  lexer   selfhost/lexer.nova   vs  `nova tokens`
#   stage 2  parser  selfhost/parser.nova  vs  `nova ast`
# Each Nova front-end is itself a Nova program, so the 4-tier discipline
# (interp / vm / vm --no-jit / vm --jit) applies to it too.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/${1:-target/release/nova}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
t="$(mktemp -d)"; trap 'rm -rf "$t"' EXIT

FILES=$(ls tests/corpus/*.nova std/*.nova examples/*.nova selfhost/*.nova)
total=$(echo "$FILES" | wc -w | tr -d ' ')
fail=0

# --- stage 1: lexer ---------------------------------------------------------
lp=0
for f in $FILES; do
  if cmp -s <("$BIN" tokens "$f" 2>/dev/null) <("$BIN" run selfhost/lexer.nova "$f" 2>/dev/null); then lp=$((lp+1))
  else echo "  lexer DIFF: $f"; fail=$((fail+1)); fi
done
echo "self-host lexer:  $lp/$total byte-identical vs \`nova tokens\`"

# --- stage 2: parser --------------------------------------------------------
pp=0
for f in $FILES; do
  if cmp -s <("$BIN" ast "$f" 2>/dev/null) <("$BIN" run selfhost/parser.nova "$f" 2>/dev/null); then pp=$((pp+1))
  else echo "  parser DIFF: $f"; fail=$((fail+1)); fi
done
echo "self-host parser: $pp/$total byte-identical vs \`nova ast\`"

# --- stage 3: checker (name resolution + unused-local diagnostics) ----------
cp=0
for f in $FILES; do
  if cmp -s <("$BIN" check "$f" 2>&1) <("$BIN" run selfhost/checker.nova "$f" 2>&1); then cp=$((cp+1))
  else echo "  checker DIFF: $f"; fail=$((fail+1)); fi
done
echo "self-host checker: $cp/$total byte-identical vs \`nova check\`"

# --- stage 4: evaluator (program-output parity vs `nova run`) ----------------
# The Nova evaluator interprets a Nova program and must print byte-identically to
# `nova run`. It is a tree-walker running on top of the reference tree-walker, so
# the three compute-heavy benchmark kernels (aot_mandel, jit_arrays,
# numeric_mixed) need a generous per-file timeout; correctness is identical.
ep=0
for f in $FILES; do
  case "$f" in *lexer.nova|*parser.nova|*checker.nova|*eval.nova) continue;; esac
  if cmp -s <(timeout 300 "$BIN" run "$f" 2>&1) <(timeout 300 "$BIN" run selfhost/eval.nova "$f" 2>&1); then ep=$((ep+1))
  else echo "  eval DIFF: $f"; fail=$((fail+1)); fi
done
et=$(echo "$FILES" | tr ' ' '\n' | grep -vE 'lexer\.nova|parser\.nova|checker\.nova|eval\.nova' | grep -c .)
echo "self-host eval:    $ep/$et byte-identical vs \`nova run\`"

# --- 4-tier discipline on the Nova front-ends themselves ---------------------
for prog in selfhost/lexer.nova selfhost/parser.nova selfhost/checker.nova selfhost/eval.nova; do
  f=std/list.nova
  "$BIN" run         "$prog" "$f" > "$t/a" 2>&1
  "$BIN" vm          "$prog" "$f" > "$t/b" 2>&1
  "$BIN" vm --no-jit "$prog" "$f" > "$t/c" 2>&1
  "$BIN" vm --jit    "$prog" "$f" > "$t/d" 2>&1
  if cmp -s "$t/a" "$t/b" && cmp -s "$t/a" "$t/c" && cmp -s "$t/a" "$t/d"; then
    echo "$(basename "$prog") 4-tier: identical"
  else echo "$(basename "$prog") 4-tier: DIVERGED"; fail=$((fail+1)); fi
done

[ "$fail" -eq 0 ] || exit 1
