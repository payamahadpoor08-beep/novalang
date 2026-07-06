#!/usr/bin/env bash
# Smoke test for the Demon (background) Compiler: `nova demon <file>` +
# the `#[demon(...)]` attribute. Covers a batch pass (parse → live diagnostics
# → hot-reload run), an error-diagnostics pass, and a daemon-mode incremental
# reload (mutate one function, confirm only it recompiles while the rest are
# reused from the intelligent cache).
set -uo pipefail
NOVA="${1:-./target/release/nova}"
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
fail=0

# --- 1. batch pass: clean build runs through the whole pipeline --------------
cat > "$D/app.nova" <<'X'
#[demon(mode: "batch", watch: ["."], cache: true, hot_reload: true, incremental: true, diagnostics: true, optimize: "speed")]
fn greet(name) { "hello " + name }
fn main() { print(greet("nova")) }
X
out=$("$NOVA" demon "$D/app.nova" 2>&1)
echo "$out" | grep -q "no errors"      || { echo "FAIL: batch diagnostics"; fail=1; }
echo "$out" | grep -q "hello nova"     || { echo "FAIL: batch program output"; fail=1; }
echo "$out" | grep -q "hot-reload: ✓"  || { echo "FAIL: batch hot-reload"; fail=1; }

# --- 2. the #[demon] attribute is captured + introspectable ------------------
cat > "$D/i.nova" <<'X'
#[demon(mode: "daemon", watch: ["src/"], cache: true)]
fn main() { print(attrs_of("main")) }
X
[ "$("$NOVA" run "$D/i.nova")" = '["demon"]' ] || { echo "FAIL: attrs_of demon"; fail=1; }

# --- 3. batch error diagnostics ---------------------------------------------
cat > "$D/bad.nova" <<'X'
#[demon(mode: "batch", diagnostics: true)]
fn main() { print(undefined_var) }
X
"$NOVA" demon "$D/bad.nova" 2>&1 | grep -q "1 error" || { echo "FAIL: error diagnostics"; fail=1; }

# --- 4. daemon-mode incremental reload --------------------------------------
cat > "$D/live.nova" <<'X'
#[demon(mode: "daemon", cache: true, incremental: true, diagnostics: true, hot_reload: true)]
fn greet(n) { "hi " + n }
fn other(x) { x * 2 }
fn main() { print(greet("a")) }
X
"$NOVA" demon "$D/live.nova" > "$D/out.txt" 2>&1 &
pid=$!
sleep 1
sed -i 's/"hi " + n/"HELLO " + n/' "$D/live.nova"
sleep 1
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
grep -q 'recompiled \["greet"\]' "$D/out.txt" || { echo "FAIL: incremental recompile"; cat "$D/out.txt"; fail=1; }
grep -q '2 reused'               "$D/out.txt" || { echo "FAIL: incremental reuse"; fail=1; }

[ "$fail" -eq 0 ] && echo "demon smoke: OK" || { echo "demon smoke: FAILED"; exit 1; }
