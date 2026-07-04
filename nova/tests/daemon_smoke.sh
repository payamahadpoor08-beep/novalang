#!/usr/bin/env bash
# Smoke test for `nova daemon`: load, run, incremental reload, hot reload.
set -euo pipefail
NOVA="${1:-./target/release/nova}"
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

cat > "$D/app.nova" <<'X'
fn greet(name) { "hello " + name }
fn add(a, b) { a + b }
fn main() { print(greet("nova")); print(add(2, 3)) }
X

# Drive the daemon: run, mutate only greet, reload (incremental), run again.
out=$(
  {
    echo "load $D/app.nova"
    echo "run $D/app.nova"
    sed -i 's/"hello " + name/"HELLO " + name + "!"/' "$D/app.nova"
    echo "reload $D/app.nova"
    echo "run $D/app.nova"
    echo "quit"
  } | "$NOVA" daemon
)

echo "$out"
echo "$out" | grep -q "hello nova"          || { echo "FAIL: first run"; exit 1; }
echo "$out" | grep -q '1 changed \["greet"\], 2 reused' || { echo "FAIL: incremental reload"; exit 1; }
echo "$out" | grep -q "HELLO nova!"          || { echo "FAIL: hot reload"; exit 1; }
echo "daemon smoke test: OK"
