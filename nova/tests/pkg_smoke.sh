#!/usr/bin/env bash
# Package manager smoke: `nova add` vendors a dep into ./nova_modules, `use "dep"`
# resolves it, and the program runs byte-identical on interp + vm.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/${1:-target/release/nova}"
t="$(mktemp -d)"; cd "$t"
printf 'fn greet(n) { "Hi " + n }\n' > lib.nova
printf 'use "mylib"\nfn main() { print(greet("Nova")) }\n' > app.nova
"$BIN" add lib.nova mylib >/dev/null 2>&1
a=$("$BIN" run app.nova 2>&1); b=$("$BIN" vm app.nova 2>&1)
cd /; rm -rf "$t"
if [ "$a" = "Hi Nova" ] && [ "$b" = "Hi Nova" ]; then echo "pkg smoke: PASS"; else echo "pkg smoke: FAIL ($a | $b)"; exit 1; fi
