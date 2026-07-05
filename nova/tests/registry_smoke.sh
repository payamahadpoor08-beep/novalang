#!/usr/bin/env bash
# Registry smoke: the full package-manager loop over a real HTTP registry.
#   publish → `nova registry` HTTP serve → resolve (semver) → fetch → sha256
#   verify → unpack (.nvpkg) → vendor → run byte-identical (interp + vm).
# Also covers: a path dependency, two-mode `[abilities]` (an attribute declared
# in nova.hgx, not on the code), and checksum-tamper rejection.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/${1:-target/release/nova}"
PORT="${PORT:-7879}"
PASS=0; FAIL=0
ok()   { echo "  ok: $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
root="$(mktemp -d)"; trap 'rm -rf "$root"; [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null' EXIT
cd "$root"

# --- 1) publish a package to a local index ----------------------------------
mkdir -p greeter/src index
printf 'fn greet(n) { "Hi " + n }\n' > greeter/src/lib.nova
printf '[package]\nname = "greeter"\nversion = "1.2.0"\n' > greeter/nova.hgx
( cd greeter && "$BIN" publish ../index ) >/dev/null 2>&1
[ -f index/greeter/index.txt ] && [ -f index/greeter/greeter-1.2.0.nvpkg ] \
  && ok "publish writes archive + index.txt" || bad "publish"

# --- 2) serve the index over HTTP -------------------------------------------
"$BIN" registry index --port "$PORT" >/dev/null 2>&1 & SRV=$!
sleep 1

# --- 3) an app depends on greeter ^1.0 from the http registry ---------------
mkdir app
printf 'use "greeter/src/lib"\nfn main() { print(greet("Nova")) }\n' > app/main.nova
printf '[package]\nname = "app"\nversion = "0.1.0"\nentry = "main.nova"\n[registry]\ndefault = "http://127.0.0.1:%s"\n[dependencies]\ngreeter = "^1.0"\n' "$PORT" > app/nova.hgx
( cd app && "$BIN" install ) >/dev/null 2>&1
grep -q '1.2.0' app/nova.lock && ok "resolve+lock picks 1.2.0 for ^1.0" || bad "lock"
[ -f app/nova_modules/greeter/src/lib.nova ] && ok "vendored (unpacked .nvpkg)" || bad "vendor"

a=$(cd app && "$BIN" run main.nova 2>&1); b=$(cd app && "$BIN" vm main.nova 2>&1)
[ "$a" = "Hi Nova" ] && [ "$b" = "Hi Nova" ] && ok "runs byte-identical (interp=vm)" || bad "run ($a|$b)"

# --- 4) a path dependency + a manifest-declared ability ---------------------
mkdir p; cd p
mkdir liblocal
printf 'fn add(a, b) { a + b }\n' > liblocal/mod.nova
printf '[package]\nname = "p"\nversion = "0.1.0"\nentry = "main.nova"\n[dependencies]\nliblocal = { path = "./liblocal" }\n[abilities]\ntraced = { attr = "trace", targets = ["main"] }\n' > nova.hgx
printf 'use "liblocal/mod"\nfn main() { print(add(2, 3)) }\n' > main.nova
"$BIN" install >/dev/null 2>&1
[ -f nova_modules/liblocal/mod.nova ] && ok "path dep vendored" || bad "path dep"
out=$("$BIN" run main.nova 2>&1)
echo "$out" | grep -q '^5$' && echo "$out" | grep -q 'trace: main' \
  && ok "two-mode ability: manifest #[trace] applied to code" || bad "manifest ability ($out)"
cd "$root"

# --- 5) checksum-tamper rejection -------------------------------------------
mkdir -p pkg/src idx2 app2
printf 'fn v() { 42 }\n' > pkg/src/lib.nova
printf '[package]\nname = "pkg"\nversion = "1.0.0"\n' > pkg/nova.hgx
( cd pkg && "$BIN" publish ../idx2 ) >/dev/null 2>&1
printf 'CORRUPT' >> idx2/pkg/pkg-1.0.0.nvpkg
kill "$SRV" 2>/dev/null
"$BIN" registry idx2 --port "$PORT" >/dev/null 2>&1 & SRV=$!
sleep 1
printf '[package]\nname = "app2"\nversion = "0.1.0"\n[registry]\ndefault = "http://127.0.0.1:%s"\n[dependencies]\npkg = "^1.0"\n' "$PORT" > app2/nova.hgx
( cd app2 && "$BIN" install ) 2>&1 | grep -q 'checksum mismatch' \
  && ok "tampered download rejected (sha256)" || bad "tamper not caught"

echo "registry smoke: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
