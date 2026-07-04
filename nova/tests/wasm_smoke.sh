#!/usr/bin/env bash
# WASM target: `nova build --aot=wasm` compiles the typed (pure int/float +
# string-literal) tier to a freestanding wasm32 module and ships it only if,
# run under node, its output is byte-identical to `nova run`. This exercises a
# few typed programs and confirms (a) a .wasm is produced and (b) node output
# matches the oracle. Skips cleanly if clang or node is missing.
#   bash tests/wasm_smoke.sh [path-to-nova]
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
NODE="$(command -v node || echo /opt/node22/bin/node)"
if ! command -v clang >/dev/null 2>&1 || [ ! -x "$NODE" ]; then
  echo "wasm smoke: SKIP (clang or node absent)"; exit 0
fi
mkdir -p build
harness="build/.wasm_run.mjs"
cat > "$harness" <<'JS'
import { readFileSync } from 'node:fs';
const bytes = readFileSync(process.argv[2]);
const out = []; const dec = new TextDecoder(); let mem;
const env = { print_i64: v => out.push(v.toString()),
  print_str: (p,l) => out.push(dec.decode(new Uint8Array(mem.buffer, Number(p), Number(l)))) };
const { instance } = await WebAssembly.instantiate(bytes, { env });
mem = instance.exports.memory; instance.exports.main();
process.stdout.write(out.length ? out.join("\n") + "\n" : "");
JS

mkdir -p build/wsrc
# typed-eligible sample programs (pure int + string-literal output)
cat > build/wsrc/fib.nova <<'EOF'
fn fib(n){ if n<2 { n } else { fib(n-1)+fib(n-2) } }
fn main(){ print("fib"); print(fib(27)) }
EOF
cat > build/wsrc/loop.nova <<'EOF'
fn tri(n){ let mut s=0; let mut i=1; while i<=n { s=s+i; i=i+1 } s }
fn main(){ print(tri(1000)); print(tri(50)) }
EOF
cat > build/wsrc/gcd.nova <<'EOF'
fn gcd(a,b){ if b==0 { a } else { gcd(b, a - (a/b)*b) } }
fn main(){ print(gcd(1071,462)); print("done") }
EOF

pass=0; fail=0
for f in build/wsrc/*.nova; do
  name=$(basename "$f" .nova)
  expect=$("$BIN" run "$f" 2>&1)
  msg=$("$BIN" build --aot=wasm "build/wsrc/$name.nova" 2>&1 | tail -1)
  w="build/$name.wasm"
  if [ ! -f "$w" ]; then echo "FAIL [$name] no wasm emitted: $msg"; fail=$((fail+1)); continue; fi
  got=$("$NODE" "$harness" "$w" 2>&1)
  if [ "$got" = "$expect" ]; then pass=$((pass+1)); else
    echo "FAIL [$name] wasm output diverged:"; diff <(echo "$expect") <(echo "$got") | head -6; fail=$((fail+1))
  fi
  rm -f "$w"
done
rm -rf build/wsrc "$harness"
echo "wasm smoke: $pass passed, $fail failed"
[ $fail -eq 0 ]
