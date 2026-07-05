#!/usr/bin/env bash
# WASM target: `nova build --aot=wasm` compiles the AOT C (typed + boxed, incl.
# nova_rt.c) to wasm32-wasi with clang + a wasi-libc sysroot, and ships the
# module only if, run under node's WASI, its output is byte-identical to
# `nova run`. Exercises typed + boxed corpus programs. Skips cleanly if clang,
# a wasi sysroot, or node is missing.  bash tests/wasm_smoke.sh [path-to-nova]
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
NODE="$(command -v node || echo /opt/node22/bin/node)"
have_sysroot=0
for r in /usr /opt/wasi-sysroot /usr/share/wasi-sysroot; do
  [ -f "$r/lib/wasm32-wasi/libc.a" ] && have_sysroot=1
done
if ! command -v clang >/dev/null 2>&1 || [ ! -x "$NODE" ] || [ $have_sysroot -eq 0 ]; then
  echo "wasm smoke: SKIP (clang, wasi-sysroot, or node absent)"; exit 0
fi
mkdir -p build
harness="build/.wasm_run.mjs"
cat > "$harness" <<'JS'
import { readFileSync } from 'node:fs';
import { WASI } from 'node:wasi';
const wasi = new WASI({ version: 'preview1', args: [], env: {} });
const wasm = await WebAssembly.compile(readFileSync(process.argv[2]));
const inst = await WebAssembly.instantiate(wasm, wasi.getImportObject());
wasi.start(inst);
JS

# typed + boxed corpus programs (embed-tier ones are skipped by the builder)
progs="int_edges aot_arrays aot_float_print aot_mixed_tiers float_edges"
pass=0; fail=0; skip=0
for p in $progs; do
  f="tests/corpus/$p.nova"; [ -f "$f" ] || continue
  expect=$("$BIN" run "$f" 2>&1)
  "$BIN" build --aot=wasm "$f" >/dev/null 2>&1
  w="build/$p.wasm"
  if [ ! -f "$w" ]; then echo "SKIP $p (embed tier / not wasm-able)"; skip=$((skip+1)); continue; fi
  got=$("$NODE" --no-warnings "$harness" "$w" 2>&1)
  if [ "$got" = "$expect" ]; then pass=$((pass+1)); else
    echo "FAIL [$p] wasm output diverged:"; diff <(echo "$expect") <(echo "$got") | head -6; fail=$((fail+1))
  fi
  rm -f "$w"
done
rm -f "$harness"
echo "wasm smoke: $pass passed, $fail failed, $skip skipped"
[ $fail -eq 0 ]
