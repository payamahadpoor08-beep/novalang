#!/usr/bin/env bash
# Minimal LSP smoke: drive `nova lsp` over stdio with initialize + didOpen (of a
# file with a type error) + shutdown/exit, and assert it returns capabilities and
# a diagnostic. Requires python3 (to frame JSON-RPC). Run from repo root.
set -u
BIN="${1:-./target/release/nova}"
cd "$(dirname "$0")/.."
command -v python3 >/dev/null 2>&1 || { echo "lsp smoke: SKIP (python3 absent)"; exit 0; }
python3 - "$BIN" <<'PY'
import subprocess, json, sys
BIN = sys.argv[1]
def frame(o):
    b = json.dumps(o).encode(); return b"Content-Length: %d\r\n\r\n%s" % (len(b), b)
msgs = [
  {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}},
  {"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///t.nova","text":"fn main() {\n  print(y + 1)\n}\n"}}},
  {"jsonrpc":"2.0","id":2,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///t.nova"}}},
  {"jsonrpc":"2.0","id":3,"method":"shutdown"},
  {"jsonrpc":"2.0","method":"exit"},
]
out = subprocess.run([BIN,"lsp"], input=b"".join(frame(m) for m in msgs),
                     capture_output=True, timeout=15).stdout.decode(errors="replace")
ok = ("capabilities" in out) and ("undefined variable: y" in out) and ("publishDiagnostics" in out) and ("hoverProvider" in out)
print("lsp smoke:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
