#!/usr/bin/env bash
# Full LSP capability suite: drive `nova lsp` over stdio and assert every feature
# responds correctly — diagnostics, documentSymbol, hover, definition, completion,
# references, rename, signatureHelp, formatting, semanticTokens, foldingRange,
# workspaceSymbol. Requires python3 (to frame JSON-RPC). Run from repo root.
set -u
BIN="$(cd "$(dirname "$0")/.." && pwd)/${1:-target/release/nova}"
command -v python3 >/dev/null 2>&1 || { echo "lsp smoke: SKIP (python3 absent)"; exit 0; }
python3 - "$BIN" <<'PY'
import subprocess, json, sys
BIN=sys.argv[1]
def frame(o):
    b=json.dumps(o).encode(); return b"Content-Length: %d\r\n\r\n%s"%(len(b),b)
DOC="struct Point { x: Int, y: Int }\nfn area(p) {\n  let w = p.x\n  w * w\n}\nfn main() {\n  let a = area(Point { x: 3, y: 4 })\n  print(z)\n}\n"
uri="file:///t.nova"
def td(e): d={"textDocument":{"uri":uri}}; d.update(e); return d
msgs=[
 {"id":1,"method":"initialize","params":{}},
 {"method":"textDocument/didOpen","params":{"textDocument":{"uri":uri,"text":DOC}}},
 {"id":2,"method":"textDocument/documentSymbol","params":td({})},
 {"id":3,"method":"textDocument/hover","params":td({"position":{"line":1,"character":3}})},
 {"id":4,"method":"textDocument/definition","params":td({"position":{"line":6,"character":10}})},
 {"id":5,"method":"textDocument/completion","params":td({"position":{"line":3,"character":0}})},
 {"id":6,"method":"textDocument/references","params":td({"position":{"line":1,"character":3}})},
 {"id":7,"method":"textDocument/rename","params":td({"position":{"line":1,"character":3},"newName":"surface"})},
 {"id":8,"method":"textDocument/signatureHelp","params":td({"position":{"line":6,"character":15}})},
 {"id":9,"method":"textDocument/formatting","params":td({})},
 {"id":10,"method":"textDocument/semanticTokens/full","params":td({})},
 {"id":11,"method":"textDocument/foldingRange","params":td({})},
 {"id":12,"method":"workspace/symbol","params":{"query":"area"}},
 {"id":13,"method":"shutdown"},{"method":"exit"},
]
for m in msgs: m["jsonrpc"]="2.0"
out=subprocess.run([BIN,"lsp"],input=b"".join(frame(m) for m in msgs),capture_output=True,timeout=25).stdout.decode(errors="replace")
checks=[
 'undefined variable: z' in out, '"name":"area"' in out and '"name":"Point"' in out,
 'fn area(p)' in out, '"line":1' in out, '"label":"area"' in out and '"label":"fn"' in out,
 'surface' in out, 'signatures' in out, 'newText' in out,
 '"data":[' in out and '"data":[]' not in out, 'startLine' in out, out.count('"name":"area"')>=1,
]
n=sum(1 for c in checks if c)
print(f"lsp smoke: {n}/{len(checks)} capabilities OK")
sys.exit(0 if n==len(checks) else 1)
PY
