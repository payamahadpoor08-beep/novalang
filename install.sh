#!/usr/bin/env bash
# Nova installer: builds the release binary and puts `nova` on your PATH.
# Usage: ./install.sh [--prefix DIR]   (default: ~/.local/bin, falls back to /usr/local/bin)
set -euo pipefail

PREFIX=""
if [ "${1:-}" = "--prefix" ] && [ -n "${2:-}" ]; then PREFIX="$2"; fi

command -v cargo >/dev/null 2>&1 || {
  echo "error: Rust toolchain not found — install it from https://rustup.rs" >&2
  exit 1
}

cd "$(dirname "$0")/nova"
echo "building Nova (release)..."
cargo build --release

BIN="target/release/nova"
[ -x "$BIN" ] || { echo "error: build produced no binary" >&2; exit 1; }

if [ -z "$PREFIX" ]; then
  if [ -d "$HOME/.local/bin" ] || mkdir -p "$HOME/.local/bin" 2>/dev/null; then
    PREFIX="$HOME/.local/bin"
  else
    PREFIX="/usr/local/bin"
  fi
fi

mkdir -p "$PREFIX"
install -m 755 "$BIN" "$PREFIX/nova"
echo "installed: $PREFIX/nova"
"$PREFIX/nova" version

case ":$PATH:" in
  *":$PREFIX:"*) ;;
  *) echo "note: add $PREFIX to your PATH (e.g. export PATH=\"$PREFIX:\$PATH\")" ;;
esac
