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

# The JIT backend (Cranelift) can only target x86-64 / aarch64 / riscv64 / s390x.
# On other hosts — notably 32-bit ARM (Termux on many phones reports `armv7l`) —
# cranelift fails to even compile, so we build with `--no-default-features`, which
# turns the JIT off. The interpreter, bytecode VM, and AOT-via-C backend need
# nothing from cranelift and stay fully functional.
BUILD_FLAGS=""
case "$(uname -m)" in
  armv6l|armv7l|armv8l|arm)
    echo "note: $(uname -m) has no JIT backend (cranelift can't target it) — building without the JIT."
    echo "      interpreter, bytecode VM, and AOT (nova build) all still work."
    BUILD_FLAGS="--no-default-features"
    ;;
esac

echo "building Nova (release)..."
if ! cargo build --release $BUILD_FLAGS; then
  # Belt-and-suspenders: if a full build failed on an arch we didn't special-case
  # (some other cranelift-unsupported host), retry once with the JIT disabled.
  if [ -z "$BUILD_FLAGS" ]; then
    echo "note: full build failed — retrying without the JIT (--no-default-features)..." >&2
    cargo build --release --no-default-features
  else
    exit 1
  fi
fi

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
