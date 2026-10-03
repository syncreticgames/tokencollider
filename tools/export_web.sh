#!/usr/bin/env bash
# Build the browser viewport into tokencollider/web/, where `tokencollider view`
# serves it from. CI runs the same script before building the wheel.
# Needs Godot 4 and its web export templates, at the same version.
# TOKENCOLLIDER_GODOT picks the binary (default: godot on PATH).
set -euo pipefail
cd "$(dirname "$0")/.."
godot=${TOKENCOLLIDER_GODOT:-godot}
out="$PWD/tokencollider/web"
rm -rf "$out"
mkdir -p "$out"
"$godot" --headless --path frontend --import
"$godot" --headless --path frontend --export-release Web "$out/index.html"
ls -l "$out"
