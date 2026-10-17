#!/bin/bash
# Build dist/TDPlay-AI-Video-Render-<version>.dmg on a Mac: the self-contained app, a link to
# Applications to drag it onto, and a note on opening it the first time.
# Usage: bash macos/build_dmg.sh [BUILD_ID]   (the GitHub workflow passes the commit)
set -euo pipefail
cd "$(dirname "${0}")/.."

build_id="${1:-$(git rev-parse --short HEAD 2>/dev/null || true)}"
version="$(python3 -c 'import re; print(re.search(r"__version__ = \"(.+?)\"", open("tdplay/__init__.py").read()).group(1))')"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT

python3 -m tdplay.macos_bundle --standalone --dest "${stage}" --build-id "${build_id}"
ln -s /Applications "${stage}/Applications"
cp macos/dmg-readme.txt "${stage}/If the app does not open.txt"

mkdir -p dist
out="dist/TDPlay-AI-Video-Render-${version}.dmg"
hdiutil create -volname "TDPlay AI Video Render" -srcfolder "${stage}" -fs HFS+ -format UDZO -ov "${out}"
echo "Built ${out}"
