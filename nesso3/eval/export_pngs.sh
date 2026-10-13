#!/bin/bash
# Rasterise the SVG figures to PNG (for platforms that do not accept SVG). Needs Google Chrome.
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
cd "$(dirname "$0")/../images" || exit 1
for f in *.svg; do
  W=$(grep -o "width='[0-9]*'" "$f" | head -1 | tr -dc 0-9); H=$(grep -o "viewBox='0 0 [0-9]* [0-9]*'" "$f" | awk '{print $4}' | tr -dc 0-9)
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size="$W,$H" --screenshot="${f%.svg}.png" "file://$PWD/$f" >/dev/null 2>&1
done
ls -la *.png
