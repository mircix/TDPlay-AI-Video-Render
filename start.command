#!/bin/bash
# Opens TDPlay AI Video Render in your browser. Double-click it, or run:  bash start.command
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:${PATH}"   # ffmpeg from Homebrew, also when started from Finder

if [[ ! -x .venv/bin/python ]]; then
  echo "TDPlay AI Video Render isn't installed yet. Double-click install.command first."
  read -r -p "Press Return to close." _
  exit 1
fi

echo "Starting TDPlay AI Video Render... (keep this window open; close it to quit)"
# caffeinate stops the Mac from idle-sleeping in the middle of a long render.
exec caffeinate -i .venv/bin/python -m tdplay
