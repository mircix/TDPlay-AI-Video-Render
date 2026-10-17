"""Build "TDPlay AI Video Render.app" so the app opens from Applications, Launchpad, Spotlight or the Dock.

install.command runs ``python -m tdplay.macos_bundle``. The bundle is a small launcher: its executable
is a shell script that runs ``python -m tdplay.mac_app`` from this install folder (output goes to
``~/TDPlay/logs/app.log``). Run install.command again if you move the folder.

The bundle is built on your Mac, so it carries no "downloaded from the internet" flag and opens
without a Gatekeeper prompt.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from tdplay import __version__

APP_NAME = "TDPlay AI Video Render"
BUNDLE_ID = "com.tdplay.ai-video-render"
EXECUTABLE = "TDPlay"
ROOT = Path(__file__).resolve().parent.parent
LSREGISTER = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/"
              "lsregister")

# Plain ASCII with every variable braced: macOS runs this with bash 3.2.
LAUNCHER = """#!/bin/bash
# Starts TDPlay AI Video Render. Written by install.command; run it again if you move the app folder.
APP_HOME={home}
LOG_DIR="${{HOME}}/TDPlay/logs"
export PATH="/opt/homebrew/bin:/usr/local/bin:${{PATH}}"
mkdir -p "${{LOG_DIR}}"
if [[ -f "${{LOG_DIR}}/app.log" ]] && (( $(wc -c < "${{LOG_DIR}}/app.log") > 5000000 )); then
  mv -f "${{LOG_DIR}}/app.log" "${{LOG_DIR}}/app.log.1"
fi
if [[ ! -x "${{APP_HOME}}/.venv/bin/python" ]]; then
  /usr/bin/osascript -e 'display alert "TDPlay AI Video Render" message "Its folder was moved or deleted. Open the TDPlay-AI-Video-Render folder and double-click install.command again."' >/dev/null 2>&1
  exit 1
fi
cd "${{APP_HOME}}" || exit 1
echo "--- $(date) ---" >> "${{LOG_DIR}}/app.log"
exec "${{APP_HOME}}/.venv/bin/python" -m tdplay.mac_app >> "${{LOG_DIR}}/app.log" 2>&1
"""


def info_plist() -> dict:
    return {
        "CFBundleName": "TDPlay",
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": EXECUTABLE,
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.video",
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
    }


def default_destination() -> Path:
    system = Path("/Applications")
    if system.is_dir() and os.access(system, os.W_OK):
        return system
    return Path.home() / "Applications"


def build(dest_dir: Path, root: Path = ROOT) -> Path:
    """Write the bundle into ``dest_dir`` (replacing an earlier TDPlay bundle, never anything else)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    app = dest_dir / f"{APP_NAME}.app"
    if app.exists():
        plist = app / "Contents" / "Info.plist"
        owner = plistlib.loads(plist.read_bytes()).get("CFBundleIdentifier") if plist.exists() else None
        if owner != BUNDLE_ID:
            raise RuntimeError(f"{app} already exists and is not TDPlay's; not replacing it.")
    tmp = dest_dir / f".{APP_NAME}.app.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    contents = tmp / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()
    (contents / "Info.plist").write_bytes(plistlib.dumps(info_plist()))
    (contents / "PkgInfo").write_text("APPL????")
    launcher = contents / "MacOS" / EXECUTABLE
    launcher.write_text(LAUNCHER.format(home=shlex.quote(str(root.resolve()))))
    launcher.chmod(0o755)
    shutil.copy2(root / "tdplay" / "assets" / "AppIcon.icns", contents / "Resources" / "AppIcon.icns")
    if app.exists():
        shutil.rmtree(app)
    tmp.rename(app)
    return app


def register(app: Path) -> None:
    """Tell Finder and the Dock about the new bundle so its icon shows right away."""
    os.utime(app)
    if Path(LSREGISTER).exists():
        subprocess.run([LSREGISTER, "-f", str(app)], capture_output=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tdplay.macos_bundle", description="Install the Mac app bundle.")
    parser.add_argument("--dest", type=Path, help="folder to put the app in (default: /Applications or ~/Applications)")
    args = parser.parse_args(argv)
    if args.dest is None and sys.platform != "darwin":
        raise SystemExit("The app bundle is for macOS; pass --dest to build one elsewhere.")
    app = build(args.dest or default_destination())
    if sys.platform == "darwin":
        register(app)
    print(f"Installed {app}")


if __name__ == "__main__":
    main()
