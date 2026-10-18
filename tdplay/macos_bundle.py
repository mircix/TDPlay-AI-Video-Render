"""Build "TDPlay AI Video Render.app" so the app opens from Applications, Launchpad, Spotlight or the Dock.

Two kinds of bundle:

* ``python -m tdplay.macos_bundle`` (what install.command runs) builds a small launcher: its
  executable is a shell script that runs ``python -m tdplay.desktop_app`` from this install folder
  (output goes to ``~/TDPlay/logs/app.log``). Run install.command again if you move the folder. It
  is built on your Mac, so it carries no "downloaded from the internet" flag.
* ``--standalone`` builds the app that goes into the disk image (``macos/build_dmg.sh``). It
  carries the app's code itself; its first launch runs ``macos/setup.sh``, which installs Python,
  MLX, LTX-2 and ffmpeg into ``~/TDPlay/runtime``, and it runs again after an update.
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
  [[ -n "${{TDPLAY_NO_DIALOGS:-}}" ]] || /usr/bin/osascript -e 'display alert "TDPlay AI Video Render" message "Its folder was moved or deleted. Open the TDPlay-AI-Video-Render folder and double-click install.command again."' >/dev/null 2>&1
  exit 1
fi
cd "${{APP_HOME}}" || exit 1
echo "--- $(date) ---" >> "${{LOG_DIR}}/app.log"
exec "${{APP_HOME}}/.venv/bin/python" -m tdplay.desktop_app >> "${{LOG_DIR}}/app.log" 2>&1
"""


# The disk-image app: sets itself up on the first launch, then starts the app from ~/TDPlay/runtime.
STANDALONE_LAUNCHER = r"""#!/bin/bash
# Starts TDPlay AI Video Render. The first launch (and the first after an update) installs what the
# app needs into ~/TDPlay/runtime (see Resources/app/macos/setup.sh); the log is ~/TDPlay/logs/setup.log.
RES="$(cd "$(dirname "${0}")/../Resources" && pwd)"
TD_HOME="${TDPLAY_HOME:-${HOME}/TDPlay}"
RUNTIME="${TD_HOME}/runtime"
LOG_DIR="${TD_HOME}/logs"
SETUP_LOG="${LOG_DIR}/setup.log"
export PATH="${RUNTIME}/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"
mkdir -p "${LOG_DIR}"
rotate() {
  if [[ -f "${1}" ]] && (( $(wc -c < "${1}") > 5000000 )); then mv -f "${1}" "${1}.1"; fi
}
# Dialogs, except with TDPLAY_NO_DIALOGS or TDPLAY_SETUP_ONLY (set up without starting the app): the tests.
dialogs() {
  [[ -x /usr/bin/osascript && -z "${TDPLAY_NO_DIALOGS:-}" && -z "${TDPLAY_SETUP_ONLY:-}" ]]
}
osa() {
  if dialogs; then /usr/bin/osascript "$@" >/dev/null 2>&1; fi
}

want="$(cat "${RES}/app/VERSION")"
have="$(cat "${RUNTIME}/VERSION" 2>/dev/null || true)"
if [[ "${want}" != "${have}" || ! -x "${RUNTIME}/venv/bin/python" ]]; then
  LOCK="${TD_HOME}/.setup-lock"
  if ! mkdir "${LOCK}" 2>/dev/null; then
    pid="$(cat "${LOCK}/pid" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      osa -e 'display dialog "TDPlay AI Video Render is still setting up. It opens when it is ready." buttons {"OK"} default button "OK" with title "TDPlay AI Video Render" giving up after 60'
      exit 0
    fi
  fi
  echo "$$" > "${LOCK}/pid"
  rotate "${SETUP_LOG}"
  echo "--- $(date): setting up ${want} ---" >> "${SETUP_LOG}"
  # A window while it works; it closes when the setup ends. "Show Details" opens the log.
  dialog=""
  if dialogs; then
    /usr/bin/osascript -e 'on run argv' -e 'activate' \
      -e 'set r to display dialog "Setting up TDPlay AI Video Render. The first time, this downloads about 1 GB and takes a few minutes. The app opens when it is ready." buttons {"Show Details", "OK"} default button "OK" with title "TDPlay AI Video Render" giving up after 3600' \
      -e 'if button returned of r is "Show Details" then do shell script "open -a Console " & quoted form of (item 1 of argv)' \
      -e 'end run' "${SETUP_LOG}" >/dev/null 2>&1 &
    dialog=$!
  fi
  status=0
  /bin/bash "${RES}/app/macos/setup.sh" "${RES}/app" "${RUNTIME}" >> "${SETUP_LOG}" 2>&1 || status=$?
  if [[ -n "${dialog}" ]]; then kill "${dialog}" 2>/dev/null; fi
  rm -rf "${LOCK}"
  if (( status != 0 )); then
    reason="$(grep '^ERROR: ' "${SETUP_LOG}" | tail -1 | sed 's/^ERROR: //')"
    osa -e 'on run argv' -e 'activate' \
        -e 'set r to display alert "TDPlay AI Video Render could not finish setting up" message (item 2 of argv) buttons {"Show Log", "OK"} default button "OK"' \
        -e 'if button returned of r is "Show Log" then do shell script "open -a Console " & quoted form of (item 1 of argv)' \
        -e 'end run' "${SETUP_LOG}" "${reason:-Something went wrong.} Open the app again to retry."
    exit 1
  fi
fi
if [[ -n "${TDPLAY_SETUP_ONLY:-}" ]]; then
  exit 0
fi
rotate "${LOG_DIR}/app.log"
cd "${RUNTIME}/app" || exit 1
echo "--- $(date) ---" >> "${LOG_DIR}/app.log"
exec "${RUNTIME}/venv/bin/python" -m tdplay.desktop_app >> "${LOG_DIR}/app.log" 2>&1
"""

#: What the disk-image app carries in Contents/Resources/app.
STANDALONE_FILES = ["tdplay", "requirements.txt", "macos/setup.sh"]


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


def build(dest_dir: Path, root: Path = ROOT, standalone: bool = False, build_id: str = "") -> Path:
    """Write the bundle into ``dest_dir`` (replacing an earlier TDPlay bundle, never anything else).

    ``standalone`` makes the disk-image app (see the module docs); ``build_id`` tells its builds apart,
    so a new build of the same version still sets itself up again.
    """
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
    if standalone:
        launcher.write_text(STANDALONE_LAUNCHER)
        app_dir = contents / "Resources" / "app"
        for name in STANDALONE_FILES:
            src, dst = root / name, app_dir / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)
        (app_dir / "VERSION").write_text(__version__ + (f"+{build_id}" if build_id else "") + "\n")
    else:
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
    parser.add_argument("--standalone", action="store_true", help="build the self-contained app for the disk image")
    parser.add_argument("--build-id", default="", help="with --standalone: tells builds of one version apart")
    args = parser.parse_args(argv)
    if args.dest is None and (sys.platform != "darwin" or args.standalone):
        raise SystemExit("Pass --dest: the folder to build the app in.")
    app = build(args.dest or default_destination(), standalone=args.standalone, build_id=args.build_id)
    if sys.platform == "darwin" and not args.standalone:
        register(app)
    print(f"Installed {app}")


if __name__ == "__main__":
    main()
