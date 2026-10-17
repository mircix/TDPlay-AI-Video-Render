"""TDPlay AI Video Render as a Mac app: the interface in its own window, no Terminal or browser tab.

The app in /Applications (made by install.command, see ``tdplay.macos_bundle``) runs
``python -m tdplay.mac_app``. It starts the local server, shows it in a native WebKit window
(pywebview), keeps the Mac from idle-sleeping while it is open, and stops a running render when
you quit. If a render, upscale or download is running, quitting asks first.
"""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "TDPlay AI Video Render"
ICON = Path(__file__).resolve().parent / "assets" / "AppIcon.png"
QUIT_WHILE_BUSY = ("A video is still being made. Quit anyway?\n\nFinished segments are kept: select the project in "
                   "the Projects tab and press Resume next time.")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(url: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"The app's server did not start within {timeout:.0f} s")


def _dark_mode() -> bool:
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True)
        return out.stdout.strip() == "Dark"
    except OSError:
        return False


def _wants_dark(theme: str) -> bool:
    return theme == "dark" or (theme == "system" and _dark_mode())


def _show_app_name_in_menu_bar() -> None:
    """The process is Python, so macOS would title the menu bar "Python"; use the app's name instead."""
    try:
        from Foundation import NSBundle

        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = APP_NAME
    except Exception:
        pass


def main() -> None:
    from tdplay import app

    port = int(os.environ.get("TDPLAY_PORT") or _free_port())
    url = f"http://127.0.0.1:{port}/"
    app.launch(port=port, open_browser=False, block=False)
    _wait_for(url)
    print(f"{APP_NAME} is running at {url}", flush=True)
    # Cmd+Q ends the process from inside AppKit, so the cleanup runs at exit rather than after start().
    atexit.register(app.shutdown)
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    try:
        import webview
    except ImportError:
        print("pywebview is not installed; opening the browser instead", flush=True)
        webbrowser.open(url)
        threading.Event().wait()
        return

    webview.settings["ALLOW_DOWNLOADS"] = True
    window = webview.create_window(
        APP_NAME, url, width=1440, height=960, min_size=(960, 640), text_select=True,
        background_color="#0b0f19" if _wants_dark(app.theme_pref()) else "#ffffff",
        localization={"global.quitConfirmation": QUIT_WHILE_BUSY},
    )

    def on_closing() -> None:
        # Only ask when quitting would interrupt something.
        window.confirm_close = app.is_busy()

    window.events.closing += on_closing
    _show_app_name_in_menu_bar()
    webview.start(icon=str(ICON), private_mode=False)
    app.shutdown()
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
