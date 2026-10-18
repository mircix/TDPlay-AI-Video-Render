"""TDPlay AI Video Render as a desktop app: the interface in its own window, no terminal or browser tab.

On a Mac the app in /Applications (see ``tdplay.macos_bundle``) runs this; on Windows the Start menu
and desktop shortcuts made by install.bat do (``pythonw -m tdplay.desktop_app``). It starts the
local server, shows it in a native window (pywebview: WebKit on macOS, WebView2 on Windows) and
keeps the computer from sleeping while it is open. Renders run in their own process
(``tdplay.worker``), so they keep going after the window closes; opening the app again shows them.
Only a model download stops when you quit, so quitting asks first while one is running.
"""

from __future__ import annotations

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
ASSETS = Path(__file__).resolve().parent / "assets"
ICON = ASSETS / ("AppIcon.ico" if os.name == "nt" else "AppIcon.png")
QUIT_WHILE_BUSY = ("A model download is still running. Quit anyway?\n\nIt continues where it stopped when you "
                   "press Download again.")


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
    if os.name == "nt":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
        except OSError:
            return False
    try:
        out = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True)
        return out.stdout.strip() == "Dark"
    except OSError:
        return False


def _keep_awake() -> None:
    """Stop the computer from idle-sleeping in the middle of a long render, while the app is open."""
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    elif shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _log_to_file_without_console() -> None:
    """pythonw (the Windows shortcut) has no console; send output to ~/TDPlay/logs/app.log instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from tdplay import models

    logs = models.HOME / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = open(logs / "app.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - lives as long as the app
    log.write(f"--- {time.ctime()} ---\n")
    sys.stdout = sys.stderr = log


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
    _log_to_file_without_console()
    if os.name == "nt":  # its own taskbar group and icon, not Python's
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TDPlay.AIVideoRender")
    from tdplay import app

    port = int(os.environ.get("TDPLAY_PORT") or _free_port())
    url = f"http://127.0.0.1:{port}/"
    app.launch(port=port, open_browser=False, block=False)
    _wait_for(url)
    print(f"{APP_NAME} is running at {url}", flush=True)
    _keep_awake()

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
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
