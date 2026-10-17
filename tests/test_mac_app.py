"""The Mac app's window logic, with pywebview replaced by tests/stubs/webview."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_window_opens_on_the_running_server(tmp_path):
    record = tmp_path / "record.json"
    env = dict(os.environ, TDPLAY_HOME=str(tmp_path / "home"), STUB_WEBVIEW_RECORD=str(record),
               PYTHONPATH=os.pathsep.join([str(ROOT / "tests" / "stubs"), str(ROOT)]))
    result = subprocess.run([sys.executable, "-m", "tdplay.desktop_app"], env=env, cwd=ROOT, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(record.read_text())
    assert data["title"] == "TDPlay AI Video Render"
    assert data["url"].startswith("http://127.0.0.1:") and data["status"] == 200
    assert data["icon_exists"] and data["downloads"] is True
    assert data["confirm_close"] is False  # nothing running: quit without asking
    assert "still being made" in data["quit_message"]["global.quitConfirmation"]


def test_quitting_asks_only_while_busy():
    from tdplay import app

    assert not app.is_busy()
    app._state["runner"] = object()
    try:
        assert app.is_busy()
    finally:
        app._state["runner"] = None
    app._state["downloading"] = True
    try:
        assert app.is_busy()
    finally:
        app._state["downloading"] = False


def test_appearance_choice_is_saved(tmp_path, monkeypatch):
    from tdplay import app, models

    monkeypatch.setattr(models, "HOME", tmp_path)
    assert app.theme_pref() == "system"
    app.save_pref("theme", "dark")
    assert app.theme_pref() == "dark"
    app.save_pref("theme", "purple")  # anything unknown falls back to the system setting
    assert app.theme_pref() == "system"
