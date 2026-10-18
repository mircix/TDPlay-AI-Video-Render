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
    assert "download is still running" in data["quit_message"]["global.quitConfirmation"]


def test_quitting_asks_only_while_downloading():
    from tdplay import app

    assert not app.is_busy()  # renders run in their own process and keep going after the app quits
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


def test_open_folder_shows_the_project_in_the_file_manager(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tdplay import app

    monkeypatch.setattr(app, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setitem(app._state, "last_project", None)
    opened = []
    # Windows: File Explorer through os.startfile ("xdg-open" and "open" don't exist there).
    monkeypatch.setattr(app, "os", SimpleNamespace(name="nt", startfile=opened.append))
    app.open_folder()  # no project yet: the projects folder, made if needed
    assert opened == [str(tmp_path / "projects")] and (tmp_path / "projects").is_dir()
    project = tmp_path / "projects" / "shot"
    project.mkdir()
    monkeypatch.setitem(app._state, "last_project", str(project))
    app.open_folder()
    app.open_folder(str(tmp_path))
    assert opened[1:] == [str(project), str(tmp_path)]
    # macOS: Finder through "open".
    monkeypatch.setattr(app, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(app.subprocess, "Popen", lambda cmd, **kwargs: opened.append(cmd))
    app.open_folder()
    assert opened[-1] == ["open", str(project)]
