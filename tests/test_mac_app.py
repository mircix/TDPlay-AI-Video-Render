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


def test_graphics_memory_choice_reaches_the_render(tmp_path, monkeypatch):
    from tdplay import app, engine, models, worker
    from tdplay.engine_torch import TorchRunner

    monkeypatch.setattr(models, "HOME", tmp_path)
    monkeypatch.setattr(app, "ON_MAC", False)
    monkeypatch.setitem(app._state, "job", None)
    monkeypatch.setattr(worker, "start", lambda project_dir, job: None)
    monkeypatch.setattr(app, "_watch", lambda *args: iter([("watching", None, None)]))
    settings = engine.Settings(width=512, height=256, model_dir="", gemma_dir="", backend="torch")
    project = engine.create_project(tmp_path / "projects", settings, [engine.Segment("a", 2)])
    assert app.gpu_memory_pref() == "balanced"
    app.save_pref("gpu_memory", "max")
    assert app.gpu_memory_pref() == "max"
    list(app._start(project, ["render"]))  # Create or Resume: the setting in effect is saved with the project
    project = engine.Project.load(project.dir)
    assert project.settings.gpu_memory == "max" and TorchRunner(project).job()["gpu_memory"] == "max"
    app.save_pref("gpu_memory", "everything")
    assert app.gpu_memory_pref() == "balanced"


def test_scene_builder_makes_the_first_frame(tmp_path, monkeypatch):
    from PIL import Image

    from tdplay import app, models
    from test_scene import BLUE, RED, fake_cutter, photo

    monkeypatch.setattr(models, "HOME", tmp_path)
    monkeypatch.setattr(app, "_cutter", [fake_cutter])
    monkeypatch.setattr(app, "ON_MAC", False)
    first, second = photo(tmp_path, "a.png", RED), photo(tmp_path, "b.png", BLUE)
    # Adding characters spreads them across the frame.
    updates = app.spread_characters(None, first, second)
    assert [u.get("value") for u in updates] == [None, 33, 67]

    picture, strength, characters, note = list(app.build_scene(
        None, None, first, second, 25, 33, 67, 95, 95, 95, 70, 70, 70,
        "not used: slot 1 has no picture", "Mia: lace dress", " Leo:  grey\n suit ", "576p", "16:9 landscape"))[-1]
    assert Path(picture).parent == tmp_path / "scenes" and "first frame above" in note
    assert characters == "Mia: lace dress\nLeo: grey suit"  # one line each, into the Characters box
    assert strength == 0.85  # so the model may repaint the pasted-in light a little
    with Image.open(picture) as built:
        assert built.size == (1920, 1080)
    *_, (_, _, _, note) = app.build_scene(None, None, None, None, 25, 50, 75, 95, 95, 95, 70, 70, 70, "", "", "",
                                          "576p", "16:9 landscape")
    assert note.startswith("⚠️ Add a location picture")
