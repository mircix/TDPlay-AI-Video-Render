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


def test_projects_can_be_saved_in_another_folder(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tdplay import app, engine, models
    from test_engine import make_settings

    monkeypatch.setattr(models, "HOME", tmp_path / "home")
    monkeypatch.setattr(app, "PROJECTS_DIR", tmp_path / "home" / "projects")
    demo = SimpleNamespace(allowed_paths=[str(tmp_path / "home")])
    monkeypatch.setitem(app._state, "demo", demo)
    assert app.projects_dir() == app.PROJECTS_DIR
    old = engine.create_project(app.projects_dir(), make_settings(), [engine.Segment("made in the default", 2)])

    assert app.use_projects_dir("Videos")[1].startswith("⚠️ Give the folder's full path")
    blocker = tmp_path / "a file"
    blocker.write_text("x")
    assert app.use_projects_dir(str(blocker / "sub"))[1].startswith("⚠️ Can't save projects in")
    assert app.projects_dir() == app.PROJECTS_DIR  # nothing changed

    chosen = tmp_path / "D drive" / "TDPlay videos"
    box, note, _ = app.use_projects_dir(f'"{chosen}"')  # Windows' "Copy as path" adds quotes
    assert box == str(chosen) and chosen.is_dir() and app.projects_dir() == chosen
    assert str(chosen) in demo.allowed_paths  # the window may show the videos saved there
    new = engine.create_project(app.projects_dir(), make_settings(), [engine.Segment("made in the chosen", 2)])
    assert {value for _, value in app.project_choices()} == {str(old.dir), str(new.dir)}

    other = tmp_path / "E"
    app.use_projects_dir(str(other))
    assert app.projects_dir() == other and chosen in app.project_folders()  # still listed
    box, _, _ = app.use_projects_dir("")  # an empty box goes back to the default
    assert box == str(app.PROJECTS_DIR) and app.projects_dir() == app.PROJECTS_DIR
    assert set(app.project_folders()) == {app.PROJECTS_DIR, chosen, other}


def test_choose_opens_the_app_windows_folder_picker(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tdplay import app, models

    monkeypatch.setattr(models, "HOME", tmp_path / "home")
    monkeypatch.setitem(app._state, "demo", None)
    picked, calls = [str(tmp_path / "picked")], []

    class Window:
        def create_file_dialog(self, dialog_type, directory=""):
            calls.append((dialog_type, directory))
            return tuple(picked) or None

    webview = SimpleNamespace(windows=[Window()], FileDialog=SimpleNamespace(FOLDER=20))
    monkeypatch.setitem(sys.modules, "webview", webview)
    box, _, _ = app.pick_projects_dir(str(tmp_path))
    assert box == picked[0] and calls == [(20, str(tmp_path))] and app.projects_dir() == tmp_path / "picked"
    picked.clear()  # cancelled: nothing changes
    assert app.pick_projects_dir(str(tmp_path))[1] == "" and app.projects_dir() == tmp_path / "picked"
    webview.windows.clear()  # in a browser there is no window of the app's own
    assert "Type or paste" in app.pick_projects_dir("")[1]


def test_lora_add_ons_in_the_window(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from tdplay import app, loras, models
    from test_loras import lora_file

    monkeypatch.setattr(models, "HOME", tmp_path / "home")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "home" / "models")
    n = loras.MAX_LORAS
    empty = [""] * n + [1.0] * n
    assert [label for label, _ in app.lora_choices()] == ["— none —"]

    # Add a file from where it is (here: a ComfyUI folder), with the app window's file picker.
    dual = lora_file(tmp_path / "ComfyUI" / "loras" / "ltx2" / "LTX2.3-IC-LORA-Dual-Character.safetensors")
    picked = [str(dual)]
    window = SimpleNamespace(create_file_dialog=lambda kind, file_types=(): tuple(picked) or None)
    monkeypatch.setitem(sys.modules, "webview", SimpleNamespace(windows=[window], FileDialog=SimpleNamespace(OPEN=10)))
    *slots, note = app.add_lora_file(*empty)
    assert slots[0]["value"] == str(dual) and slots[n] == 1.0 and "Added **LTX2.3-IC-LORA-Dual-Character" in note
    assert str(dual) in [path for _, path in slots[0]["choices"]]
    picked[0] = str(lora_file(tmp_path / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"))
    assert "isn't needed" in app.add_lora_file(*empty)[-1]  # refused, with the reason

    # Download one of the app's add-ons: it lands in the next free slot at its suggested strength.
    omni = loras.ADDONS[0]

    def download(repo, patterns):
        assert (repo, patterns) == (omni.repo, (omni.file,))
        yield models.DownloadProgress(repo, 100, 617_000_000)
        lora_file(omni.path)
        yield models.DownloadProgress(repo, 617_000_000, 617_000_000, finished=True)

    monkeypatch.setattr(models, "download", download)
    current = [str(dual), "", "", "", 1.0, 1.0, 1.0, 1.0]
    *slots, note = list(app.addon_downloader(0)(*current))[-1]
    assert [s["value"] for s in slots[:n]] == [str(dual), str(omni.path), "", ""] and slots[n + 1] == 0.9
    assert "OmniNFT RL** is ready" in note and not app._state.get("downloading")

    # The slots are remembered for next time.
    app.remember_loras(str(dual), str(omni.path), "", "", 0.6, 0.9, 1.0, 1.0)
    restored = app.saved_loras()
    assert [s["value"] for s in restored[:n]] == [str(dual), str(omni.path), "", ""] and restored[n:n + 2] == [0.6, 0.9]
    assert app.picked_lora(str(omni.path))[0] == 0.9 and "rank 32" in app.picked_lora(str(dual))[1]


def test_a_render_checks_its_loras_first(tmp_path, monkeypatch):
    from tdplay import app, engine, loras, models
    from test_loras import lora_file

    monkeypatch.setattr(models, "HOME", tmp_path / "home")
    monkeypatch.setattr(app, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(app, "ON_MAC", False)
    monkeypatch.setenv("TDPLAY_MODEL_DIR", str(tmp_path / "models"))
    created = []
    monkeypatch.setattr(engine, "create_project", lambda root, settings, *a, **k: created.append(settings) or 1 / 0)
    good = lora_file(tmp_path / "good.safetensors")
    prompts = ["a shot"] + [""] * (app.MAX_SEGMENTS - 1)
    seconds = [4] * app.MAX_SEGMENTS
    settings = ["ltx-2.3-fp8", "576p", "16:9 landscape", True, True, False, 0.0, 8, True, -1, "off", "fast", 1.0]

    def generate(*lora_values):
        try:
            return list(app.generate(None, None, "", 1, *prompts, *seconds, *settings, *lora_values))
        except ZeroDivisionError:  # create_project was reached
            return None

    assert generate(str(good), "", "", "", 0.7, 1.0, 1.0, 1.0) is None
    assert created[-1].loras == [{"path": str(good), "strength": 0.7}]
    assert generate(str(good), "", "", "", 0.0, 1.0, 1.0, 1.0) is None and created[-1].loras == []  # 0 = off
    missing = generate(str(tmp_path / "gone.safetensors"), "", "", "", 1.0, 1.0, 1.0, 1.0)
    assert "there any more" in missing[-1][0] and len(created) == 2
    assert loras.MAX_LORAS == 4


def test_character_creator_in_the_window(tmp_path, monkeypatch):
    import shlex
    from types import SimpleNamespace

    from PIL import Image

    from tdplay import app, characters, creator, models

    monkeypatch.setattr(models, "HOME", tmp_path / "home")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "home" / "models")
    monkeypatch.setenv("TDPLAY_CREATOR_CMD", shlex.join([sys.executable, str(ROOT / "tests" / "fake_creator.py")]))
    monkeypatch.setattr(app, "active_job", lambda: None)
    assert "Download the picture model" in list(app.make_from_description("a woman", True, "1:1 square"))[-1][-1]
    assert "FLUX.2 [klein] 4B" in app.models_table() and "not downloaded" in app.models_table()

    # The download fetches only the diffusers layout, which mflux reads too.
    def download(repo, patterns):
        assert repo == creator.REPO and "transformer/*" in patterns and "*.safetensors" not in patterns
        yield models.DownloadProgress(repo, 8_000_000_000, 16_000_000_000)
        monkeypatch.setattr(creator, "ready", lambda: True)
        yield models.DownloadProgress(repo, 16_000_000_000, 16_000_000_000, finished=True)

    monkeypatch.setattr(models, "download", download)
    monkeypatch.setattr(app.shutil, "disk_usage", lambda path: SimpleNamespace(free=18e9))
    assert "Not enough free disk space" in list(app.download_creator())[-1][0]
    monkeypatch.setattr(app.shutil, "disk_usage", lambda path: SimpleNamespace(free=500e9))
    note, row = list(app.download_creator())[-1]
    assert note == "The picture model is ready." and row["visible"] is False and not app._state.get("downloading")

    # Four pictures from a description, shown as they are made; a click makes one the character's photo.
    steps = list(app.make_from_description("a woman with long black hair", True, "Full body (2:3)"))
    gallery, pictures, wanted, note = steps[-1]
    assert len(pictures) == creator.COUNT and gallery == pictures and "Click the picture you like" in note
    assert [len(s[1]) for s in steps if isinstance(s[1], list)][:2] == [1, 2]  # one at a time
    with Image.open(pictures[0]) as picture:
        assert picture.size == creator.SHAPES["Full body (2:3)"]
    photo, notes, note = app.use_picture(SimpleNamespace(index=2), pictures, wanted, "")
    assert photo == pictures[2] and notes == "a woman with long black hair" and "Press **Save**" in note
    assert app.use_picture(SimpleNamespace(index=1), pictures, wanted, "red coat")[1] == app.gr.skip()
    _, photo_out, folder, note = app.save_character("Mia", photo, "black lace slip dress", None)
    assert "Saved **Mia**" in note and characters.load(folder).notes == "black lace slip dress"

    # The same person in other clothes, saved as a second character.
    steps = list(app.change_outfit(photo_out, "a red summer dress"))
    _, outfits, wanted, note = steps[-1]
    assert len(outfits) == creator.COUNT and wanted == {"mode": "outfit", "text": "a red summer dress"}
    photo, notes, note = app.use_picture(SimpleNamespace(index=0), outfits, wanted, "black lace slip dress")
    assert notes == app.gr.skip() and "Update **Clothes, hair, look**" in note
    app.save_new_character("Mia (summer)", photo, "a red summer dress")
    assert [c.name for c in characters.everyone()] == ["Mia", "Mia (summer)"]
    with Image.open(characters.everyone()[0].photo) as before:
        assert before.size == creator.SHAPES["Full body (2:3)"]  # the first Mia keeps her photo

    # Problems are said, not raised; and pictures and renders take turns on the graphics card.
    assert "Add a photo first" in list(app.change_outfit(None, "a dress"))[-1][-1]
    assert "Describe the character first" in list(app.make_from_description(" ", True, "1:1 square"))[-1][-1]
    monkeypatch.setattr(app, "active_job", lambda: "/projects/shot")
    assert "Wait until the video" in list(app.make_from_description("a man", True, "1:1 square"))[-1][-1]
    monkeypatch.setattr(app, "active_job", lambda: None)
    monkeypatch.setattr(creator, "running", lambda: True)
    assert "Character pictures are being made" in next(app._start(None, ["render"]))[0]
    assert app.stop_pictures() == "Nothing is being made."
