"""The character creator: prompts, picture shapes, and making pictures in their own process (stand-in model)."""

import json
import shlex
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tdplay import creator, models

FAKE = [sys.executable, str(Path(__file__).resolve().parent / "fake_creator.py")]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HOME", tmp_path / "home")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "home" / "models")
    monkeypatch.setenv("TDPLAY_CREATOR_CMD", shlex.join(FAKE))
    for name in ("FAKE_CREATOR_FAIL", "FAKE_CREATOR_CRASH", "FAKE_CREATOR_SLEEP"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "home"


def seen(home: Path) -> dict:
    return json.loads((home / "creator" / "seen.json").read_text())


def test_prompts():
    studio = creator.describe_prompt("  a woman with long black hair,\n black lace slip dress. ")
    assert studio.startswith("a woman with long black hair, black lace slip dress. Full-body photo")
    assert "plain light grey studio background" in studio
    assert creator.describe_prompt("a knight in silver armor", studio=False) == "a knight in silver armor."
    outfit = creator.outfit_prompt("a red summer dress and white sneakers.")
    assert outfit.startswith("Change the person's clothes to a red summer dress and white sneakers. Keep the same")
    for empty in ("", "  ", None):
        with pytest.raises(ValueError, match="Describe"):
            creator.describe_prompt(empty)
        with pytest.raises(ValueError, match="Describe"):
            creator.outfit_prompt(empty)


def test_picture_shapes_are_about_a_megapixel():
    for width, height in creator.SHAPES.values():
        assert width % 16 == 0 and height % 16 == 0
        assert 0.9 < width * height / creator.AREA < 1.1
    width, height = creator.fit(3000, 4000)  # a phone photo
    assert width % 16 == 0 and height % 16 == 0 and abs(width / height - 0.75) < 0.02
    assert 0.95 < width * height / creator.AREA < 1.05
    assert creator.fit(100, 100) == (1024, 1024)  # small photos are made bigger


def test_an_outfit_change_keeps_the_photos_shape(tmp_path):
    photo = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, shown upright: 3000 x 4000
    Image.new("RGB", (4000, 3000), "white").save(photo, exif=exif)
    request = creator.outfit_job(str(photo), "a red dress", seed=7, count=2)
    assert request["mode"] == "outfit" and request["photo"] == str(photo) and request["seeds"] == [7, 8]
    assert (request["width"], request["height"]) == creator.fit(3000, 4000)
    with pytest.raises(ValueError, match="Add a photo"):
        creator.outfit_job(None, "a red dress")
    with pytest.raises(ValueError, match="Describe"):
        creator.outfit_job(str(photo), " ")


def test_pictures_are_made_in_their_own_process(home):
    request = creator.describe_job("a woman with long black hair", True, "16:9 landscape", seed=10, count=3)
    events = list(creator.make(request))
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "status" and kinds.count("picture") == 3 and kinds[-1] == "done"
    assert events[-1]["made"] == 3
    pictures = [e["path"] for e in events if e["kind"] == "picture"]
    colors = set()
    for path, seed in zip(pictures, (10, 11, 12)):
        assert Path(path).parent == home / "creator" and f"_describe_{seed}.png" in path
        with Image.open(path) as picture:
            assert picture.size == (1344, 768)
            colors.add(picture.getpixel((5, 5)))
    assert len(colors) == 3  # one picture per seed
    assert seen(home)["prompt"] == creator.describe_prompt("a woman with long black hair")
    assert not list((home / "creator").glob("job_*.json"))  # the request file is cleaned up
    assert "Loading the stand-in model" in (home / "creator" / "log.txt").read_text()  # its output is logged
    assert not creator.running()


def test_an_outfit_is_changed_on_the_upright_photo(home, tmp_path):
    photo = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    sideways = np.zeros((3000, 4000, 3), np.uint8)
    sideways[:, :2000] = (255, 0, 0)  # the left half; shown upright, that is the top half
    Image.fromarray(sideways).save(photo, exif=exif)
    events = list(creator.make(creator.outfit_job(str(photo), "a green coat", seed=3, count=1)))
    assert events[-1]["kind"] == "done", events
    width, height = creator.fit(3000, 4000)
    with Image.open(events[-2]["path"]) as picture:
        assert picture.size == (width, height)
        top, bottom = picture.getpixel((width // 4, height // 16)), picture.getpixel((width // 4, height * 7 // 16))
    assert top[0] > 200 and bottom[0] < 50  # the model saw the photo upright
    assert not list((home / "creator").glob("reference_*"))  # its upright copy is cleaned up
    assert "Change the person's clothes to a green coat" in seen(home)["prompt"]


def test_running_out_of_graphics_memory_is_explained(home, monkeypatch):
    monkeypatch.setenv("FAKE_CREATOR_FAIL", "2")
    events = list(creator.make(creator.describe_job("a man in a grey suit", False, "1:1 square", count=4)))
    assert [e["kind"] for e in events if e["kind"] != "status"] == ["picture", "error"]
    assert "ran out of memory" in events[-1]["text"] and "HIP out of memory" in events[-1]["text"]
    assert events[-1]["made"] == 1


def test_a_crash_is_reported_with_the_end_of_its_log(home, monkeypatch):
    monkeypatch.setenv("FAKE_CREATOR_CRASH", "1")
    events = list(creator.make(creator.describe_job("a man in a grey suit", False, "1:1 square", count=2)))
    assert events[-1]["kind"] == "error"
    assert "stopped unexpectedly (exit code 3)" in events[-1]["text"]
    assert "Loading the stand-in model" in events[-1]["text"]


def test_stop_ends_the_picture_process(home, monkeypatch):
    monkeypatch.setenv("FAKE_CREATOR_SLEEP", "60")
    events = []
    thread = threading.Thread(target=lambda: events.extend(creator.make(
        creator.describe_job("a woman", True, "1:1 square", count=2))))
    started = time.time()
    thread.start()
    while not any(e["kind"] == "status" and "Making picture 1" in e["text"] for e in events):
        assert time.time() - started < 30
        time.sleep(0.05)
    assert creator.running()
    assert creator.stop()
    thread.join(10)
    assert not thread.is_alive() and events[-1] == {"kind": "stopped"}
    assert not creator.running() and not creator.stop()
    assert time.time() - started < 30


def test_leaving_the_events_ends_the_process(home, monkeypatch):
    monkeypatch.setenv("FAKE_CREATOR_SLEEP", "60")
    events = creator.make(creator.describe_job("a woman", True, "1:1 square", count=1))
    assert next(events)["kind"] == "status"
    assert creator.running()
    events.close()  # the window went away
    assert not creator.running()


def test_the_model_is_ready_once_all_of_it_is_downloaded(home):
    assert not creator.ready()
    root = creator.model_dir()
    files = {"model_index.json": b"{}", "transformer/diffusion_pytorch_model.safetensors": b"x" * 10}
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    manifest = models.MODELS_DIR / "black-forest-labs--FLUX.2-klein-4B.manifest"
    manifest.write_text("abc\n" + "".join(f"{n}\t{len(d)}\n" for n, d in files.items()))
    assert creator.ready()
    (root / "model_index.json").unlink()  # half deleted
    assert not creator.ready()
