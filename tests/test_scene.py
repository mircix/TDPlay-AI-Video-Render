"""The scene builder (``tdplay.scene``): characters cut out of their photos and placed on a location."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tdplay import scene

RED, BLUE, SKY = (200, 40, 40), (40, 60, 200), (90, 160, 220)


def photo(tmp_path: Path, name: str, color, size=(300, 600)) -> str:
    """A 'character photo': a colored figure on a light background."""
    image = Image.new("RGB", size, (230, 230, 230))
    w, h = size
    ImageDraw.Draw(image).rectangle((w // 3, h // 6, 2 * w // 3, h - h // 12), fill=color)
    path = tmp_path / name
    image.save(path)
    return str(path)


def fake_cutter(image: Image.Image) -> np.ndarray:
    """Stands in for ISNet: the figure is whatever isn't the light background."""
    return (np.asarray(image, np.float32).mean(axis=2) < 200).astype(np.float32)


def test_cut_out_crops_to_the_subject(tmp_path):
    with Image.open(photo(tmp_path, "a.png", RED)) as picture:
        cut = scene.cut_out(picture, fake_cutter(picture))
    assert cut.mode == "RGBA" and cut.size == (101, 451)
    rgba = np.asarray(cut)
    assert rgba[..., 3].min() == 255 and tuple(rgba[200, 50, :3]) == RED
    with pytest.raises(ValueError, match="No person"):
        scene.cut_out(picture, np.zeros((600, 300), np.float32))


def test_characters_stand_where_they_were_placed(tmp_path):
    background = Image.new("RGB", (1920, 1080), SKY)
    red = Image.new("RGBA", (100, 400), RED + (255,))
    blue = Image.new("RGBA", (100, 400), BLUE + (255,))
    placements = [scene.Placement(x=0.3, y=0.9, size=0.5), scene.Placement(x=0.35, y=0.95, size=0.6)]
    out = np.asarray(scene.compose(background, [red, blue], placements)).astype(int)
    # Red: 540 px tall, centered at x=576, feet at y=972. Blue stands lower (nearer the camera), so it is
    # drawn over red where they overlap. Both take on a little of the sky's light and stay their own color.
    assert np.abs(out[600, 530] - expected_light(RED, SKY)).max() <= 1
    assert np.abs(out[900, 600] - expected_light(BLUE, SKY)).max() <= 1
    assert np.abs(out[300, 576] - SKY).max() == 0  # above red's head: still sky
    assert np.abs(out[900, 1500] - SKY).max() == 0


def expected_light(color, light, amount=scene.HARMONIZE):
    color, light = np.array(color, float), np.array(light, float)
    level = np.clip((light.mean() / color.mean()) ** amount, 0.8, 1.25)
    cast = np.clip(light / light.mean(), 0.75, 1.25)
    return np.round(np.clip(color * ((1 - amount) + amount * cast) * level, 0, 255))


def test_a_characters_own_colors_survive_the_light(tmp_path):
    # A red shirt over dark trousers under a blue sky: the trousers must not turn teal, the shirt stays red.
    figure = np.zeros((200, 100, 4), np.uint8)
    figure[:100, :, :3], figure[100:, :, :3], figure[..., 3] = (180, 40, 60), (35, 35, 40), 255
    out = np.asarray(scene.harmonize(Image.fromarray(figure, "RGBA"), np.array(SKY, float))).astype(int)
    shirt, trousers = out[50, 50, :3], out[150, 50, :3]
    assert shirt[0] > 2 * shirt[1] and shirt[0] > 2 * shirt[2]
    assert trousers.max() < 60 and trousers[1] - trousers[0] <= 8  # still dark, not pushed toward green


def test_build_a_first_frame(tmp_path):
    location = tmp_path / "place.png"
    Image.new("RGB", (4000, 1500), SKY).save(location)  # wider than the shot: cropped to 16:9
    chars = [photo(tmp_path, "a.png", RED), photo(tmp_path, "b.png", BLUE)]
    placements = [scene.Placement(x=scene.default_x(i, 2)) for i in range(2)]
    path = scene.build(str(location), chars, placements, 1024, 576, tmp_path / "scenes", cutter=fake_cutter)
    with Image.open(path) as built:
        assert built.size == (1920, 1080)
        pixels = np.asarray(built).astype(int)
    assert pixels[700, 640, 0] > 150 and pixels[700, 1280, 2] > 150  # red at 1/3, blue at 2/3 of the width
    assert scene.build(None, chars[:1], placements[:1], 576, 1024, tmp_path, cutter=fake_cutter).exists()
    # A location alone needs no cut-out model; nothing at all is an error.
    assert scene.build(str(location), [], [], 1024, 1024, tmp_path, cutter=None).exists()
    with pytest.raises(ValueError, match="location"):
        scene.build(None, [], [], 1024, 576, tmp_path, cutter=fake_cutter)


def test_the_cut_out_model_is_downloaded_once_and_checked(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    source.write_bytes(b"not really a model" * 1000)
    monkeypatch.setattr(scene.models, "HOME", tmp_path / "home")
    monkeypatch.setattr(scene, "CUTOUT_URL", source.as_uri())
    monkeypatch.setattr(scene, "CUTOUT_BYTES", source.stat().st_size)
    monkeypatch.setattr(scene, "CUTOUT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="didn't download correctly"):
        scene.ensure_cutout_model()
    assert not scene.cutout_model_path().exists() and not scene.cutout_model_path().with_suffix(".part").exists()

    monkeypatch.setattr(scene, "CUTOUT_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    seen = []
    path = scene.ensure_cutout_model(lambda done, total: seen.append((done, total)))
    assert path.read_bytes() == source.read_bytes() and seen[-1] == (source.stat().st_size,) * 2
    source.unlink()  # already there: not downloaded again
    assert scene.ensure_cutout_model() == path


@pytest.mark.skipif(not os.environ.get("TDPLAY_CUTOUT_MODEL"), reason="set TDPLAY_CUTOUT_MODEL to the ISNet .onnx")
def test_the_real_cut_out_model_finds_a_figure(tmp_path):
    pytest.importorskip("onnxruntime")
    with Image.open(photo(tmp_path, "a.png", (40, 60, 140), size=(600, 800))) as picture:
        mask = scene.Cutter(os.environ["TDPLAY_CUTOUT_MODEL"])(picture)
    assert mask.shape == (800, 600)
    assert mask[300:600, 250:350].mean() > 0.9 and mask[:, :150].mean() < 0.1
