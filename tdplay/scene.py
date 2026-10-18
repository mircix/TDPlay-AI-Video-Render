"""The scene builder: a shot's first frame made from a location picture and up to three character pictures.

LTX-2 takes a picture only as a frame of the video, not as "this is what a character looks like" or
"this is the place". So the scene builder makes that frame: it cuts each character out of their photo
(ISNet, a small background-removal model, run with ONNX Runtime on the CPU), places them on the location,
nudges their colors toward the location's light, and saves the picture. It then starts the shot like any
first-frame picture. A first-frame strength a little under 1 lets the model repaint it slightly, which
blends the pasted-in light further.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from tdplay import models

CUTOUT_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
CUTOUT_SHA256 = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
CUTOUT_BYTES = 178_648_008
MAX_CHARACTERS = 3
#: Longest side of a built first frame (the render scales it to the shot's size).
LONG_SIDE = 1920
#: First-frame strength suggested for a built scene: the model may repaint the pasted-in light a little.
SCENE_STRENGTH = 0.85
#: How far a character takes on the location's light: its color cast and brightness (0 = not at all, 1 = fully).
HARMONIZE = 0.3
#: Mask values (0-1) under this are background, over ``1 - CLEAN`` fully the character.
CLEAN = 0.05


@dataclass
class Placement:
    """Where a character stands in the frame."""

    #: Horizontal center: 0 = left edge, 1 = right edge.
    x: float = 0.5
    #: Where the bottom of the cut-out (the feet, for a full-body photo) is: 0 = top, 1 = bottom.
    y: float = 0.95
    #: Height as a share of the frame's height.
    size: float = 0.7


def default_x(index: int, count: int) -> float:
    """Characters spread evenly across the frame."""
    return (index + 1) / (count + 1)


# -- the cut-out model -------------------------------------------------------------------------------


def cutout_model_path() -> Path:
    return models.HOME / "models" / "cutout" / "isnet-general-use.onnx"


def ensure_cutout_model(on_progress: Callable[[int, int], None] | None = None) -> Path:
    """The cut-out model, downloaded once (about 180 MB) and checked against its known checksum."""
    path = cutout_model_path()
    if path.exists() and path.stat().st_size == CUTOUT_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    digest, done = hashlib.sha256(), 0
    with urllib.request.urlopen(CUTOUT_URL, timeout=60) as response, open(part, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
            done += len(chunk)
            if on_progress:
                on_progress(done, CUTOUT_BYTES)
    if digest.hexdigest() != CUTOUT_SHA256:
        part.unlink(missing_ok=True)
        raise RuntimeError("The cut-out model didn't download correctly. Check the internet connection and try again.")
    part.replace(path)
    return path


class Cutter:
    """Finds the main subject of a photo: ISNet (DIS), 1024x1024 input, on the CPU."""

    SIZE = 1024

    def __init__(self, model_path: str | Path | None = None):
        import onnxruntime as ort

        path = model_path or ensure_cutout_model()
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def __call__(self, image: Image.Image) -> np.ndarray:
        """The subject's mask, 0-1, at the photo's size."""
        rgb = image.convert("RGB").resize((self.SIZE, self.SIZE), Image.Resampling.BILINEAR)
        x = (np.asarray(rgb, np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
        out = self._session.run(None, {self._input: x})[0][0, 0]
        lo, hi = float(out.min()), float(out.max())
        mask = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
        small = Image.fromarray(np.round(mask * 255).astype(np.uint8))
        return np.asarray(small.resize(image.size, Image.Resampling.BILINEAR), np.float32) / 255.0


def cut_out(photo: Image.Image, mask: np.ndarray) -> Image.Image:
    """The photo's subject as an RGBA picture, cropped to it."""
    alpha = np.clip((mask - CLEAN) / (1 - 2 * CLEAN), 0, 1)
    ys, xs = np.nonzero(alpha > 0.1)
    if len(ys) == 0:
        raise ValueError("No person or object was found in a character picture. Try a photo with a clear subject.")
    rgba = np.dstack([np.asarray(photo.convert("RGB")), np.round(alpha * 255).astype(np.uint8)])
    return Image.fromarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1], "RGBA")


# -- building the picture ------------------------------------------------------------------------------


def frame_size(width: int, height: int) -> tuple[int, int]:
    """The built picture's size: the shot's shape, ``LONG_SIDE`` on its long side."""
    scale = LONG_SIDE / max(width, height)
    return round(width * scale), round(height * scale)


def backdrop(location: Image.Image | None, size: tuple[int, int]) -> Image.Image:
    """The location cropped to the shot's shape, or a plain studio backdrop."""
    if location is not None:
        return ImageOps.fit(location.convert("RGB"), size, Image.Resampling.LANCZOS)
    w, h = size
    shade = np.linspace(150, 95, h, dtype=np.float32)[:, None, None]
    return Image.fromarray(np.broadcast_to(shade, (h, w, 3)).astype(np.uint8))


def harmonize(cutout: Image.Image, light: np.ndarray, amount: float = HARMONIZE) -> Image.Image:
    """Give the cut-out part of the location's light: ``light`` is the location's average color.

    Its color cast (bluish under a sky, warm indoors) tints the character, and the character's
    brightness moves toward the location's. Both are gains, so the character's own colors (a red
    shirt, dark trousers) keep their look instead of being pushed toward the opposite color.
    """
    rgba = np.asarray(cutout, np.float32).copy()
    alpha = rgba[..., 3] / 255.0
    if alpha.sum() == 0:
        return cutout
    cast = np.clip(light / max(float(light.mean()), 1.0), 0.75, 1.25)  # a very colorful place only tints a little
    brightness = (rgba[..., :3].mean(axis=2) * alpha).sum() / alpha.sum()
    level = np.clip((float(light.mean()) / max(brightness, 1.0)) ** amount, 0.8, 1.25)
    rgba[..., :3] = np.clip(rgba[..., :3] * ((1 - amount) + amount * cast) * level, 0, 255)
    return Image.fromarray(np.round(rgba).astype(np.uint8), "RGBA")


def compose(background: Image.Image, cutouts: list[Image.Image], placements: list[Placement]) -> Image.Image:
    """Place the cut-outs on the background, farther back (higher up) ones first."""
    canvas = background.copy()
    w, h = canvas.size
    light = np.asarray(background, np.float32).reshape(-1, 3).mean(axis=0)
    for i in sorted(range(len(cutouts)), key=lambda i: placements[i].y):
        cutout, place = cutouts[i], placements[i]
        height = max(1, round(place.size * h))
        width = max(1, round(cutout.width * height / cutout.height))
        sized = harmonize(cutout.resize((width, height), Image.Resampling.LANCZOS), light)
        canvas.paste(sized, (round(place.x * w - width / 2), round(place.y * h - height)), sized)
    return canvas


def build(location: str | None, characters: list[str], placements: list[Placement], width: int, height: int,
          out_dir: str | Path, cutter: Callable[[Image.Image], np.ndarray] | None = None) -> Path:
    """Build the first frame for a ``width`` x ``height`` shot; returns the saved picture's path."""
    if not characters and not location:
        raise ValueError("Add a location picture, character pictures, or both.")
    if len(characters) > MAX_CHARACTERS:
        raise ValueError(f"Up to {MAX_CHARACTERS} characters.")
    if characters and cutter is None:
        cutter = Cutter()
    cutouts = []
    for path in characters:
        with Image.open(path) as photo:
            photo = ImageOps.exif_transpose(photo).convert("RGB")
            cutouts.append(cut_out(photo, cutter(photo)))
    background = None
    if location:
        with Image.open(location) as picture:
            background = backdrop(ImageOps.exif_transpose(picture), frame_size(width, height))
    else:
        background = backdrop(None, frame_size(width, height))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"scene_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000:03d}.png"
    compose(background, cutouts, placements).save(path)
    return path
