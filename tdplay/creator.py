"""The character creator: pictures of a character made from a description, and a photo with other clothes on.

Both use FLUX.2 [klein] 4B (Black Forest Labs, Apache 2.0), one model that makes a picture from text and
changes a picture from text, in 4 steps. It runs in its own process, ``python -m tdplay.creator JOB.json``,
so the graphics card's memory is free again as soon as the pictures are made: on a PC with diffusers on
PyTorch (AMD ROCm or NVIDIA CUDA), on a Mac with mflux on MLX. Both load the same download (about 16 GB).
The process prints one ``TDPLAY-CREATOR {json}`` line per event: what it is doing, and each saved picture.
"""

from __future__ import annotations

import atexit
import json
import os
import random
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from PIL import Image, ImageOps

from tdplay import engine, media, models

REPO = "black-forest-labs/FLUX.2-klein-4B"
#: The diffusers layout, which mflux reads too (the repo also has a single-file copy of the transformer).
FILES = ("model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*")
DOWNLOAD_GB = 16
#: The distilled model's steps (it runs without guidance).
STEPS = 4
#: Pictures one press makes, to choose from.
COUNT = 4
#: Picture shapes, about a megapixel each (what the model does best, and quick); multiples of 16.
SHAPES = {
    "Full body (2:3)": (832, 1248),
    "16:9 landscape": (1344, 768),
    "9:16 portrait": (768, 1344),
    "1:1 square": (1024, 1024),
}
AREA = 1024 * 1024
EVENT = "TDPLAY-CREATOR "
#: What a studio picture adds to the description: one whole person on a plain background, easy to cut out.
STUDIO = ("Full-body photo of one person standing, the whole body from head to feet in the frame, facing the "
          "camera, plain light grey studio background, soft even light, sharp focus, photorealistic")


def model_dir() -> Path:
    return models.local_dir(REPO)


def ready() -> bool:
    return models.is_downloaded(REPO)


def download() -> Iterator[models.DownloadProgress]:
    return models.download(REPO, FILES)


def pictures_dir() -> Path:
    return models.HOME / "creator"


def describe_prompt(description: str, studio: bool = True) -> str:
    text = " ".join(str(description or "").split()).rstrip(".")
    if not text:
        raise ValueError("Describe the character first: who they are, their clothes and hair.")
    return f"{text}. {STUDIO}." if studio else f"{text}."


def outfit_prompt(outfit: str) -> str:
    text = " ".join(str(outfit or "").split()).rstrip(".")
    if not text:
        raise ValueError("Describe the new clothes first.")
    return (f"Change the person's clothes to {text}. Keep the same person: the same face, hair, body, pose, "
            "background and light.")


def fit(width: int, height: int, area: int = AREA) -> tuple[int, int]:
    """A picture's shape at about ``area`` pixels, both sides multiples of 16."""
    scale = (area / max(1, width * height)) ** 0.5
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def job(mode: str, prompt: str, size: tuple[int, int], photo: str | None = None, count: int = COUNT,
        seed: int | None = None) -> dict:
    """What one press asks for: ``mode`` is "describe" (a new picture) or "outfit" (``photo`` changed)."""
    first = random.randrange(2**31) if seed is None else int(seed)
    return {"mode": mode, "prompt": prompt, "width": size[0], "height": size[1], "photo": photo,
            "seeds": [first + i for i in range(count)], "out_dir": str(pictures_dir())}


def describe_job(description: str, studio: bool, shape: str, **kwargs) -> dict:
    return job("describe", describe_prompt(description, studio), SHAPES.get(shape, SHAPES["Full body (2:3)"]),
               **kwargs)


def outfit_job(photo: str | None, outfit: str, **kwargs) -> dict:
    """The photo with other clothes, in the photo's shape."""
    if not photo:
        raise ValueError("Add a photo first, or click a character in the library.")
    prompt = outfit_prompt(outfit)
    with Image.open(photo) as picture:
        size = ImageOps.exif_transpose(picture).size
    return job("outfit", prompt, fit(*size), photo=str(photo), **kwargs)


# -- the app's side ----------------------------------------------------------------------------------

#: The picture process while one runs (the app runs one at a time).
_running: list[subprocess.Popen] = []


def command() -> list[str]:
    override = os.environ.get("TDPLAY_CREATOR_CMD")
    return shlex.split(override) if override else [sys.executable, "-m", "tdplay.creator"]


def running() -> bool:
    return any(proc.poll() is None for proc in _running)


def stop() -> bool:
    """Stop the picture process; whether one was running."""
    stopped = False
    for proc in list(_running):
        if proc.poll() is None:
            proc.stopped = True
            proc.kill()
            stopped = True
    return stopped


atexit.register(stop)  # quitting the app ends a picture process it started


def make(request: dict) -> Iterator[dict]:
    """Run ``request`` in its own process, yielding its events; closing the iterator stops the process.

    Events are ``{"kind": "status", "text": ...}``, ``{"kind": "picture", "path": ...}`` and, last,
    ``{"kind": "done"}``, ``{"kind": "stopped"}`` (after :func:`stop`) or ``{"kind": "error", "text": ...}``.
    """
    out = Path(request["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"job_{time.time_ns()}.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(engine.APP_ROOT), os.environ.get("PYTHONPATH")])))
    # As for renders: AMD's MIOpen (the VAE's convolutions) logs only errors and doesn't benchmark every method.
    env = {"MIOPEN_LOG_LEVEL": "3", "MIOPEN_FIND_MODE": "FAST", **env}
    proc = subprocess.Popen([*command(), str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            stdin=subprocess.DEVNULL, **media.NO_WINDOW)
    _running.append(proc)
    tail: list[str] = []
    final = None
    try:
        with open(out / "log.txt", "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n--- {time.ctime()}: {request['mode']} ---\n")
            for raw in proc.stdout:
                line = raw.decode(errors="replace").strip()
                if not line.startswith(EVENT):
                    if line:
                        log.write(line + "\n")
                        log.flush()
                        tail = (tail + [line])[-12:]
                    continue
                try:
                    event = json.loads(line[len(EVENT):])
                except ValueError:
                    continue
                if event.get("kind") in ("done", "error"):
                    final = event
                else:
                    yield event
        code = proc.wait()
    finally:
        if proc.poll() is None:  # the app stopped following it
            proc.kill()
            proc.wait()
        _running.remove(proc)
        path.unlink(missing_ok=True)
    if getattr(proc, "stopped", False):
        final = {"kind": "stopped"}
    if final is None:
        detail = "\n".join(tail[-6:])
        final = {"kind": "error", "text": f"The picture maker stopped unexpectedly (exit code {code})."
                                          + (f"\n\n{detail}" if detail else "")}
    yield final


# -- the picture-making process ----------------------------------------------------------------------


def _emit(kind: str, **data) -> None:
    print(EVENT + json.dumps({"kind": kind, **data}, ensure_ascii=False), flush=True)


def reference(photo: str | Path, out_dir: Path) -> Path:
    """The photo upright, in RGB, at most about a megapixel (what the model reads it at)."""
    with Image.open(photo) as picture:
        picture = ImageOps.exif_transpose(picture).convert("RGB")
    if picture.width * picture.height > AREA:
        picture = picture.resize(fit(picture.width, picture.height), Image.Resampling.LANCZOS)
    path = out_dir / f"reference_{time.time_ns()}.png"
    picture.save(path)
    return path


Maker = Callable[[str, int, int, int, "str | None"], Image.Image]


def torch_maker(mode: str) -> Maker:
    """diffusers' FLUX.2 [klein] pipeline: one part of the model on the graphics card at a time (13 GB at most)."""
    import torch
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(str(model_dir()), dtype=torch.bfloat16)
    if torch.cuda.is_available():  # AMD ROCm and NVIDIA CUDA both
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    encoded: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def make(prompt: str, seed: int, width: int, height: int, photo: str | None) -> Image.Image:
        if prompt not in encoded:  # once for all the pictures
            encoded[prompt] = pipe.encode_prompt(prompt=prompt)[0]
        source = Image.open(photo).convert("RGB") if photo else None
        return pipe(image=source, prompt_embeds=encoded[prompt], width=width, height=height,
                    num_inference_steps=STEPS, guidance_scale=1.0,
                    generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    return make


def mlx_maker(mode: str) -> Maker:
    """mflux's FLUX.2 [klein], 8-bit on Macs with less than 32 GB of memory."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit

    kind = Flux2KleinEdit if mode == "outfit" else Flux2Klein
    model = kind(model_config=ModelConfig.flux2_klein_4b(), model_path=str(model_dir()),
                 quantize=8 if models.system_ram_gb() < 32 else None)

    def make(prompt: str, seed: int, width: int, height: int, photo: str | None) -> Image.Image:
        extra = {"image_paths": [photo]} if photo else {}
        return model.generate_image(seed=seed, prompt=prompt, num_inference_steps=STEPS, width=width,
                                    height=height, guidance=1.0, **extra).image

    return make


def _friendly(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if "out of memory" in text.lower() or "OutOfMemory" in text:
        return ("The graphics card ran out of memory. Wait until a video render has finished, close other apps "
                f"and try again.\n\n{text}")
    return text


def run(request: dict, maker: Callable[[str], Maker] | None = None) -> int:
    out = Path(request["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seeds, made = request["seeds"], 0
    photo = None
    try:
        if request.get("photo"):
            photo = reference(request["photo"], out)
        _emit("status", text="Loading the picture model (about 16 GB; the first time takes longest)…")
        if maker is None:
            maker = mlx_maker if models.BACKEND == "mlx" else torch_maker
        make_one = maker(request["mode"])
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for i, seed in enumerate(seeds):
            _emit("status", text=f"Making picture {i + 1} of {len(seeds)}…")
            picture = make_one(request["prompt"], seed, request["width"], request["height"],
                               str(photo) if photo else None)
            path = out / f"{stamp}_{request['mode']}_{seed}.png"
            picture.save(path)
            made += 1
            _emit("picture", path=str(path), seed=seed)
    except Exception as exc:
        _emit("error", text=_friendly(exc), made=made)
        return 1
    finally:
        if photo:
            photo.unlink(missing_ok=True)
    _emit("done", made=made)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m tdplay.creator JOB.json", file=sys.stderr)
        return 2
    return run(json.loads(Path(args[0]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    sys.exit(main())
