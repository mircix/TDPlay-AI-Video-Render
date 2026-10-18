"""Upscaler tests: real Real-ESRGAN (MLX on a Mac, PyTorch on Windows) on a tiny clip, parity, the engine step."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import fake_ltx
from tdplay import engine, media, models
from tdplay.engine import Segment

FAKE = [sys.executable, str(Path(__file__).parent / "fake_ltx.py")]
FAKE_UPSCALE = [sys.executable, str(Path(__file__).parent / "fake_upscale.py")]


@pytest.fixture(scope="module")
def weights_home(tmp_path_factory):
    """Real-ESRGAN weights are downloaded once per test run (2.4 MB for the fast model)."""
    home = tmp_path_factory.mktemp("models")
    old = models.MODELS_DIR
    models.MODELS_DIR = home
    from tdplay import upscale

    try:
        upscale.weights_path("fast")
    except Exception as exc:  # offline
        models.MODELS_DIR = old
        pytest.skip(f"cannot download Real-ESRGAN weights: {exc}")
    yield home
    models.MODELS_DIR = old


def tiny_clip(path: Path, frames: int = 12, w: int = 48, h: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    video = []
    for i in range(frames):
        f = np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 90)], -1).astype(np.uint8)
        f[8:20, 4 + i:16 + i] = (240, 200, 30)
        video.append(f)
    video = np.stack(video)
    fake_ltx.write(str(path), video, fake_ltx.tone(0, frames, 24), 24)
    return video


def test_target_sizes():
    from tdplay.upscale import target_size

    assert target_size("1080p", 896, 512) == (1920, 1080)
    assert target_size("4k", 1280, 704) == (3840, 2160)
    assert target_size("1080p", 512, 896) == (1080, 1920)
    assert target_size("4k", 576, 576) == (2160, 2160)


def use_backend(backend: str, monkeypatch) -> None:
    """Run the upscaler on ``backend`` ("mlx": the Mac one, "torch": the Windows one), if it is installed."""
    pytest.importorskip("mlx.core" if backend == "mlx" else "spandrel")
    monkeypatch.setenv("TDPLAY_UPSCALE_BACKEND", backend)


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_real_upscale_of_a_tiny_clip(tmp_path, weights_home, backend, monkeypatch):
    from tdplay import upscale

    use_backend(backend, monkeypatch)

    src = tmp_path / "in.mp4"
    tiny_clip(src)
    seen = []
    out = upscale.upscale_video(src, tmp_path / "out.mp4", 192, 128, "fast", lambda d, t: seen.append(d))
    info = media.probe(out)
    assert (info.width, info.height, info.frames, info.has_audio) == (192, 128, 12, True)
    assert seen == list(range(1, 13))
    # The upscaled frames, shrunk back, must look like the originals (catches channel/axis mix-ups).
    back = media.read_frames(out, 48, 32).astype(float)
    orig = media.read_frames(src, 48, 32).astype(float)
    psnr = 10 * np.log10(255**2 / np.mean((back - orig) ** 2))
    assert psnr > 26, psnr


def test_mlx_network_matches_pytorch(weights_home):
    pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    spandrel = pytest.importorskip("spandrel")
    import mlx.core as mx
    from safetensors.numpy import load_file

    from tdplay import upscale

    img = (np.random.default_rng(0).random((24, 40, 3)) * 255).astype(np.uint8)
    up = upscale.Upscaler("fast", dtype=mx.float32)
    mine = up._run(img)
    sd = {k: torch.from_numpy((v.transpose(0, 3, 1, 2) if v.ndim == 4 else v).astype(np.float32))
          for k, v in load_file(str(upscale.weights_path("fast"))).items()}
    model = spandrel.ModelLoader().load_from_state_dict(sd).model.eval()
    with torch.no_grad():
        ref = model(torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None]).clamp(0, 1)
    assert np.abs(mine - ref[0].permute(1, 2, 0).numpy()).max() < 1e-4


def test_mac_and_windows_upscalers_agree(weights_home):
    pytest.importorskip("mlx.core")
    pytest.importorskip("spandrel")
    import mlx.core as mx

    from tdplay import upscale

    rgb = (np.random.default_rng(2).random((20, 28, 3)) * 255).astype(np.uint8)
    on_mac = upscale.Upscaler("fast", dtype=mx.float32).frame(rgb).astype(int)
    on_windows = upscale.TorchUpscaler("fast").frame(rgb).astype(int)
    assert np.abs(on_mac - on_windows).max() <= 1


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_tiles_join_without_visible_seams(weights_home, backend, monkeypatch):
    from tdplay import upscale

    use_backend(backend, monkeypatch)
    yy, xx = np.mgrid[0:40, 0:56]
    rgb = np.stack([xx * 4, yy * 6, np.full_like(xx, 100)], -1).astype(np.uint8)
    if backend == "mlx":
        import mlx.core as mx

        up = upscale.Upscaler("fast", dtype=mx.float32)
    else:
        up = upscale.TorchUpscaler("fast")
    up.tile = 0
    whole = up.frame(rgb).astype(int)
    up.tile = 24
    tiled = up.frame(rgb).astype(int)
    assert np.abs(whole - tiled).max() <= 3


def test_engine_upscales_after_the_last_segment(tmp_path):
    settings = engine.Settings(width=512, height=256, model_dir="/m", seed=1, upscale="1080p")
    project = engine.create_project(tmp_path, settings, [Segment("a", 2), Segment("b", 1)])
    events = list(engine.Runner(project, FAKE, FAKE_UPSCALE).run())
    assert events[-1].kind == "done", events[-1].text
    assert events[-1].path.endswith("render_1080p.mp4")
    info = media.probe(events[-1].path)
    assert (info.width, info.height, info.frames, info.has_audio) == (1920, 1080, 73, True)
    assert any("frame 5/5" in e.text for e in events if e.kind == "progress")
    assert project.output.exists()  # the original-size render is kept

    # ...and a finished project can be upscaled again later, to another size.
    events = list(engine.Runner(engine.Project.load(project.dir), FAKE, FAKE_UPSCALE).upscale("4k", "best"))
    assert events[-1].kind == "done"
    assert (media.probe(events[-1].path).width, media.probe(events[-1].path).height) == (3840, 2160)
