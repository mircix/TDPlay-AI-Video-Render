"""Smoke test of an installed TDPlay AI Video Render (the installer workflow runs it on macOS and Windows).

    INSTALLED_PYTHON tests/smoke_installed.py APP_ROOT

1. The app's server starts from the installed code and serves its page.
2. A two-segment shot renders as a background job through the installed code, with the stand-in
   model from this folder (no GPU or model weights needed), and comes out with every frame.
3. The scene builder downloads its cut-out model, runs it on ONNX Runtime and builds a first frame.
4. The character creator's picture library is installed (mflux on a Mac, diffusers elsewhere), and pictures
   are made in their own process through the installed code (by a stand-in model: no weights needed).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def check_server(app_root: Path, env: dict) -> None:
    port = "7911"
    server = subprocess.Popen([sys.executable, "-m", "tdplay"], cwd=app_root,
                              env=dict(env, TDPLAY_PORT=port, TDPLAY_NO_BROWSER="1"))
    try:
        deadline = time.time() + 180
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as page:
                    assert b"TDPlay AI Video Render" in page.read()
                    break
            except OSError:
                assert server.poll() is None, "the server stopped"
                assert time.time() < deadline, "the server did not start"
                time.sleep(1)
    finally:
        server.terminate()
        server.wait(timeout=30)
    print("1. the app's server starts and serves the page", flush=True)


def check_render(home: Path) -> None:
    from tdplay import engine, media, worker

    backend = "mlx" if sys.platform == "darwin" else "torch"
    models = home / "models-fake"
    models.mkdir()
    (models / "ltx-2.3-22b-distilled-fp8.safetensors").write_bytes(b"")
    settings = engine.Settings(width=512, height=256, model_dir=str(models), gemma_dir=str(models), seed=3,
                               backend=backend, upsampler=str(models / "up.safetensors"))
    project = engine.create_project(home / "projects", settings, [engine.Segment("a", 2), engine.Segment("b", 1)])
    proc = worker.start(project.dir, ["render"])
    deadline = time.time() + 300
    for event in worker.follow(project.dir, proc, poll=0.2):
        assert time.time() < deadline, "the render took too long"
        if event is not None and event.kind != "progress":
            print("   ", event.kind, event.text.splitlines()[0] if event.text else "", flush=True)
            last = event
    assert last.kind == "done", last.text
    frames = media.probe(last.path).frames
    assert frames == 49 + 24, frames
    assert proc.wait(timeout=60) == 0 and not worker.is_running(project.dir)
    print(f"2. a {frames}-frame shot rendered as a background job", flush=True)
    print(f"   ffmpeg: {media.tool('ffmpeg')}", flush=True)


def check_scene(home: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from tdplay import scene

    figure = Image.new("RGB", (600, 800), (230, 230, 232))
    ImageDraw.Draw(figure).rectangle((240, 150, 360, 760), fill=(40, 60, 140))
    figure.save(home / "figure.png")
    cutter = scene.Cutter()  # downloads the model into TDPLAY_HOME and checks its checksum
    mask = cutter(figure)
    assert mask[300:700, 260:340].mean() > 0.9 and mask[:, :150].mean() < 0.1, "the cut-out model missed the figure"
    built = scene.build(None, [str(home / "figure.png")], [scene.Placement()], 1024, 576, home / "scenes",
                        cutter=cutter)
    with Image.open(built) as picture:
        assert picture.size == (1920, 1080)
        assert np.asarray(picture)[700, 960, 2] > 100  # the figure stands in the middle
    print("Scene builder OK.")


def check_creator(home: Path) -> None:
    from PIL import Image

    from tdplay import creator

    if sys.platform == "darwin":
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit  # noqa: F401

        ModelConfig.flux2_klein_4b()
    else:
        from diffusers import Flux2KleinPipeline  # noqa: F401
    os.environ["TDPLAY_CREATOR_CMD"] = shlex.join([sys.executable, str(TESTS / "fake_creator.py")])
    request = creator.describe_job("a woman with long black hair", True, "Full body (2:3)", count=2)
    events = list(creator.make(request))
    assert events[-1]["kind"] == "done", events
    pictures = [e["path"] for e in events if e["kind"] == "picture"]
    assert len(pictures) == 2
    with Image.open(pictures[0]) as picture:
        assert picture.size == creator.SHAPES["Full body (2:3)"]
    print("Character creator OK.")


def main() -> None:
    app_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(app_root))  # the installed code, not this checkout's
    import tdplay

    assert Path(tdplay.__file__).resolve().is_relative_to(app_root), tdplay.__file__
    home = Path(tempfile.mkdtemp(prefix="tdplay-smoke-"))
    fake = TESTS / ("fake_ltx.py" if sys.platform == "darwin" else "fake_ltx_torch.py")
    env = dict(os.environ, TDPLAY_HOME=str(home), TDPLAY_NO_NOTIFY="1",
               TDPLAY_LTX_CMD=shlex.join([sys.executable, str(fake)]),
               TDPLAY_TORCH_CMD=shlex.join([sys.executable, str(fake)]))
    from tdplay import media

    env["PATH"] = os.pathsep.join([str(Path(media.tool("ffmpeg")).parent), env.get("PATH", "")])
    os.environ.update(env)
    check_server(app_root, env)
    check_render(home)
    check_scene(home)
    check_creator(home)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
