"""Smoke test of an installed TDPlay AI Video Render (the installer workflow runs it on macOS and Windows).

    INSTALLED_PYTHON tests/smoke_installed.py APP_ROOT

1. The app's server starts from the installed code and serves its page.
2. A two-segment shot renders as a background job through the installed code, with the stand-in
   model from this folder (no GPU or model weights needed), and comes out with every frame.
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
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
