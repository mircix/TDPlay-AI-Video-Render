"""End-to-end engine tests against ``fake_ltx.py`` (no Mac or model weights needed)."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import fake_ltx
from tdplay import engine, media
from tdplay.engine import Segment

FAKE = [sys.executable, str(Path(__file__).parent / "fake_ltx.py")]
W, H = 512, 256


def make_settings(**overrides) -> engine.Settings:
    values = dict(width=W, height=H, model_dir="/models/pack", gemma_dir="/models/gemma", seed=7)
    values.update(overrides)
    return engine.Settings(**values)


def reference(tmp_path: Path) -> str:
    path = tmp_path / "character.png"
    Image.new("RGB", (640, 480), (120, 90, 60)).save(path)
    return str(path)


def frame_numbers(path: str | Path) -> list[int]:
    return [fake_ltx.read_index(f) for f in media.read_frames(path, W, H)]


def assert_seamless(path: str | Path, expected_frames: int, first_index: int = 0) -> None:
    assert frame_numbers(path) == list(range(first_index, first_index + expected_frames))
    assert media.probe(path).frames == expected_frames
    audio = media.read_audio(path)[:, 0]
    ideal = fake_ltx.tone(first_index, expected_frames, 24)[:, 0]
    assert abs(len(audio) - len(ideal)) < 2048  # AAC pads to whole 1024-sample frames
    n = min(len(audio), len(ideal))
    snr = 10 * np.log10(np.mean(ideal[:n] ** 2) / np.mean((audio[:n] - ideal[:n]) ** 2))
    assert snr > 20, f"audio is not continuous (SNR {snr:.1f} dB)"


def run(project: engine.Project) -> list[engine.Event]:
    return list(engine.Runner(project, FAKE).run())


def test_frame_math():
    assert engine.first_segment_frames(5) == 121
    assert engine.first_segment_frames(1) == 49  # the first segment is at least 2 s
    assert engine.extend_latent_frames(5) == 15
    assert engine.extend_latent_frames(1) == 3


def test_three_segments_make_one_seamless_shot(tmp_path):
    project = engine.create_project(
        tmp_path / "projects", make_settings(anchor_strength=0.7),
        [Segment("a woman walks", 3), Segment("she keeps walking", 2), Segment("she stops", 4)],
        reference_image=reference(tmp_path),
    )
    events = run(project)
    assert events[-1].kind == "done", events[-1].text
    assert [e.kind for e in events].count("preview") == 3
    assert_seamless(events[-1].path, 73 + 48 + 96)

    # Color lock: the fake VAE adds a red bias on every extend; the shot must not drift.
    frames = media.read_frames(events[-1].path, W, H)
    drift = frames[-1][H // 4, W - 1].astype(int) - frames[0][H // 4, W - 1].astype(int)
    assert np.abs(drift).max() <= 3, drift

    saved = engine.Project.load(project.dir)
    assert saved.status == "complete" and saved.done == 3
    log = (project.dir / "log.txt").read_text()
    assert "--image" in log and "last 0.70" in log


def test_folders_with_spaces_and_apostrophes(tmp_path):
    root = tmp_path / "AI Video Workspace" / "Tom's renders"
    project = engine.create_project(root, make_settings(), [Segment("a", 2), Segment("b", 1)])
    events = run(project)
    assert events[-1].kind == "done", events[-1].text
    assert_seamless(events[-1].path, 49 + 24)


def test_without_color_lock_the_bias_accumulates(tmp_path):
    project = engine.create_project(
        tmp_path, make_settings(color_lock=False), [Segment("a", 2), Segment("b", 2), Segment("c", 2)]
    )
    events = run(project)
    frames = media.read_frames(events[-1].path, W, H)
    drift = frames[-1][H // 4, W - 1].astype(int) - frames[0][H // 4, W - 1].astype(int)
    # The fake VAE bias plus each H.264 round trip's own shift pile up without the lock.
    assert np.abs(drift).max() >= 6, drift


def test_resume_after_out_of_memory_kill(tmp_path, monkeypatch):
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2), Segment("b", 2), Segment("c", 3)])
    monkeypatch.setenv("FAKE_LTX_FAIL_ON", "seg_02")
    events = run(project)
    assert events[-1].kind == "error"
    assert "ran out of memory" in events[-1].text
    assert engine.Project.load(project.dir).done == 1

    monkeypatch.delenv("FAKE_LTX_FAIL_ON")
    events = run(engine.Project.load(project.dir))
    assert events[-1].kind == "done", events[-1].text
    assert_seamless(events[-1].path, 49 + 48 + 72)
    assert len(list((project.dir / "carry").iterdir())) == 3  # only the current carry is kept


def test_stop_then_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LTX_SLEEP", "0.15")
    project = engine.Project.load(
        engine.create_project(tmp_path, make_settings(), [Segment("a", 2), Segment("b", 2), Segment("c", 2)]).dir
    )
    runner = engine.Runner(project, FAKE)
    events: list[engine.Event] = []

    def consume():
        for event in runner.run():
            events.append(event)
            if event.kind == "preview":
                runner.cancel()

    thread = threading.Thread(target=consume)
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert events[-1].kind == "stopped", events[-1].text
    stopped = engine.Project.load(project.dir)
    assert stopped.status == "stopped" and stopped.done == 1

    monkeypatch.delenv("FAKE_LTX_SLEEP")
    events = run(stopped)
    assert events[-1].kind == "done"
    assert_seamless(events[-1].path, 49 + 48 + 48)


def test_continue_an_existing_video(tmp_path):
    start = tmp_path / "mine.mp4"
    base = fake_ltx.base_color(3)
    frames = np.stack([fake_ltx.render(i, W, H, base) for i in range(60)])
    fake_ltx.write(str(start), frames, fake_ltx.tone(0, 60, 24), 24)
    project = engine.create_project(tmp_path, make_settings(), [Segment("b", 2), Segment("c", 1)],
                                    start_video=str(start))
    events = run(project)
    assert events[-1].kind == "done", events[-1].text
    assert_seamless(events[-1].path, 60 + 48 + 24)


def test_closing_the_stream_kills_the_model_process(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LTX_SLEEP", "0.5")
    project = engine.create_project(tmp_path, make_settings(), [Segment("-a prompt that looks like a flag", 3)])
    runner = engine.Runner(project, FAKE)
    stream = runner.run()
    for event in stream:
        if event.kind == "progress" and "step" in event.text:
            break
    proc = runner._proc
    stream.close()
    assert proc.wait(timeout=15) is not None
    assert engine.Project.load(project.dir).status == "stopped"


def test_helper_modules_run_from_the_project_folder(tmp_path, monkeypatch):
    # The upscaler and the Windows backend run as ``python -m tdplay.<module>`` inside the project folder.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2)])
    runner = engine.Runner(project, FAKE)
    lines = list(runner._process([sys.executable, "-m", "tdplay.upscale", "--help"], dict(os.environ), "Upscaling"))
    assert any("usage" in line for line in lines), lines


def test_validation(tmp_path):
    with pytest.raises(ValueError, match="no prompt"):
        engine.create_project(tmp_path, make_settings(), [Segment("  ", 3)])
    with pytest.raises(ValueError, match="multiples of 64"):
        engine.create_project(tmp_path, make_settings(width=500), [Segment("a", 3)])
    with pytest.raises(ValueError, match="at least 2 s"):
        engine.create_project(tmp_path, make_settings(), [Segment("a", 1)])


def test_arguments_are_accepted_by_the_real_ltx_cli(tmp_path):
    cli = pytest.importorskip("ltx_pipelines_mlx.cli")
    project = engine.create_project(
        tmp_path, make_settings(anchor_strength=0.7), [Segment("-a dash-led prompt", 3), Segment("b", 5)],
        reference_image=reference(tmp_path),
    )
    runner = engine.Runner(project, FAKE)
    parser = cli._build_parser()

    gen = parser.parse_args(runner._generate_args(0, tmp_path / "seg_01.mp4"))
    assert (gen.command, gen.distilled, gen.frames, gen.frame_rate, gen.low_ram) == ("generate", True, 73, 24.0, True)
    assert (gen.height, gen.width, gen.gemma, gen.seed) == (H, W, "/models/gemma", 7)
    assert gen.prompt == "-a dash-led prompt"

    ext = parser.parse_args(runner._extend_args(1, tmp_path / "seg_02.mp4", tmp_path / "carry.mp4"))
    assert (ext.command, ext.distilled, ext.extend_frames, ext.direction) == ("extend", True, 15, "after")
    assert ext.seed == 8
    anchors = ext.images
    assert len(anchors) == 1 and anchors[0].frame_idx == -1 and anchors[0].strength == pytest.approx(0.7)


@pytest.mark.skipif(not os.environ.get("TDPLAY_NETWORK_TESTS"), reason="set TDPLAY_NETWORK_TESTS=1")
def test_download_plans_skip_unused_weights():
    from tdplay import models

    _, files = models.plan_files("dgrauet/ltx-2.3-mlx-q4")
    names = {n for n, _ in files}
    assert {"transformer-distilled-1.1.safetensors", "connector.safetensors", "vae_encoder.safetensors",
            "vae_decoder.safetensors", "audio_vae.safetensors", "vocoder.safetensors",
            "spatial_upscaler_x2_v1_1.safetensors", "config.json"} <= names
    assert not names & {"transformer-dev.safetensors", "transformer-distilled.safetensors",
                        "temporal_upscaler_x2_v1_0.safetensors", "spatial_upscaler_x1_5_v1_0.safetensors"}
    assert not any("lora" in n for n in names)

    _, files = models.plan_files("dgrauet/ltx-2.5-mlx-q8")
    names = {n for n, _ in files}
    assert {"transformer-distilled.safetensors", "text_encoder.safetensors", "tokenizer.json"} <= names
    assert "transformer-dev.safetensors" not in names
    assert sum(size for _, size in files) < 50e9
