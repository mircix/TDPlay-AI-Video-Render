"""The Windows/Linux runner (whole shot in one LTX-2 run) against tests/fake_ltx_torch.py."""

from __future__ import annotations

import sys
from pathlib import Path

import fake_ltx
import numpy as np

from tdplay import engine, media
from tdplay.engine import Segment
from tdplay.engine_torch import TorchRunner
from test_engine import H, W, assert_seamless, reference

FAKE = [sys.executable, str(Path(__file__).parent / "fake_ltx_torch.py")]
FAKE_UPSCALE = [sys.executable, str(Path(__file__).parent / "fake_upscale.py")]


def torch_settings(tmp_path: Path, **overrides) -> engine.Settings:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    (models / "ltx-2.3-22b-distilled-fp8.safetensors").write_bytes(b"")
    values = dict(width=W, height=H, model_dir=str(models), gemma_dir="/models/gemma", upsampler="/models/up.safetensors",
                  backend="torch", offload="disk", seed=5)
    values.update(overrides)
    return engine.Settings(**values)


def run(project, **kw):
    return list(TorchRunner(project, FAKE, FAKE_UPSCALE).run())


def test_job_has_one_prompt_and_length_per_segment(tmp_path):
    project = engine.create_project(
        tmp_path / "p", torch_settings(tmp_path, anchor_strength=0.6),
        [Segment("a", 3), Segment("b", 2), Segment("c", 4)], reference_image=reference(tmp_path))
    job = TorchRunner(project, FAKE).job()
    assert job["prompts"] == ["a", "b", "c"]
    project.characters = "Mia: red coat"
    assert TorchRunner(project, FAKE).job()["prompts"] == [f"{x}. Mia: red coat." for x in "abc"]
    project.characters = ""
    assert (job["first_frames"], job["new_frames"]) == (73, [48, 96])
    assert job["checkpoint"].endswith("ltx-2.3-22b-distilled-fp8.safetensors")
    # the photo starts the shot, and re-anchors the last frame of each later segment
    assert [(i["frame"], i["strength"]) for i in job["images"]] == [(0, 1.0), (120, 0.6), (216, 0.6)]
    assert job["offload"] == "disk" and job["seed"] == 5
    project.settings.image_strength = 0.85  # a built scene: the model may repaint the first frame a little
    assert TorchRunner(project, FAKE).job()["images"][0]["strength"] == 0.85


def test_whole_shot_with_previews(tmp_path):
    project = engine.create_project(tmp_path / "p", torch_settings(tmp_path),
                                    [Segment("a", 3), Segment("b", 2), Segment("c", 4)])
    events = run(project)
    assert events[-1].kind == "done", events[-1].text
    assert [e.text for e in events if e.kind == "preview"] == [f"Segment {i} of 3 done." for i in (1, 2, 3)]
    assert any("denoising step 11/11" in e.text for e in events if e.kind == "progress")
    assert_seamless(events[-1].path, 73 + 48 + 96)


def test_failure_then_resume_restarts_the_shot(tmp_path, monkeypatch):
    project = engine.create_project(tmp_path / "p", torch_settings(tmp_path), [Segment("a", 2), Segment("b", 2)])
    monkeypatch.setenv("FAKE_LTX_FAIL_ON", "2")
    events = run(project)
    assert events[-1].kind == "error" and "out of memory" in events[-1].text
    assert engine.Project.load(project.dir).done == 1

    monkeypatch.delenv("FAKE_LTX_FAIL_ON")
    events = run(engine.Project.load(project.dir))
    assert events[-1].kind == "done"
    assert_seamless(events[-1].path, 49 + 48)


def test_upscale_after_the_shot(tmp_path):
    project = engine.create_project(tmp_path / "p", torch_settings(tmp_path, upscale="4k"), [Segment("a", 2)])
    events = run(project)
    assert events[-1].path.endswith("render_4k.mp4")
    assert (media.probe(events[-1].path).width, media.probe(events[-1].path).height) == (3840, 2160)


def start_video(tmp_path: Path, frames: int = 60, sound: bool = True) -> str:
    """A clip of your own: numbered frames 0.., with the matching tone (or silence)."""
    path = tmp_path / "mine.mp4"
    base = fake_ltx.base_color(3)
    audio = fake_ltx.tone(0, frames, 24) * (1 if sound else 0)
    fake_ltx.write(str(path), np.stack([fake_ltx.render(i, W, H, base) for i in range(frames)]), audio, 24)
    return str(path)


def test_continue_an_existing_video(tmp_path):
    project = engine.create_project(tmp_path / "p", torch_settings(tmp_path), [Segment("b", 2), Segment("c", 1)],
                                    start_video=start_video(tmp_path))
    events = run(project)
    assert events[-1].kind == "done", events[-1].text
    assert [e.text for e in events if e.kind == "preview"] == ["Segment 1 of 2 done.", "Segment 2 of 2 done."]
    # The clip, then the new frames, numbered on without a gap, with the tone running through the seam.
    assert_seamless(events[-1].path, 60 + 48 + 24)
    # The model's VAE color shift is cancelled at the seam and after it.
    frames = media.read_frames(events[-1].path, W, H)
    drift = frames[-1][H // 4, W - 1].astype(int) - frames[0][H // 4, W - 1].astype(int)
    assert np.abs(drift).max() <= 3, drift


def test_continue_with_a_short_first_segment(tmp_path):
    # 1 s adds 24 frames: the first decoded piece is just the re-decoded second, the seam spans two pieces.
    project = engine.create_project(tmp_path / "p", torch_settings(tmp_path), [Segment("b", 1)],
                                    start_video=start_video(tmp_path, frames=25))
    events = run(project)
    assert events[-1].kind == "done", events[-1].text
    assert_seamless(events[-1].path, 25 + 24)


def test_job_for_a_continued_video(tmp_path):
    project = engine.create_project(
        tmp_path / "p", torch_settings(tmp_path, anchor_strength=0.6), [Segment("b", 2), Segment("c", 4)],
        reference_image=reference(tmp_path), start_video=start_video(tmp_path))
    job = TorchRunner(project, FAKE).job()
    # The first chunk is the clip's last second plus segment 1's new frames.
    assert (job["first_frames"], job["new_frames"]) == (25 + 48, [96])
    # The clip starts the shot, so the photo only re-anchors the end of each segment.
    assert [(i["frame"], i["strength"]) for i in job["images"]] == [(72, 0.6), (168, 0.6)]
    assert Path(job["start_tail"]).name == "start_tail.mp4" and job["start_tail_audio"] is True
    assert media.probe(job["start_tail"]).frames == 25 and media.probe(job["start_tail"]).has_audio
    assert media.probe(project.path("chunks", "start_head.mp4")).frames == 35
    assert (job["seam_frames"], job["color_lock"]) == (8, True)

    silent = engine.create_project(tmp_path / "q", torch_settings(tmp_path), [Segment("b", 2)],
                                   start_video=start_video(tmp_path, sound=False))
    assert TorchRunner(silent, FAKE).job()["start_tail_audio"] is False  # the prompt makes the sound
    plain = engine.create_project(tmp_path / "r", torch_settings(tmp_path), [Segment("a", 2)])
    assert TorchRunner(plain, FAKE).job()["start_tail"] is None
