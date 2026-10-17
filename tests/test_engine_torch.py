"""The Windows/Linux runner (whole shot in one LTX-2 run) against tests/fake_ltx_torch.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
    assert (job["first_frames"], job["new_frames"]) == (73, [48, 96])
    assert job["checkpoint"].endswith("ltx-2.3-22b-distilled-fp8.safetensors")
    # the photo starts the shot, and re-anchors the last frame of each later segment
    assert [(i["frame"], i["strength"]) for i in job["images"]] == [(0, 1.0), (120, 0.6), (216, 0.6)]
    assert job["offload"] == "disk" and job["seed"] == 5


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


def test_continuing_a_video_is_mac_only(tmp_path):
    with pytest.raises(ValueError, match="Windows"):
        engine.create_project(tmp_path / "p", torch_settings(tmp_path), [Segment("a", 2)], start_video="x.mp4")
