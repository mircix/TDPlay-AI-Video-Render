"""The PyTorch backend's plan, checked against the real LTX-2 classes (skipped where ltx_pipelines is absent)."""

from types import SimpleNamespace

import pytest

from tdplay import ltx_torch


def test_plan_matches_segments():
    plan = ltx_torch.plan_frames(73, [48, 96])
    assert [c["pixel_frames"] for c in plan] == [73, 73, 121]
    assert [c["start_pixel_frame"] for c in plan] == [0, 48, 96]
    assert [c["prev_video_carry_frames"] for c in plan] == [0, 25, 25]
    assert [c["next_video_carry_frames"] for c in plan] == [25, 25, 0]
    stitched = sum(c["pixel_frames"] - c["prev_video_carry_frames"] for c in plan)
    assert stitched == ltx_torch.segment_ends(73, [48, 96])[-1] == 217
    assert ltx_torch.plan_frames(49, [])[0]["next_video_carry_frames"] == 0
    with pytest.raises(ValueError):
        ltx_torch.plan_frames(72, [48])


def test_plan_is_valid_for_ltx_pipelines_and_chunks_get_their_own_prompt():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_pipelines")
    from ltx_core.types import VIDEO_SCALE_FACTORS
    from ltx_pipelines.chunks.layout import ChunkLayout
    from ltx_pipelines.distilled import VideoPixelShape

    layouts = [ChunkLayout(**spec) for spec in ltx_torch.plan_frames(73, [48, 96])]  # validates the plan
    cls = ltx_torch._pipeline_class()
    pipe = object.__new__(cls)  # no weights: only the chunk plumbing is exercised
    pipe.device = torch.device("cpu")
    pipe.stage = SimpleNamespace(video_scale_factors=VIDEO_SCALE_FACTORS)
    contexts = [SimpleNamespace(video_encoding=torch.full((1, 3, 4), float(i)),
                                audio_encoding=torch.full((1, 3, 4), -float(i))) for i in range(3)]
    target = VideoPixelShape(batch=1, frames=217, height=128, width=192, fps=24.0)
    chunks = list(pipe.segment_chunks(layouts, contexts, target, torch.bfloat16, lambda chunk: []))
    assert [c.layout.pixel_frames for c in chunks] == [73, 73, 121]
    assert [int(c.video_context[0, 0, 0]) for c in chunks] == [0, 1, 2]
    assert [c.video.shape[2] for c in chunks] == [10, 10, 16]  # latent frames: (pixels - 1) / 8 + 1
    assert chunks[0].video.shape[3:] == (4, 6)  # 128x192 at 1/32
    assert all(c.audio is not None for c in chunks)


def test_prompts_are_encoded_once_and_the_first_reaches_prepare_run():
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_pipelines")
    import ltx_pipelines.distilled as D

    cls = ltx_torch._pipeline_class()
    pipe = object.__new__(cls)
    pipe.device = torch.device("cpu")
    calls = []
    encoded = [SimpleNamespace(video_encoding=torch.zeros(1), audio_encoding=torch.zeros(1), i=i) for i in range(2)]

    def encoder(prompts, **kwargs):
        calls.append(list(prompts))
        return encoded

    pipe.prompt_encoder = encoder
    seen = {}

    def fake_prepare_run(**kwargs):  # what the real _prepare_run does with the encoder
        (first,) = pipe.prompt_encoder([kwargs["prompt"]], enhance_first_prompt=False, enhance_static_cache=False,
                                       enhance_prompt_image=None)
        seen.update(first=first, num_frames=kwargs["num_frames"], chunk=kwargs["chunk_config"])
        raise StopIteration  # stop before any model work

    pipe._prepare_run = fake_prepare_run
    with pytest.raises(StopIteration):
        pipe.stream_segments(prompts=["a", "b"], first_frames=49, new_frames=[48], seed=1, height=128, width=192,
                             frame_rate=24.0, images=[])
    assert calls == [["a", "b"]]  # one Gemma pass for every prompt
    assert seen["first"].i == 0 and seen["num_frames"] == 97
    assert isinstance(seen["chunk"], D.ChunkConfig) and seen["chunk"].chunk_pixel_frames == 73
    assert pipe.prompt_encoder is encoder  # restored afterwards


def test_weights_are_read_without_mapping_the_files(tmp_path, monkeypatch):
    """LTX-2's loader goes through safetensors' pread backend (Windows can't map the 29.5 GB model)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_core")
    import safetensors
    from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
    from safetensors.torch import save_file

    from tdplay.ltx_torch import read_weights_without_mapping

    path = tmp_path / "model.safetensors"
    save_file({"a.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}, str(path))
    backends = []
    original = safetensors.safe_open

    def spy(*args, **kwargs):
        backends.append(kwargs.get("backend"))
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", spy)
    read_weights_without_mapping()
    read_weights_without_mapping()  # a second call doesn't wrap twice
    loaded = SafetensorsStateDictLoader().load(str(path), None, torch.device("cpu"))
    assert backends == ["pread"]
    assert loaded.sd["a.weight"].float().sum().item() == 15
