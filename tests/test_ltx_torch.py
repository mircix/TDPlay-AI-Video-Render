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


def test_lightricks_fp8_checkpoint_streams_from_disk(tmp_path, monkeypatch):
    """Low-RAM mode streams blocks from disk; the fp8 checkpoint's *_scale tensors must not stop it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_core")
    from ltx_core.block_streaming import builder
    from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer.model_configurator import LTXV_MODEL_COMFY_RENAMING_MAP
    from ltx_core.quantization.fp8_cast import build_policy
    from safetensors.torch import save_file

    from tdplay.ltx_torch import stream_without_scale_keys

    # Laid out like ltx-2.3-22b-distilled-fp8: bf16 blocks, and 8-bit blocks whose Linears carry a
    # weight_scale and an input_scale. The values are exact in fp8, so the check can be exact too.
    w = torch.tensor([[0.5, 1.0, 1.5, 2.0]] * 4)
    p = "model.diffusion_model.transformer_blocks."
    tensors = {p + "0.attn1.to_q.weight": w.to(torch.bfloat16), "model.diffusion_model.proj_out.weight": w}
    for layer in ("attn1.to_q", "attn1.to_gate_logits"):
        tensors[f"{p}1.{layer}.weight"] = (w / 2).to(torch.float8_e4m3fn)
        tensors[f"{p}1.{layer}.weight_scale"] = torch.tensor(2.0)
        tensors[f"{p}1.{layer}.input_scale"] = torch.tensor(0.25)
    path = str(tmp_path / "ltx-fp8.safetensors")
    save_file(tensors, path)
    policy = build_policy(path)
    sd_ops = SDOps("chain", mapping=(*LTXV_MODEL_COMFY_RENAMING_MAP.mapping, *policy.sd_ops.mapping))

    monkeypatch.setattr(builder, "_scan_checkpoint_keys", builder._scan_checkpoint_keys)  # undone after the test
    blocks, _ = builder._scan_checkpoint_keys([path], sd_ops, "transformer_blocks")
    assert any(name.endswith("input_scale") for _, name in blocks[1])  # what made LTX-2 1.4.2 stop

    stream_without_scale_keys()
    stream_without_scale_keys()  # a second call doesn't wrap twice
    blocks, non_block = builder._scan_checkpoint_keys([path], sd_ops, "transformer_blocks")
    assert sorted(name for _, name in blocks[1]) == ["attn1.to_gate_logits.weight", "attn1.to_q.weight"]
    assert [name for _, name in non_block] == ["proj_out.weight"]

    # LTX-2's own disk reader then fills block 1, with the weight scales folded in.
    target = {"attn1.to_q.weight": torch.empty(4, 4, dtype=torch.float8_e4m3fn),  # stays 8-bit
              "attn1.to_gate_logits.weight": torch.empty(4, 4, dtype=torch.bfloat16)}  # upcast to bf16
    DiskBlockReader(DiskTensorReader([path]), blocks, sd_ops, "transformer_blocks").read_into(target, 1)
    for value in target.values():
        assert torch.equal(value.float(), w)


def test_resident_blocks_fit_next_to_the_reserve():
    from tdplay.ltx_torch import FIRST_RESERVE_GB, STREAMING_SLOTS, gpu_reserve, resident_block_count

    gb = 2**30
    # A 24 GB card with ~18.5 GB free after the transformer's other weights; 0.39 GB blocks.
    slot = int(0.39 * gb)
    draft = resident_block_count(int(18.5 * gb), slot, 48, gpu_reserve("draft", 0))
    refine = resident_block_count(int(18.5 * gb), slot, 48, gpu_reserve("refine", 0))
    assert 25 <= draft <= 35 and 15 <= refine < draft
    assert resident_block_count(int(80 * gb), slot, 48, 0) == 48  # never more than the model has
    assert resident_block_count(int(3 * gb), slot, 48, int(FIRST_RESERVE_GB["draft"] * gb)) == 0
    assert STREAMING_SLOTS == 2


def test_streaming_keeps_resident_blocks_on_the_gpu(tmp_path, monkeypatch):
    """LTX-2's own streaming classes on the CPU: resident blocks are read once, the rest every pass."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_core")
    from ltx_core.block_streaming import builder, provider, source
    from ltx_core.block_streaming.block_fetcher import BlockFetcher
    from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader
    from ltx_core.block_streaming.pool import BufferPool
    from ltx_core.block_streaming.stream_sync import create_stream_sync
    from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
    from ltx_core.loader.sd_ops import SDOps
    from safetensors.torch import save_file

    from tdplay import ltx_torch

    for cls, name in ((provider.WeightsProvider, "get"), (source.DiskWeightSource, "get"),
                      (builder.StreamingModelBuilder, "build")):
        monkeypatch.setattr(cls, name, getattr(cls, name))  # undone after the test
    ltx_torch.keep_blocks_on_gpu()

    n, resident = 6, frozenset({0, 1, 2})
    weights = [torch.randn(4, 4) for _ in range(n)]
    path = str(tmp_path / "blocks.safetensors")
    save_file({f"transformer_blocks.{i}.weight": w for i, w in enumerate(weights)}, path)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4, bias=False) for _ in range(n))

        def forward(self, x):
            for block in self.transformer_blocks:
                x = block(x)
            return x

    with torch.device("meta"):
        model = Model()
    blocks = model.transformer_blocks
    key_map, _ = builder._scan_checkpoint_keys([path], SDOps("all").with_matching(), "transformer_blocks")
    layouts = builder._block_layouts(blocks, key_map, torch.float32)
    reads = []

    class CountingReader(DiskTensorReader):
        def get_tensor(self, key):
            reads.append(int(key.split(".")[1]))
            return super().get_tensor(key)

    cpu, sync = torch.device("cpu"), create_stream_sync(torch.device("cpu"))
    slot = max(sum(t.numel() * 4 for t in model.transformer_blocks[0].parameters()), 64)
    disk = source.DiskWeightSource(
        BufferPool(slot, 2 + 2, cpu, reuse_barrier=sync.reuse_barrier),
        BlockFetcher(DiskBlockReader(CountingReader([path]), key_map, None, "transformer_blocks")),
        layouts, blocks_number=n, prefetch_depth=2)
    gpu = provider.WeightsProvider(BufferPool(slot, len(resident) + 2, cpu, reuse_barrier=sync.reuse_barrier),
                                   sync, cpu, disk)
    gpu.tdplay_resident = disk.tdplay_resident = resident
    wrapper = BlockStreamingWrapper(model, blocks, gpu, cpu)

    x = torch.randn(3, 4)
    expected = x
    for w in weights:
        expected = expected @ w.T
    with torch.inference_mode():
        for _ in range(3):
            assert torch.allclose(wrapper(x), expected, atol=1e-5)
    wrapper.teardown()
    counts = [reads.count(i) for i in range(n)]
    # Resident blocks are read from disk once; streamed ones once per pass, and the read-ahead after the
    # last block wraps around to the next streamed ones (3 and 4), as LTX-2's own read-ahead does.
    assert counts == [1, 1, 1, 4, 4, 3], counts


def test_resident_block_count_learns_from_the_measured_pass(monkeypatch, capsys):
    """The build glue: sizes the GPU pool from free memory, then uses the pass's measured peak."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_core")
    from types import SimpleNamespace

    from ltx_core.block_streaming import builder, provider, source

    from tdplay import ltx_torch

    gb = 2**30
    for cls, name in ((provider.WeightsProvider, "get"), (source.DiskWeightSource, "get"),
                      (builder.StreamingModelBuilder, "build")):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    monkeypatch.setattr(ltx_torch, "_working_bytes_per_token", {})
    pools = []

    class Pool:  # stands in for a GPU buffer pool
        def __init__(self, slot_nbytes, capacity, device, **kwargs):
            pools.append(capacity)

    def fake_build(self, device=None, dtype=None, cpu_slots_count=None, gpu_slots_count=None, **kwargs):
        model = SimpleNamespace(blocks=torch.nn.ModuleList(torch.nn.Identity() for _ in range(48)))
        blocks = builder.resolve_attr(model, "blocks")
        builder.BufferPool(int(0.39 * gb), 2, device)
        assert len(blocks) == 48
        return SimpleNamespace(_provider=SimpleNamespace(_source=SimpleNamespace()), teardown=lambda: None)

    monkeypatch.setattr(builder.StreamingModelBuilder, "build", fake_build)
    monkeypatch.setattr(builder, "BufferPool", Pool)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(18.5 * gb), 24 * gb))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 6 * gb)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 8 * gb)  # 2 GB working memory
    ltx_torch.keep_blocks_on_gpu()
    model = SimpleNamespace(blocks_prefix="transformer_blocks")

    stage = ltx_torch.GpuPass(lambda **kwargs: builder.StreamingModelBuilder.build(model, device="cuda"), "draft")
    latent = torch.zeros(1, 128, 7, 9, 16)  # 1008 latent tokens
    first = stage(modalities=SimpleNamespace(video=SimpleNamespace(latent=latent)))
    first.teardown()
    second = stage(modalities=SimpleNamespace(video=SimpleNamespace(latent=latent)))
    resident_first = len(first._provider.tdplay_resident)
    resident_second = len(second._provider.tdplay_resident)
    assert pools == [resident_first + 2, resident_second + 2]
    provider = first._provider
    assert provider._source.tdplay_resident == provider.tdplay_resident == frozenset(range(resident_first))
    assert 0 < resident_first < resident_second <= 48  # 6 GB first reserve, then the measured 2 GB (+25%, +1 GB)
    assert f"keeping {resident_first} of 48 transformer blocks" in capsys.readouterr().out
