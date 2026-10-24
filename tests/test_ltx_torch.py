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


def test_a_start_video_is_pinned_at_the_start_of_the_first_chunk(monkeypatch):
    """Its last second, encoded at each pass's resolution, is what LTX-2 pins for a carry; its sound too."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_pipelines")
    from dataclasses import replace

    import ltx_pipelines.utils.helpers as helpers
    from ltx_core.conditioning.types.latent_cond import AudioConditionByLatentIndex, VideoConditionByLatentIndex
    from ltx_core.types import VIDEO_SCALE_FACTORS
    from ltx_pipelines.chunks.layout import ChunkLayout
    from ltx_pipelines.distilled import VideoPixelShape

    encoded = []

    def video_latent_from_file(video_encoder, file_path, output_shape, device, dtype):
        assert video_encoder == "video encoder"
        encoded.append(("video", file_path, output_shape.frames, output_shape.height, output_shape.width))
        return torch.zeros(1, 128, 4, output_shape.height // 32, output_shape.width // 32)

    def audio_latent_from_file(audio_encoder, file_path, output_shape, device, dtype):
        assert audio_encoder == "audio encoder"
        encoded.append(("audio", file_path, output_shape.frames))
        return torch.zeros(1, 8, 7, 16)

    monkeypatch.setattr(helpers, "video_latent_from_file", video_latent_from_file)
    monkeypatch.setattr(helpers, "audio_latent_from_file", audio_latent_from_file)
    cls = ltx_torch._pipeline_class()
    pipe = object.__new__(cls)
    pipe.device = torch.device("cpu")
    pipe.stage = SimpleNamespace(video_scale_factors=VIDEO_SCALE_FACTORS)
    pipe.image_conditioner = lambda fn: fn("video encoder")
    pipe.audio_conditioner = lambda fn: fn("audio encoder")
    layouts = [ChunkLayout(**spec) for spec in ltx_torch.plan_frames(25 + 48, [48])]
    contexts = [SimpleNamespace(video_encoding=torch.zeros(1), audio_encoding=torch.zeros(1))] * 2
    target = VideoPixelShape(batch=1, frames=121, height=128, width=192, fps=24.0)
    start = cls.StartVideo(pipe, "start_tail.mp4", 24.0, with_audio=True)
    first, second = pipe.segment_chunks(layouts, contexts, target, torch.bfloat16, lambda chunk: ["photo"], start)

    photo, pinned = first.make_video_conditionings(first)
    assert photo == "photo" and isinstance(pinned, VideoConditionByLatentIndex)
    assert (pinned.latent_idx, pinned.strength, pinned.latent.shape[2:]) == (0, 1.0, (4, 4, 6))
    (sound,) = first.make_audio_conditionings(first)
    assert isinstance(sound, AudioConditionByLatentIndex) and sound.latent_idx == 0
    assert second.make_video_conditionings(second) == ["photo"] and second.make_audio_conditionings(second) == []
    # The draft pass reuses its encoding; the refine pass (full resolution) gets its own.
    first.make_video_conditionings(first)
    full = replace(first, video=torch.zeros(1, 128, 10, 8, 12))
    assert full.make_video_conditionings(full)[1].latent.shape[3:] == (8, 12)
    assert encoded == [("video", "start_tail.mp4", 25, 128, 192), ("audio", "start_tail.mp4", 25),
                       ("video", "start_tail.mp4", 25, 256, 384)]
    assert cls.StartVideo(pipe, "start_tail.mp4", 24.0, with_audio=False).audio_conditionings(first) == []


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
        seen.update(first=first, num_frames=kwargs["num_frames"], chunk=kwargs["chunk_config"],
                    tiling=kwargs["tiling_config"])
        raise StopIteration  # stop before any model work

    pipe._prepare_run = fake_prepare_run
    with pytest.raises(StopIteration):
        pipe.stream_segments(prompts=["a", "b"], first_frames=49, new_frames=[48], seed=1, height=128, width=192,
                             frame_rate=24.0, images=[])
    assert calls == [["a", "b"]]  # one Gemma pass for every prompt
    assert seen["first"].i == 0 and seen["num_frames"] == 97
    assert isinstance(seen["chunk"], D.ChunkConfig) and seen["chunk"].chunk_pixel_frames == 73
    assert pipe.prompt_encoder is encoder  # restored afterwards
    assert seen["tiling"] == ltx_torch.decode_tiling()


def test_frames_are_decoded_whole_so_no_tile_seams_show():
    pytest.importorskip("ltx_pipelines")
    import ltx_pipelines.utils.helpers as helpers
    from ltx_core.tiling import TileSizeConfig
    from ltx_core.types import VIDEO_SCALE_FACTORS, VideoPixelShape

    for height, width in [(1024, 576), (576, 1024), (1280, 704), (704, 1280), (896, 512), (576, 576)]:
        shape = VideoPixelShape(batch=1, frames=121, height=height, width=width, fps=24)
        # LTX-2's own choice cuts a 576p frame or bigger into overlapping tiles, whose overlaps show as lines.
        auto = TileSizeConfig.from_long_side(long_side=helpers._CONV_AUTO_LONG_SIDE, height=height, width=width,
                                             scale_factors=VIDEO_SCALE_FACTORS, frames=helpers._CONV_AUTO_FRAMES)
        whole = ltx_torch.decode_tiling()
        whole.validate(VIDEO_SCALE_FACTORS, shape)
        frames, rows, cols = whole.to_splitters(VIDEO_SCALE_FACTORS)
        assert len(rows(height // 32).intervals) == 1 and len(cols(width // 32).intervals) == 1
        assert whole.frames == auto.frames  # time is still decoded in overlapping pieces, as LTX-2 does
        assert len(frames(16).intervals) == 2
        if max(height, width) > 768:
            _, auto_rows, auto_cols = auto.to_splitters(VIDEO_SCALE_FACTORS)
            assert len(auto_rows(height // 32).intervals) * len(auto_cols(width // 32).intervals) > 1


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
    from tdplay.ltx_torch import MIN_RESERVE_GB, STREAMING_SLOTS, gpu_reserve, resident_block_count

    gb = 2**30
    # A 24 GB card with ~18.5 GB free after the transformer's other weights; 0.39 GB blocks.
    slot = int(0.39 * gb)
    draft = resident_block_count(int(18.5 * gb), slot, 48, gpu_reserve("draft", 0))
    refine = resident_block_count(int(18.5 * gb), slot, 48, gpu_reserve("refine", 0))
    assert 25 <= draft <= 35 and 15 <= refine < draft
    assert resident_block_count(int(80 * gb), slot, 48, 0) == 48  # never more than the model has
    assert resident_block_count(int(3 * gb), slot, 48, int(MIN_RESERVE_GB["draft"] * gb)) == 0
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


def test_a_lora_reaches_every_block_once_whether_resident_or_streamed(tmp_path, monkeypatch):
    """LTX-2's streaming classes with a real LoRA file: W + strength * B @ A in every pass, never twice."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ltx_core")
    from ltx_core.block_streaming import builder, provider, source
    from ltx_core.block_streaming.block_fetcher import BlockFetcher
    from ltx_core.block_streaming.disk import DiskBlockReader, DiskTensorReader, LoraSource
    from ltx_core.block_streaming.pool import BufferPool
    from ltx_core.block_streaming.stream_sync import create_stream_sync
    from ltx_core.block_streaming.wrapper import BlockStreamingWrapper
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP, SDOps
    from safetensors.torch import save_file

    from tdplay import ltx_torch

    for cls, name in ((provider.WeightsProvider, "get"), (source.DiskWeightSource, "get"),
                      (builder.StreamingModelBuilder, "build")):
        monkeypatch.setattr(cls, name, getattr(cls, name))
    ltx_torch.keep_blocks_on_gpu()

    n, resident, strength = 6, frozenset({0, 1, 2}), 0.8
    torch.manual_seed(0)
    weights = [torch.randn(4, 4) for _ in range(n)]
    a, b = [torch.randn(2, 4) for _ in range(n)], [torch.randn(4, 2) for _ in range(n)]
    path, lora = str(tmp_path / "blocks.safetensors"), str(tmp_path / "lora.safetensors")
    # Each block has a named layer, like LTX-2's (attn1.to_q, ff.net.2...): LoRAs are fused into "<layer>.weight".
    save_file({f"transformer_blocks.{i}.proj.weight": w for i, w in enumerate(weights)}, path)
    save_file({**{f"diffusion_model.transformer_blocks.{i}.proj.lora_A.weight": x for i, x in enumerate(a)},
               **{f"diffusion_model.transformer_blocks.{i}.proj.lora_B.weight": x for i, x in enumerate(b)}}, lora)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.proj(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(Block() for _ in range(n))

        def forward(self, x):
            for block in self.transformer_blocks:
                x = block(x)
            return x

    with torch.device("meta"):
        model = Model()
    blocks = model.transformer_blocks
    key_map, _ = builder._scan_checkpoint_keys([path], SDOps("all").with_matching(), "transformer_blocks")
    layouts = builder._block_layouts(blocks, key_map, torch.float32)
    cpu, sync = torch.device("cpu"), create_stream_sync(torch.device("cpu"))
    slot = 64
    disk = source.DiskWeightSource(
        BufferPool(slot, 4, cpu, reuse_barrier=sync.reuse_barrier),
        BlockFetcher(DiskBlockReader(DiskTensorReader([path]), key_map, None, "transformer_blocks")),
        layouts, blocks_number=n, prefetch_depth=2)
    gpu = provider.WeightsProvider(BufferPool(slot, len(resident) + 2, cpu, reuse_barrier=sync.reuse_barrier),
                                   sync, cpu, disk, [LoraSource(lora, LTXV_LORA_COMFY_RENAMING_MAP, strength)],
                                   "transformer_blocks")
    gpu.tdplay_resident = disk.tdplay_resident = resident
    wrapper = BlockStreamingWrapper(model, blocks, gpu, cpu)

    x = torch.randn(3, 4)

    def through(lora_times):
        y = x
        for w, down, up in zip(weights, a, b, strict=True):
            y = y @ (w + lora_times * strength * up @ down).T
        return y

    def off(y, target):  # LTX-2 fuses in bf16: about 1% over six chained layers
        return float((y - target).norm() / target.norm())

    expected = through(1)
    with torch.inference_mode():
        for _ in range(3):  # resident blocks keep their fused weights, streamed ones are fused again each pass
            out = wrapper(x)
            assert off(out, expected) < 0.03 and off(out, through(0)) > 0.3 and off(out, through(2)) > 0.3
    wrapper.teardown()


def test_lora_changes_survive_rounding_to_fp8():
    """A LoRA's change smaller than half an fp8 step: nearest rounding drops it, ours keeps it on average."""
    torch = pytest.importorskip("torch")
    from tdplay import ltx_torch

    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        values = torch.tensor([1.0, -1.0, 0.3, 200.0, 2.0**-6, 0.0, 447.0])
        weight = values.repeat(50000, 1).to(dtype)
        delta = torch.tensor([0.02, -0.02, 0.004, 5.0, -0.0001, 0.001, 3.0]).repeat(50000, 1).to(torch.bfloat16)
        largest = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
        target = (weight.float() + delta.float()).clamp(-largest, largest).mean(0)
        fused = ltx_torch.round_to_fp8_stochastically(delta.clone(), weight)
        assert fused.dtype == dtype and fused.shape == weight.shape
        # On average right: within 0.3%, or 2% for the tiniest values, whose fp8 steps are coarse next to them.
        assert torch.allclose(fused.float().mean(0), target, rtol=0.003, atol=2e-5), (dtype, fused.float().mean(0))
        nearest = (weight.float() + delta.float()).to(dtype).float().mean(0)
        assert not torch.allclose(nearest, target, rtol=0.003)  # what LTX-2 does without Triton


def test_loras_go_to_ltx2_with_comfyui_names_and_rounding(monkeypatch):
    pytest.importorskip("ltx_core")
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core.quantization import fp8_cast

    from tdplay import ltx_torch

    (lora,) = ltx_torch.lora_inputs({"loras": [{"path": "/x/dual.safetensors", "strength": 0.7}]})
    assert (lora.path, lora.strength, lora.sd_ops) == ("/x/dual.safetensors", 0.7, LTXV_LORA_COMFY_RENAMING_MAP)
    assert ltx_torch.lora_inputs({}) == ()
    monkeypatch.setattr(fp8_cast, "fuse_cast_fp8_weight", fp8_cast.fuse_cast_fp8_weight)
    ltx_torch.fuse_loras_with_stochastic_rounding()
    ltx_torch.fuse_loras_with_stochastic_rounding()  # once is enough
    assert fp8_cast.fuse_cast_fp8_weight.tdplay


@pytest.fixture
def fake_gpu(monkeypatch):
    """The build glue around a fake 24 GB card: LTX-2's build replaced, torch.cuda's memory queries faked."""
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
    monkeypatch.setattr(ltx_torch, "_extra_reserve", {})
    monkeypatch.setattr(ltx_torch, "MIN_RESERVE_GB", dict(ltx_torch.GPU_MEMORY["balanced"][1]))
    card = SimpleNamespace(free=21 * gb, allocated=2 * gb, peak=4 * gb, pools=[], fraction=0.8)

    class Pool:  # stands in for a GPU buffer pool
        def __init__(self, slot_nbytes, capacity, device, **kwargs):
            card.pools.append(capacity)

    def fake_build(self, device=None, dtype=None, cpu_slots_count=None, gpu_slots_count=None, **kwargs):
        model = SimpleNamespace(blocks=torch.nn.ModuleList(torch.nn.Identity() for _ in range(48)))
        assert len(builder.resolve_attr(model, "blocks")) == 48
        builder.BufferPool(int(0.39 * gb), 2, device)
        return SimpleNamespace(_provider=SimpleNamespace(_source=SimpleNamespace()), teardown=lambda: None)

    monkeypatch.setattr(builder.StreamingModelBuilder, "build", fake_build)
    monkeypatch.setattr(builder, "BufferPool", Pool)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, device=None: setattr(card, "fraction", fraction))
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda device=None: card.fraction)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(total_memory=24 * gb))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (card.free, 24 * gb))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: card.allocated)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: card.allocated)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: card.peak)
    ltx_torch.keep_blocks_on_gpu()
    model = SimpleNamespace(blocks_prefix="transformer_blocks")

    def run(kind: str = "draft", stage=None):
        stage = stage or (lambda **kwargs: builder.StreamingModelBuilder.build(model, device="cuda"))
        latent = torch.zeros(1, 128, 7, 9, 16)  # 1008 latent tokens
        wrapper = ltx_torch.GpuPass(stage, kind)(modalities=SimpleNamespace(video=SimpleNamespace(latent=latent)))
        wrapper.teardown()
        return len(wrapper._provider.tdplay_resident)

    card.run, card.build = run, lambda: builder.StreamingModelBuilder.build(model, device="cuda")
    return card


def test_resident_blocks_leave_the_rest_of_the_card_alone(fake_gpu, monkeypatch, capsys):
    """The model gets 80 % of the card at most, and never less working memory than the first segment had."""
    from tdplay import ltx_torch

    gb = 2**30
    ltx_torch.limit_gpu_memory()
    assert fake_gpu.fraction == 0.8
    monkeypatch.setenv("TDPLAY_GPU_MEMORY_SHARE", "0.6")
    ltx_torch.limit_gpu_memory()
    assert fake_gpu.fraction == 0.6
    monkeypatch.delenv("TDPLAY_GPU_MEMORY_SHARE")
    ltx_torch.limit_gpu_memory()

    # 21 GB free, but only 0.8 * 24 - 2 = 17.2 GB within the share: 6 GB working memory, 2 slots, 26 blocks.
    first = fake_gpu.run()
    assert first == int((17.2 * gb - 6 * gb - 2 * int(0.39 * gb)) // int(0.39 * gb)) == 26
    assert fake_gpu.pools == [first + 2]
    assert f"keeping {first} of 48 transformer blocks" in capsys.readouterr().out
    # The pass needed only 2 GB: the next one still leaves 6 GB, as the first segment did.
    assert fake_gpu.run() == first
    # A card that says it has more free than it has (Windows counts shared system memory): still the share.
    fake_gpu.free = 38 * gb
    assert fake_gpu.run() == first
    # A pass that needed 10 GB: the next one leaves 10 * 1.25 + 1 GB.
    fake_gpu.peak = 12 * gb
    fake_gpu.run()
    assert fake_gpu.run() == int((17.2 * gb - 13.5 * gb - 2 * int(0.39 * gb)) // int(0.39 * gb)) == 7


def test_maximum_graphics_memory_keeps_more_blocks_on_the_card(fake_gpu):
    from tdplay import ltx_torch

    gb, slot = 2**30, int(0.39 * 2**30)
    balanced = fake_gpu.run(), fake_gpu.run("refine")
    ltx_torch.limit_gpu_memory("max")
    assert fake_gpu.fraction == 0.9
    # 0.9 * 24 - 2 = 19.6 GB within the share; 4 GB (draft) / 8 GB (refine) working memory, 2 slots.
    maximum = fake_gpu.run(), fake_gpu.run("refine")
    assert maximum == tuple(int((19.6 * gb - reserve * gb - 2 * slot) // slot) for reserve in (4, 8)) == (38, 27)
    assert maximum[0] > balanced[0] and maximum[1] > balanced[1]
    ltx_torch.limit_gpu_memory("something else")  # an unknown setting is the balanced one
    assert fake_gpu.fraction == 0.8 and (fake_gpu.run(), fake_gpu.run("refine")) == balanced


def test_a_pass_that_runs_out_of_memory_runs_again_with_more_room(fake_gpu, capsys):
    torch = pytest.importorskip("torch")
    attempts = []

    def stage(**kwargs):
        wrapper = fake_gpu.build()
        attempts.append(len(wrapper._provider.tdplay_resident))
        if len(attempts) < 3:
            raise torch.OutOfMemoryError("HIP out of memory")
        return wrapper

    assert fake_gpu.run(stage=stage) == attempts[-1]
    assert attempts == [26, 16, 6]  # 4 GB more working memory each time
    assert "out of memory in the draft pass; trying it again" in capsys.readouterr().out
    assert fake_gpu.run() == 6  # later draft passes keep the extra room
    assert fake_gpu.run("refine") == 16  # 10 GB for a refine pass, which hasn't run out

    def never_fits(**kwargs):
        raise torch.OutOfMemoryError("HIP out of memory")

    with pytest.raises(torch.OutOfMemoryError):
        fake_gpu.run(stage=never_fits)
