"""PyTorch backend (Windows/Linux, AMD ROCm or NVIDIA CUDA): renders a whole shot with Lightricks' LTX-2.

Every segment of the shot becomes one chunk of LTX-2's own long-video mode (``ltx_pipelines``
chunks): the previous chunk's last 25 frames, and the audio under them, are pinned at the start of
the next chunk, and LTX-2 cross-fades the decoded overlap. TDPlay only adds what the official
pipeline lacks for this: one prompt per chunk, and chunk lengths that follow the segment lengths.
This is the same continuation the Mac version does, segment by segment, with ``extend --distilled``.

A shot can continue the user's own video (``start_tail`` in the job: its last 25 frames, with sound).
Those are encoded with LTX-2's video and audio encoders and pinned at the start of the first chunk,
the way a chunk's carry is pinned at the start of the next one. The first 25 decoded frames are
then the model's re-decode of the user's frames: they are cross-faded back into the originals.

The model is LTX-2.3 distilled in 8-bit (``fp8-cast``: weights stored in fp8, upcast during compute,
which works on GPUs without fp8 hardware), with the Gemma 3 text encoder. Weights stream from RAM
or from disk (``--offload``), so it fits a 24 GB card.

Run by ``tdplay.engine_torch`` as ``python -m tdplay.ltx_torch JOB.json``. It prints
``SEGMENT i/n`` when a segment starts and ``FRAMES done/total`` as frames are decoded, and writes
the decoded pieces as ``chunks/piece_NNN.mp4`` + ``.wav`` into the project folder.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path

#: Pixel frames carried from one chunk into the next (upstream ``ChunkConfig.next_video_carry_frames``).
CARRY_FRAMES = 25


#: How the VAE decodes a shot: in pieces of 80 frames that overlap by 24, as LTX-2 does, but whole frames.
#: LTX-2's automatic choice also cuts a frame longer than 768 px into overlapping tiles, and the overlaps
#: show as faint see-through lines (a cross about 70% across and 70% down a 576p shot). Its VAE decodes
#: frame by frame anyway, so a whole frame needs only a few GB of the card's memory.
DECODE_FRAMES = (80, 24)


def decode_tiling():
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig

    frames, overlap = DECODE_FRAMES
    return TileSizeConfig(frames=DimensionSizeConfig(tile_size=frames, overlap=overlap),
                          height=DimensionSizeConfig(), width=DimensionSizeConfig())


def plan_frames(first_frames: int, new_frames: list[int], carry: int = CARRY_FRAMES) -> list[dict]:
    """Chunk windows for a shot: the first segment's frames, then ``carry + new`` per later segment.

    Returns plain dicts (``ChunkLayout`` fields), so the plan can be checked without torch.
    """
    if (first_frames - 1) % 8:
        raise ValueError(f"first segment must be 8k+1 frames, got {first_frames}")
    if any(f <= 0 or f % 8 for f in new_frames):
        raise ValueError(f"later segments must add a positive multiple of 8 frames, got {new_frames}")
    sizes = [first_frames] + [carry + f for f in new_frames]
    plan, start, last = [], 0, len(sizes) - 1
    for i, size in enumerate(sizes):
        prev = 0 if i == 0 else carry
        plan.append({
            "pixel_frames": size,
            "start_pixel_frame": start - prev,
            "prev_video_carry_frames": prev,
            "next_video_carry_frames": 0 if i == last else carry,
            "next_video_blend_frames": 0 if i == last else carry,
        })
        start += size - prev
    return plan


def segment_ends(first_frames: int, new_frames: list[int]) -> list[int]:
    """Frame count of the stitched shot at the end of each segment."""
    ends, total = [], first_frames
    ends.append(total)
    for f in new_frames:
        total += f
        ends.append(total)
    return ends


def _pipeline_class():
    """Defined lazily so this module imports without torch (the plan helpers above are tested on their own)."""
    import ltx_pipelines.distilled as D
    import torch
    from ltx_core.conditioning.types.latent_cond import AudioConditionByLatentIndex, VideoConditionByLatentIndex
    from ltx_pipelines.chunks.layout import ChunkLayout
    from ltx_pipelines.utils.blocks import AudioConditioner
    from ltx_pipelines.utils.helpers import audio_latent_from_file, create_initial_av_latents, video_latent_from_file

    class SegmentedDistilledPipeline(D.DistilledPipeline):
        """``DistilledPipeline.stream_chunks`` with one prompt and one length per chunk."""

        class StartVideo:
            """The user's video a shot continues: its last frames and sound, pinned at the start of chunk 0.

            Encoded when chunk 0 is denoised, once per pass (the draft runs at half resolution, the refine at
            full), as ``VideoConditionByLatentIndex`` / ``AudioConditionByLatentIndex`` at latent index 0:
            what ``denoise_chunks`` pins for a carry from a previous chunk.
            """

            def __init__(self, pipeline, path: str, fps: float, with_audio: bool):
                self.pipeline, self.path, self.fps, self.with_audio = pipeline, path, fps, with_audio
                self._video: dict[tuple[int, int], torch.Tensor] = {}
                self._audio: list = []

            def shape(self, height: int = 32, width: int = 32):
                return D.VideoPixelShape(batch=1, frames=CARRY_FRAMES, height=height, width=width, fps=self.fps)

            def video_conditionings(self, make):
                def conditionings(chunk) -> list:
                    items = list(make(chunk))
                    if chunk.video is None:
                        return items
                    size = tuple(chunk.video.shape[3:])
                    if size not in self._video:
                        scale = self.pipeline.stage.video_scale_factors
                        shape = self.shape(size[0] * scale.height, size[1] * scale.width)
                        self._video[size] = self.pipeline.image_conditioner(lambda enc: video_latent_from_file(
                            video_encoder=enc, file_path=self.path, output_shape=shape,
                            device=self.pipeline.device, dtype=chunk.video.dtype))
                    pinned = VideoConditionByLatentIndex(latent=self._video[size], strength=1.0, latent_idx=0)
                    return [*items, pinned]

                return conditionings

            def audio_conditionings(self, chunk) -> list:
                if not self.with_audio or chunk.audio is None:  # a silent video: the prompt makes the sound
                    return []
                if not self._audio:
                    self._audio.append(self.pipeline.audio_conditioner(lambda enc: audio_latent_from_file(
                        audio_encoder=enc, file_path=self.path, output_shape=self.shape(),
                        device=self.pipeline.device, dtype=chunk.audio.dtype)))
                latent = self._audio[0]  # None if the file has no sound track
                return [] if latent is None else [AudioConditionByLatentIndex(latent, strength=1.0, latent_idx=0)]

        def __init__(self, *args, model_paths, **kwargs):
            super().__init__(*args, model_paths=model_paths, **kwargs)
            # The audio encoder (built only when a start video's sound is encoded).
            self.audio_conditioner = AudioConditioner(model_paths.audio_vae(), self.dtype, self.device)

        def segment_chunks(self, layouts, contexts, target, dtype, make_video_conditionings,
                           start: SegmentedDistilledPipeline.StartVideo | None = None) -> Iterator:
            for i, (layout, ctx) in enumerate(zip(layouts, contexts, strict=True)):
                print(f"SEGMENT {i + 1}/{len(layouts)}", flush=True)
                video, audio = create_initial_av_latents(
                    width=target.width, height=target.height, frames=layout.pixel_frames, fps=target.fps,
                    device=self.device, dtype=dtype, video_scale_factors=self.stage.video_scale_factors,
                )
                factories = {"make_video_conditionings": make_video_conditionings}
                if start is not None and i == 0:
                    factories = {"make_video_conditionings": start.video_conditionings(make_video_conditionings),
                                 "make_audio_conditionings": start.audio_conditionings}
                yield D.Chunk(
                    layout=layout, video=video, audio=audio,
                    video_context=ctx.video_encoding, audio_context=ctx.audio_encoding, **factories,
                )

        def stream_segments(self, *, prompts, first_frames, new_frames, seed, height, width, frame_rate, images,
                            start_tail: str | None = None, start_audio: bool = False):
            layouts = [ChunkLayout(**spec) for spec in plan_frames(first_frames, new_frames)]
            total = sum(layout.pixel_frames - layout.prev_video_carry_frames for layout in layouts)

            # Encode every prompt in one Gemma pass; _prepare_run then gets the first one back.
            contexts = self.prompt_encoder(list(prompts))
            encoder = self.prompt_encoder
            self.prompt_encoder = lambda _prompts, **_kwargs: [contexts[0]]
            try:
                ctx = self._prepare_run(
                    prompt=prompts[0], seed=seed, height=height, width=width, frame_rate=frame_rate, images=images,
                    num_frames=total, vae_dtype=None, tiling_config=decode_tiling(), enhance_prompt=False,
                    enhance_static_cache=False, generated_keyframes=0, decode_with_keyframes=False,
                    chunk_config=D.ChunkConfig(
                        chunk_pixel_frames=max(layout.pixel_frames for layout in layouts),
                        next_video_carry_frames=CARRY_FRAMES if len(layouts) > 1 else 0,
                    ),
                )
            finally:
                self.prompt_encoder = encoder

            target = D.VideoPixelShape(batch=1, frames=ctx.num_frames, height=ctx.stage_1_height,
                                       width=ctx.stage_1_width, fps=frame_rate)
            start = self.StartVideo(self, start_tail, frame_rate, start_audio) if start_tail else None
            chunks = self.segment_chunks(layouts, contexts, target, ctx.dtype,
                                         self._video_conditionings(ctx.images, ctx.num_frames, None), start)
            stage_1 = D.DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
            stage_2 = D.STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
            base = D.denoise_chunks(chunks, GpuPass(self.stage, "draft"), sigmas=stage_1, noiser=ctx.noiser,
                                    fps=frame_rate, noise_scale=1.0,
                                    **self._sampler_kwargs(seed, D.ANCESTRAL_NOISE_SEED_OFFSET))
            upscaled = D.spatially_upsample_chunks(base, self.upsampler)
            refined = D.denoise_chunks(upscaled, GpuPass(self.stage, "refine"), sigmas=stage_2, noiser=ctx.noiser,
                                       fps=frame_rate,
                                       **self._sampler_kwargs(seed, D.ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET))
            decoded = D.decode_chunks(refined, self.video_decoder, self.audio_decoder, fps=frame_rate,
                                      tiling_config=ctx.tiling_config, generator=ctx.generator, dtype=ctx.vae_dtype)
            return decoded, ctx.num_frames

    return SegmentedDistilledPipeline


def _to_numpy(piece, fps: int):
    """A decoded piece as ``uint8 (F, H, W, 3)`` frames and ``float32 (N, 2)`` audio at 48 kHz."""
    import torch

    from tdplay import media

    frames = (piece.video.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    samples = media.samples_for_frames(len(frames), fps)
    if piece.audio is None:
        import numpy as np

        return frames, np.zeros((samples, media.CHANNELS), np.float32)
    wave = piece.audio.waveform.float().cpu()
    if wave.shape[0] != 2:
        wave = wave.T
    if piece.audio.sampling_rate != media.SAMPLE_RATE:
        import torchaudio

        wave = torchaudio.functional.resample(wave, piece.audio.sampling_rate, media.SAMPLE_RATE)
    return frames, media.fit_audio(wave.T.contiguous().numpy(), samples)


def continue_start_video(pieces, tail_path: str, *, fps: int, width: int, height: int, seam_frames: int,
                         color_lock: bool):
    """Decoded pieces of a shot that continues the user's video, with the seam blended in.

    The first ``CARRY_FRAMES`` decoded frames are the model's re-decode of the video's last frames.
    They are replaced by the originals, cross-faded into the re-decode over the last ``seam_frames``
    (``tdplay.engine.splice``, as on the Mac). With ``color_lock``, all new frames get the color
    correction measured on them, so no seam shows where one piece meets the next either.
    """
    import numpy as np

    from tdplay import engine, media

    tail = media.read_frames(tail_path, width, height)
    tail_audio = media.fit_audio(media.read_audio(tail_path), media.samples_for_frames(len(tail), fps))
    pieces = iter(pieces)
    held = []
    for piece in pieces:  # the first piece can be shorter than the re-decoded frames: gather enough
        held.append(piece)
        if sum(len(frames) for frames, _ in held) > len(tail):
            break
    out = np.concatenate([frames for frames, _ in held])
    if len(out) <= len(tail):
        raise RuntimeError(f"The model returned {len(out)} frames, expected more than {len(tail)}.")
    luts = engine.color_lock_luts(tail, out[:len(tail)]) if color_lock else None
    yield engine.splice((tail, tail_audio), out, np.concatenate([audio for _, audio in held]), fps=fps,
                        seam_frames=seam_frames, luts=luts)
    for frames, audio in pieces:
        if luts is not None:
            engine.apply_luts(frames, luts)
        yield frames, audio


# -- LoRAs on the fp8 model -----------------------------------------------------------------------------
#
# LTX-2 folds a LoRA into each fp8 block as it reaches the GPU: dequantize, add the LoRA's change, round
# back to fp8 (``fp8_cast.fuse_cast_fp8_weight``). A LoRA's change to a weight is often smaller than the
# gap between the two fp8 numbers around it, so rounding to the nearest one drops it. LTX-2 rounds up or
# down at random instead, with odds that keep the change on average, but only in a Triton kernel for
# NVIDIA. Elsewhere (Windows has no Triton; AMD cards) it rounds to nearest. This does the same random
# rounding in plain PyTorch.

#: fp8 formats: mantissa bits, exponent of the gap between the smallest numbers, largest number.
_FP8 = {"float8_e4m3fn": (3, -9, 448.0), "float8_e5m2": (2, -16, 57344.0)}
#: Values rounded at a time: caps the temporary memory at ~0.2 GB whatever the weight's size.
_CHUNK = 1 << 22


def round_to_fp8_stochastically(delta, weight):
    """``weight + delta`` in ``weight``'s fp8 format, each value rounded up or down at random, in proportion.

    Each value goes to one of the two fp8 numbers around it, with odds by how close it is to each,
    so the result is right on average.
    """
    import torch

    mantissa, smallest, largest = _FP8[str(weight.dtype).removeprefix("torch.")]
    tiny = 2.0**smallest
    out = torch.empty_like(weight)
    flat_out, flat_weight, flat_delta = out.view(-1), weight.reshape(-1), delta.reshape(-1)
    for start in range(0, flat_weight.numel(), _CHUNK):
        x = flat_delta[start:start + _CHUNK].float().add_(flat_weight[start:start + _CHUNK].float())
        a = x.abs().clamp_(max=largest)
        near = a.to(weight.dtype).float()  # the nearest fp8 magnitude, below or above
        fraction, exponent = torch.frexp(near)
        up = torch.where(near == 0, tiny, torch.ldexp(torch.ones_like(near), exponent - 1 - mantissa)).clamp_(min=tiny)
        down = torch.where(fraction == 0.5, up / 2, up).clamp_(min=tiny)  # below a power of two the gaps halve
        below = near <= a
        low, high = torch.where(below, near, near - down), torch.where(below, near + up, near)
        chosen = torch.where(torch.rand_like(a) * (high - low) < a - low, high, low).clamp_(max=largest)
        flat_out[start:start + _CHUNK] = torch.copysign(chosen, x).to(weight.dtype)
    return out


def fuse_loras_with_stochastic_rounding() -> None:
    """Make LTX-2's fp8 LoRA fusion round stochastically where its Triton kernel can't run."""
    import torch
    from ltx_core.quantization import fp8_cast

    if getattr(fp8_cast.fuse_cast_fp8_weight, "tdplay", False):
        return
    triton_fuse = fp8_cast.fuse_cast_fp8_weight

    def fuse(delta_bf16, weight_fp8):
        if torch.version.hip is None and fp8_cast._use_triton_fp8_kernel(weight_fp8.device):
            return triton_fuse(delta_bf16, weight_fp8)
        return round_to_fp8_stochastically(delta_bf16, weight_fp8)

    fuse.tdplay = True
    fp8_cast.fuse_cast_fp8_weight = fuse


def lora_inputs(job: dict) -> tuple:
    """The job's LoRAs for LTX-2: ComfyUI-format files (``diffusion_model.`` prefix) with their strengths."""
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

    return tuple(LoraPathStrengthAndSDOps(str(lora["path"]), float(lora["strength"]), LTXV_LORA_COMFY_RENAMING_MAP)
                 for lora in job.get("loras", []))


def read_weights_without_mapping() -> None:
    """Have safetensors read model files with plain reads (``backend="pread"``) instead of mapping them.

    By default safetensors maps a whole file copy-on-write (``torch.UntypedStorage.from_file``), and
    Windows sets memory aside for all of it up front: the 29.5 GB LTX-2 file doesn't fit on a 32 GB PC.
    PyTorch doesn't notice that the mapping failed, and loading stops later with "Attempted to access
    the data pointer on an invalid python storage". The pread backend maps nothing. LTX-2 looks up
    ``safetensors.safe_open`` each time it opens a file, so this covers all of its loading.
    """
    import safetensors

    if getattr(safetensors.safe_open, "tdplay_pread", False):
        return
    original = safetensors.safe_open

    def safe_open(filename, framework, device="cpu", **kwargs):
        kwargs.setdefault("backend", "pread")
        return original(filename, framework, device=device, **kwargs)

    safe_open.tdplay_pread = True
    safetensors.safe_open = safe_open


#: Prequantized-weight scales stored next to the 8-bit Linears in Lightricks' fp8 checkpoint.
SCALE_KEY_SUFFIXES = (".weight_scale", ".bias_scale", ".input_scale")


def stream_without_scale_keys() -> None:
    """Let LTX-2 stream its 8-bit checkpoint from disk: keep the ``*_scale`` tensors out of the block plan.

    Lightricks' fp8 checkpoint stores a ``weight_scale`` and an ``input_scale`` next to each 8-bit
    Linear. The fp8-cast policy folds the weight scales into the weights from a table it reads up
    front (``fp8_cast._read_scales``) and has no use for activation scales, but LTX-2 1.4.2's
    streaming builder plans every checkpoint key as a block parameter and stops with
    ``KeyError: 'attn1.to_gate_logits.input_scale'`` (Low-RAM mode streams from disk).
    """
    from ltx_core.block_streaming import builder

    if getattr(builder._scan_checkpoint_keys, "tdplay", False):
        return
    scan = builder._scan_checkpoint_keys

    def scan_without_scales(checkpoint_paths, sd_ops, blocks_prefix):
        block_key_map, non_block_keys = scan(checkpoint_paths, sd_ops, blocks_prefix)
        block_key_map = {idx: [(key, name) for key, name in entries if not key.endswith(SCALE_KEY_SUFFIXES)]
                         for idx, entries in block_key_map.items()}
        non_block_keys = [(key, name) for key, name in non_block_keys if not key.endswith(SCALE_KEY_SUFFIXES)]
        return block_key_map, non_block_keys

    scan_without_scales.tdplay = True
    builder._scan_checkpoint_keys = scan_without_scales


# -- Keeping part of the transformer on the GPU -----------------------------------------------------------
#
# LTX-2 streams the transformer's 48 blocks through 2 GPU slots, evicting the oldest. Every denoising step
# runs the blocks in order, so with Low-RAM mode each step reads the whole transformer from disk again while
# most of the GPU's memory sits unused. These patches keep as many blocks as fit on the GPU for a whole pass
# and stream only the rest. The pass's working memory (activations) must still fit: a reserve is left free
# for it, at least a minimum and more if a pass of that kind measurably needed more.
#
# The model never gets the whole card, though. The display runs on it too, and when a card's memory runs out
# Windows doesn't fail the allocation: it moves memory to system RAM, which slows everything to a crawl and
# can make the display driver hang (black screens, a restart). So the model is held to a share of the card
# (PyTorch enforces it); a pass that runs out of memory within that is run again with more room.

#: The "Graphics card memory" setting: the share of the card the model may use, and the least working memory
#: kept free in a draft / refine pass. On a 24 GB card "max" keeps 44 / 33 of the 48 blocks on it, as many as
#: ran fine without the share; "balanced" leaves Windows and other apps more room.
GPU_MEMORY = {
    "balanced": (0.8, {"draft": 6.0, "refine": 10.0}),
    "max": (0.9, {"draft": 4.0, "refine": 8.0}),
}
#: The minimum working memory in effect (set by limit_gpu_memory).
MIN_RESERVE_GB = dict(GPU_MEMORY["balanced"][1])
#: More working memory a pass gets each time it ran out, and how often it is tried again.
OOM_STEP_GB = 4.0
OOM_RETRIES = 3
#: Blocks that keep streaming through the GPU besides the resident ones (LTX-2's own default).
STREAMING_SLOTS = 2

_pass: ContextVar[tuple[str, int]] = ContextVar("tdplay_gpu_pass", default=("refine", 0))
#: Peak working memory per latent token, measured per pass kind ("draft": half resolution, "refine": full).
_working_bytes_per_token: dict[str, float] = {}
#: Extra working memory per pass kind, after a pass of that kind ran out of memory.
_extra_reserve: dict[str, int] = {}


class GpuPass:
    """LTX-2's denoising stage, labelled with its pass ("draft" or "refine") and size while it runs."""

    def __init__(self, stage, kind: str):
        self._stage, self._kind = stage, kind

    def __call__(self, *args, **kwargs):
        import torch

        video = kwargs["modalities"].video
        tokens = video.latent.numel() // video.latent.shape[1] if video is not None else 0
        token = _pass.set((self._kind, tokens))
        try:
            for attempt in range(OOM_RETRIES + 1):
                try:
                    return self._stage(*args, **kwargs)
                except torch.OutOfMemoryError:
                    if attempt == OOM_RETRIES:
                        raise
                # Out of the except block, so the failed attempt's tensors are unreachable and can be freed.
                _extra_reserve[self._kind] = _extra_reserve.get(self._kind, 0) + int(OOM_STEP_GB * 2**30)
                print(f"GPU: out of memory in the {self._kind} pass; trying it again with fewer blocks on the card",
                      flush=True)
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            _pass.reset(token)

    def __getattr__(self, name):
        return getattr(self._stage, name)


def gpu_reserve(kind: str, tokens: int) -> int:
    """Bytes of GPU memory to keep free for a pass's working memory."""
    if os.environ.get("TDPLAY_GPU_RESERVE_GB"):
        return int(float(os.environ["TDPLAY_GPU_RESERVE_GB"]) * 2**30)
    reserve = MIN_RESERVE_GB[kind] * 2**30
    per_token = _working_bytes_per_token.get(kind)
    if per_token and tokens:  # measured, plus a quarter and 1 GB for the allocator
        reserve = max(reserve, per_token * tokens * 1.25 + 2**30)
    return int(reserve) + _extra_reserve.get(kind, 0)


def limit_gpu_memory(setting: str = "balanced") -> None:
    """Hold the model to its share of the graphics card (``TDPLAY_GPU_MEMORY_SHARE`` overrides it)."""
    import torch

    share, reserve = GPU_MEMORY.get(setting, GPU_MEMORY["balanced"])
    MIN_RESERVE_GB.update(reserve)
    if torch.cuda.is_available():
        share = float(os.environ.get("TDPLAY_GPU_MEMORY_SHARE") or share)
        torch.cuda.set_per_process_memory_fraction(min(max(share, 0.1), 1.0))


def gpu_budget(device) -> int:
    """Bytes this process can still put on the GPU: what is free on the card, within its share of it."""
    import torch

    torch.cuda.empty_cache()  # what earlier passes left in PyTorch's cache goes back to the card first
    free, total = torch.cuda.mem_get_info(device)
    total = min(total, torch.cuda.get_device_properties(device).total_memory)
    allocated = torch.cuda.memory_allocated(device)
    within_share = int(torch.cuda.get_per_process_memory_fraction(device) * total) - allocated
    return max(0, min(free + torch.cuda.memory_reserved(device) - allocated, within_share))


def resident_block_count(available: int, slot_nbytes: int, blocks: int, reserve: int) -> int:
    """How many blocks fit on the GPU next to the streaming slots and the reserve."""
    spare = available - reserve - STREAMING_SLOTS * slot_nbytes
    return max(0, min(blocks, int(spare // slot_nbytes))) if slot_nbytes > 0 else 0


def _provider_get(self, idx: int):
    """``WeightsProvider.get``, except that resident blocks are never evicted from the GPU."""
    from ltx_core.block_streaming.provider import CachedBlock
    from ltx_core.block_streaming.utils import carve_buffer, layout_nbytes

    if idx in self._cache:
        return self._cache[idx].views
    if len(self._cache) >= self._pool.capacity:
        resident = getattr(self, "tdplay_resident", frozenset())
        victim = next((i for i in self._cache if i not in resident), next(iter(self._cache)))
        evicted = self._cache.pop(victim)
        self._pool.release(evicted.raw, event=self._events.pop(victim, None))
    layout = self._source.block_layout(idx)
    raw = self._pool.acquire()
    gpu_weights = carve_buffer(raw, layout)
    cpu_buffer = self._source.get(idx)
    h2d_event = self._copy_to_gpu(idx, raw, gpu_weights, cpu_buffer, layout_nbytes(layout))
    self._source.release(idx, event=h2d_event)
    self._cache[idx] = CachedBlock(raw, gpu_weights)
    return gpu_weights


def _disk_source_get(self, idx: int):
    """``DiskWeightSource.get``, except that it reads ahead only blocks that aren't resident on the GPU.

    Read-ahead of a resident block would hold a CPU buffer that is never collected.
    """
    if idx in self._in_flight:
        raise RuntimeError(f"Block {idx} is already in flight; release it before getting it again")
    scheduled = self._scheduled.pop(idx, None)
    if scheduled is None:
        scheduled = self._schedule(idx)
    error = scheduled.status.wait()
    if error is not None:
        self._pool.release(scheduled.raw)
        raise error
    self._in_flight[idx] = scheduled.raw
    resident = getattr(self, "tdplay_resident", frozenset())
    ahead = [j for k in range(1, self._blocks_number) if (j := (idx + k) % self._blocks_number) not in resident]
    for j in ahead[: self._prefetch_depth]:
        self._ensure_scheduled(j)
    return scheduled.raw


def keep_blocks_on_gpu() -> None:
    """Keep as many transformer blocks on the GPU as fit, for each pass; stream only the rest."""
    if os.environ.get("TDPLAY_GPU_RESIDENT") == "0":
        return
    import torch
    from ltx_core.block_streaming import builder, provider, source

    if getattr(builder.StreamingModelBuilder.build, "tdplay", False):
        return
    provider.WeightsProvider.get = _provider_get
    source.DiskWeightSource.get = _disk_source_get
    original_build = builder.StreamingModelBuilder.build

    def build(self, device=None, dtype=None, cpu_slots_count=None, gpu_slots_count=None, **kwargs):
        device = torch.device(device) if device is not None else torch.device("cuda")
        if self.blocks_prefix != "transformer_blocks" or gpu_slots_count is not None or device.type != "cuda":
            return original_build(self, device=device, dtype=dtype, cpu_slots_count=cpu_slots_count,
                                  gpu_slots_count=gpu_slots_count, **kwargs)
        kind, tokens = _pass.get()
        plan = {"resident": 0, "slot": 0, "blocks": 0}
        resolve_attr, buffer_pool = builder.resolve_attr, builder.BufferPool

        def counting_resolve_attr(obj, attr):
            found = resolve_attr(obj, attr)
            plan["blocks"] = len(found)
            return found

        def gpu_pool(slot_nbytes, capacity, pool_device, **pool_kwargs):
            if torch.device(pool_device).type != "cuda":
                return buffer_pool(slot_nbytes, capacity, pool_device, **pool_kwargs)
            try:
                available = gpu_budget(pool_device)
            except (RuntimeError, AttributeError):  # the driver can't say: stream everything, as LTX-2 does
                available = 0
            resident = resident_block_count(available, slot_nbytes, plan["blocks"], gpu_reserve(kind, tokens))
            while True:  # if the GPU can't give one buffer that large after all, keep fewer blocks
                try:
                    pool = buffer_pool(slot_nbytes, max(resident + capacity * (resident < plan["blocks"]), 1),
                                       pool_device, **pool_kwargs)
                    break
                except torch.OutOfMemoryError:
                    if resident == 0:
                        raise
                    resident //= 2
            plan.update(resident=resident, slot=slot_nbytes)
            return pool

        builder.resolve_attr, builder.BufferPool = counting_resolve_attr, gpu_pool
        try:
            wrapper = original_build(self, device=device, dtype=dtype, cpu_slots_count=cpu_slots_count, **kwargs)
        finally:
            builder.resolve_attr, builder.BufferPool = resolve_attr, buffer_pool
        resident = frozenset(range(plan["resident"]))
        wrapper._provider.tdplay_resident = resident
        wrapper._provider._source.tdplay_resident = resident
        print(f"GPU: keeping {len(resident)} of {plan['blocks']} transformer blocks on the card "
              f"({len(resident) * plan['slot'] / 2**30:.1f} GB) for the {kind} pass", flush=True)

        torch.cuda.reset_peak_memory_stats(device)
        weights = torch.cuda.memory_allocated(device)
        teardown = wrapper.teardown

        def teardown_and_measure():
            if tokens:
                working = max(0, torch.cuda.max_memory_allocated(device) - weights)
                _working_bytes_per_token[kind] = max(working / tokens, _working_bytes_per_token.get(kind, 0.0))
            teardown()
            torch.cuda.empty_cache()  # give the blocks' memory back to the card between passes

        wrapper.teardown = teardown_and_measure
        return wrapper

    build.tdplay = True
    builder.StreamingModelBuilder.build = build


def render(job: dict) -> None:
    import torch
    from ltx_core.quantization.fp8_cast import build_policy
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

    read_weights_without_mapping()
    stream_without_scale_keys()
    limit_gpu_memory(job.get("gpu_memory", "balanced"))
    keep_blocks_on_gpu()
    out_dir = Path(job["chunks_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(job["fps"])
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("No GPU found by PyTorch: rendering on the CPU will be extremely slow.", flush=True)

    loras = lora_inputs(job)
    if loras:
        fuse_loras_with_stochastic_rounding()
        for lora in loras:
            print(f"LoRA: {Path(lora.path).name} at strength {lora.strength:g}", flush=True)
    pipeline = _pipeline_class()(
        model_paths=ModelPaths.from_monolith(job["checkpoint"], job["gemma_root"]),
        spatial_upsampler_path=job["upsampler"],
        loras=loras,
        quantization=build_policy(job["checkpoint"]),
        offload_mode=OffloadMode(job["offload"]),
    )
    images = [ImageConditioningInput(i["path"], int(i["frame"]), float(i["strength"])) for i in job["images"]]
    with torch.inference_mode():
        decoded, total = pipeline.stream_segments(
            prompts=job["prompts"], first_frames=int(job["first_frames"]), new_frames=[int(f) for f in job["new_frames"]],
            seed=int(job["seed"]), height=int(job["height"]), width=int(job["width"]), frame_rate=float(fps),
            images=images, start_tail=job.get("start_tail"), start_audio=bool(job.get("start_tail_audio")),
        )
        write_pieces((_to_numpy(piece, fps) for piece in decoded), job, total)


def write_pieces(pieces, job: dict, total: int) -> None:
    """Write decoded ``(frames, audio)`` pieces as ``chunks/piece_NNN.mp4`` + ``.wav``, printing ``FRAMES``.

    A shot that continues a start video gets its seam blended in first (:func:`continue_start_video`).
    """
    from tdplay import media

    fps, out_dir = int(job["fps"]), Path(job["chunks_dir"])
    if job.get("start_tail"):
        pieces = continue_start_video(pieces, job["start_tail"], fps=fps, width=int(job["width"]),
                                      height=int(job["height"]), seam_frames=int(job.get("seam_frames", 8)),
                                      color_lock=bool(job.get("color_lock", True)))
    done = 0
    for k, (frames, audio) in enumerate(pieces):
        tmp_v, tmp_a = out_dir / f"piece_{k:03d}.tmp.mp4", out_dir / f"piece_{k:03d}.tmp.wav"
        media.write_video(tmp_v, frames, fps)
        media.write_wav(tmp_a, audio)
        tmp_v.replace(out_dir / f"piece_{k:03d}.mp4")
        tmp_a.replace(out_dir / f"piece_{k:03d}.wav")
        done += len(frames)
        print(f"FRAMES {done}/{total}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m tdplay.ltx_torch JOB.json")
    render(json.loads(Path(args[0]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
