"""PyTorch backend (Windows/Linux, AMD ROCm or NVIDIA CUDA): renders a whole shot with Lightricks' LTX-2.

Every segment of the shot becomes one chunk of LTX-2's own long-video mode (``ltx_pipelines``
chunks): the previous chunk's last 25 frames, and the audio under them, are pinned at the start of
the next chunk, and LTX-2 cross-fades the decoded overlap. TDPlay only adds what the official
pipeline lacks for this: one prompt per chunk, and chunk lengths that follow the segment lengths.
This is the same continuation the Mac version does, segment by segment, with ``extend --distilled``.

The model is LTX-2.3 distilled in 8-bit (``fp8-cast``: weights stored in fp8, upcast during compute,
which works on GPUs without fp8 hardware), with the Gemma 3 text encoder. Weights stream from RAM
or from disk (``--offload``), so it fits a 24 GB card.

Run by ``tdplay.engine_torch`` as ``python -m tdplay.ltx_torch JOB.json``. It prints
``SEGMENT i/n`` when a segment starts and ``FRAMES done/total`` as frames are decoded, and writes
the decoded pieces as ``chunks/piece_NNN.mp4`` + ``.wav`` into the project folder.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

#: Pixel frames carried from one chunk into the next (upstream ``ChunkConfig.next_video_carry_frames``).
CARRY_FRAMES = 25


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
    from ltx_pipelines.chunks.layout import ChunkLayout
    from ltx_pipelines.utils.helpers import create_initial_av_latents

    class SegmentedDistilledPipeline(D.DistilledPipeline):
        """``DistilledPipeline.stream_chunks`` with one prompt and one length per chunk."""

        def segment_chunks(self, layouts, contexts, target, dtype, make_video_conditionings) -> Iterator:
            for i, (layout, ctx) in enumerate(zip(layouts, contexts, strict=True)):
                print(f"SEGMENT {i + 1}/{len(layouts)}", flush=True)
                video, audio = create_initial_av_latents(
                    width=target.width, height=target.height, frames=layout.pixel_frames, fps=target.fps,
                    device=self.device, dtype=dtype, video_scale_factors=self.stage.video_scale_factors,
                )
                yield D.Chunk(
                    layout=layout, video=video, audio=audio,
                    video_context=ctx.video_encoding, audio_context=ctx.audio_encoding,
                    make_video_conditionings=make_video_conditionings,
                )

        def stream_segments(self, *, prompts, first_frames, new_frames, seed, height, width, frame_rate, images):
            layouts = [ChunkLayout(**spec) for spec in plan_frames(first_frames, new_frames)]
            total = sum(layout.pixel_frames - layout.prev_video_carry_frames for layout in layouts)

            # Encode every prompt in one Gemma pass; _prepare_run then gets the first one back.
            contexts = self.prompt_encoder(list(prompts))
            encoder = self.prompt_encoder
            self.prompt_encoder = lambda _prompts, **_kwargs: [contexts[0]]
            try:
                ctx = self._prepare_run(
                    prompt=prompts[0], seed=seed, height=height, width=width, frame_rate=frame_rate, images=images,
                    num_frames=total, vae_dtype=None, tiling_config=D.AUTO_TILING, enhance_prompt=False,
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
            chunks = self.segment_chunks(layouts, contexts, target, ctx.dtype,
                                         self._video_conditionings(ctx.images, ctx.num_frames, None))
            stage_1 = D.DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
            stage_2 = D.STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
            base = D.denoise_chunks(chunks, self.stage, sigmas=stage_1, noiser=ctx.noiser, fps=frame_rate,
                                    noise_scale=1.0, **self._sampler_kwargs(seed, D.ANCESTRAL_NOISE_SEED_OFFSET))
            upscaled = D.spatially_upsample_chunks(base, self.upsampler)
            refined = D.denoise_chunks(upscaled, self.stage, sigmas=stage_2, noiser=ctx.noiser, fps=frame_rate,
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


def render(job: dict) -> None:
    import torch
    from ltx_core.quantization.fp8_cast import build_policy
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import ImageConditioningInput, OffloadMode

    from tdplay import media

    read_weights_without_mapping()
    stream_without_scale_keys()
    out_dir = Path(job["chunks_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(job["fps"])
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("No GPU found by PyTorch: rendering on the CPU will be extremely slow.", flush=True)

    pipeline = _pipeline_class()(
        model_paths=ModelPaths.from_monolith(job["checkpoint"], job["gemma_root"]),
        spatial_upsampler_path=job["upsampler"],
        loras=(),
        quantization=build_policy(job["checkpoint"]),
        offload_mode=OffloadMode(job["offload"]),
    )
    images = [ImageConditioningInput(i["path"], int(i["frame"]), float(i["strength"])) for i in job["images"]]
    with torch.inference_mode():
        decoded, total = pipeline.stream_segments(
            prompts=job["prompts"], first_frames=int(job["first_frames"]), new_frames=[int(f) for f in job["new_frames"]],
            seed=int(job["seed"]), height=int(job["height"]), width=int(job["width"]), frame_rate=float(fps),
            images=images,
        )
        done = 0
        for k, piece in enumerate(decoded):
            frames, audio = _to_numpy(piece, fps)
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
    render(json.loads(Path(args[0]).read_text()))


if __name__ == "__main__":
    main()
