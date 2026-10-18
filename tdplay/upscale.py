"""Upscale a finished video to 1080p or 4K with Real-ESRGAN on the GPU (MLX on a Mac, PyTorch elsewhere).

Two models (weights from the mlx-community MLX ports of the official Real-ESRGAN releases, BSD-3):

* ``fast``: realesr-general-x4v3 (SRVGGNetCompact, 1.2 M parameters). Quick, made for real-world video.
* ``best``: RealESRGAN_x4plus (RRDBNet, 16.7 M parameters). More detail, several times slower.

Both enlarge 4x. ffmpeg then scales the result to exactly 1920x1080 / 3840x2160 (cover + center
crop, so the frame is filled) and encodes it; the audio is copied unchanged.

Runs as its own process (``python -m tdplay.upscale IN OUT --width W --height H``) and prints
``PROGRESS done/total`` lines, like the video model runs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from tdplay import media, models

#: name -> (Hugging Face repo, pinned revision)
WEIGHTS = {
    "fast": ("mlx-community/Real-ESRGAN-general-x4v3", "e9a382fa779f227abf65ad49d4e5b90c1202d683"),
    "best": ("mlx-community/Real-ESRGAN-x4plus", "16c284e716eeec6a5634154c813b9148d35e8ea8"),
}
SCALE = 4
#: Short side of each output size.
TARGETS = {"1080p": 1080, "4k": 2160}
#: Input tile size (pixels) for the big model, which would otherwise need several GB per 704p frame.
TILE = {"fast": 0, "best": 192}
TILE_PAD = 24


def target_size(size: str, width: int, height: int) -> tuple[int, int]:
    """Output size for a ``width x height`` render: 16:9, 9:16 or square at the chosen short side."""
    short = TARGETS[size]
    long = short * 16 // 9
    aspect = width / height
    if aspect > 1.2:
        return long, short
    if aspect < 1 / 1.2:
        return short, long
    return short, short


def weights_path(quality: str) -> Path:
    """Download (once) and return the model's weights file."""
    from huggingface_hub import hf_hub_download

    repo, revision = WEIGHTS[quality]
    return Path(hf_hub_download(repo, "model.safetensors", revision=revision, local_dir=models.local_dir(repo)))


# ---------------------------------------------------------------------------
# Networks (the weight files store convolutions as (out, kh, kw, in), the MLX layout)
# ---------------------------------------------------------------------------


class _Tiled:
    """Shared frame handling: each backend supplies ``_run(rgb) -> float32 (4H, 4W, 3) in [0, 1]``."""

    tile = 0

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def frame(self, rgb: np.ndarray) -> np.ndarray:
        """Upscale one ``(H, W, 3) uint8`` frame 4x, in tiles when the model needs them."""
        h, w = rgb.shape[:2]
        if not self.tile or (h <= self.tile and w <= self.tile):
            out = self._run(rgb)
        else:
            out = np.empty((h * SCALE, w * SCALE, 3), np.float32)
            t, p = self.tile, TILE_PAD
            for y0 in range(0, h, t):
                for x0 in range(0, w, t):
                    y1, x1 = min(y0 + t, h), min(x0 + t, w)
                    # Each tile sees TILE_PAD pixels of context on every side; only its own area is kept.
                    py0, px0, py1, px1 = max(0, y0 - p), max(0, x0 - p), min(h, y1 + p), min(w, x1 + p)
                    tile = self._run(rgb[py0:py1, px0:px1])
                    oy, ox = (y0 - py0) * SCALE, (x0 - px0) * SCALE
                    out[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = tile[oy:oy + (y1 - y0) * SCALE,
                                                                             ox:ox + (x1 - x0) * SCALE]
        return np.round(out * 255.0).astype(np.uint8)


class TorchUpscaler(_Tiled):
    """Real-ESRGAN through PyTorch (spandrel), on an AMD (ROCm) or NVIDIA GPU, else the CPU."""

    def __init__(self, quality: str = "fast"):
        import spandrel
        import torch
        from safetensors.numpy import load_file

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        # Same weight files as the MLX path; convolutions go back to PyTorch's (out, in, kh, kw).
        state = {k: torch.from_numpy((v.transpose(0, 3, 1, 2) if v.ndim == 4 else v).astype(np.float32))
                 for k, v in load_file(str(weights_path(quality))).items()}
        self.model = spandrel.ModelLoader().load_from_state_dict(state).model.eval().to(self.device, self.dtype)
        self.tile = TILE[quality]

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        # frames read from ffmpeg are read-only buffers, which PyTorch can't share: copy first
        x = torch.as_tensor(np.array(rgb), device=self.device).permute(2, 0, 1)[None].to(self.dtype) / 255.0
        with torch.inference_mode():
            y = self.model(x).clamp_(0.0, 1.0)
        return y[0].permute(1, 2, 0).float().cpu().numpy()


def make_upscaler(quality: str = "fast") -> _Tiled:
    """MLX on a Mac, PyTorch everywhere else (``TDPLAY_UPSCALE_BACKEND=mlx|torch`` overrides)."""
    backend = os.environ.get("TDPLAY_UPSCALE_BACKEND") or ("mlx" if sys.platform == "darwin" else "torch")
    return Upscaler(quality) if backend == "mlx" else TorchUpscaler(quality)


class Upscaler(_Tiled):
    """Real-ESRGAN in MLX, on the Mac's GPU."""

    def __init__(self, quality: str = "fast", dtype=None):
        import mlx.core as mx

        self.mx = mx
        self.quality = quality
        self.dtype = dtype or (mx.float16 if mx.default_device().type == mx.gpu else mx.float32)
        raw = mx.load(str(weights_path(quality)))
        self.w = {k: v.astype(self.dtype) for k, v in raw.items()}
        self.tile = TILE[quality]

    # -- building blocks --------------------------------------------------------

    def _conv(self, x, name):
        return self.mx.conv2d(x, self.w[f"{name}.weight"], padding=1) + self.w[f"{name}.bias"]

    def _up(self, x, r):
        return self.mx.repeat(self.mx.repeat(x, r, axis=1), r, axis=2)

    def _srvgg(self, x):
        mx = self.mx
        out = x
        i = 0
        while f"body.{i + 1}.weight" in self.w:  # conv, PReLU pairs
            out = self._conv(out, f"body.{i}")
            alpha = self.w[f"body.{i + 1}.weight"]
            out = mx.maximum(out, 0) + alpha * mx.minimum(out, 0)
            i += 2
        out = self._conv(out, f"body.{i}")
        n, h, w, c = out.shape
        r = SCALE
        # PyTorch pixel_shuffle, NHWC: channel c*r*r + i*r + j -> pixel (h*r + i, w*r + j)
        out = out.reshape(n, h, w, c // (r * r), r, r).transpose(0, 1, 4, 2, 5, 3).reshape(n, h * r, w * r, c // (r * r))
        return out + self._up(x, r)

    def _rrdbnet(self, x):
        mx = self.mx

        def lrelu(t):
            return mx.maximum(t, 0.2 * t)

        def rdb(t, p):
            x1 = lrelu(self._conv(t, f"{p}.conv1"))
            x2 = lrelu(self._conv(mx.concatenate([t, x1], -1), f"{p}.conv2"))
            x3 = lrelu(self._conv(mx.concatenate([t, x1, x2], -1), f"{p}.conv3"))
            x4 = lrelu(self._conv(mx.concatenate([t, x1, x2, x3], -1), f"{p}.conv4"))
            x5 = self._conv(mx.concatenate([t, x1, x2, x3, x4], -1), f"{p}.conv5")
            return x5 * 0.2 + t

        feat = self._conv(x, "conv_first")
        body = feat
        b = 0
        while f"body.{b}.rdb1.conv1.weight" in self.w:
            out = rdb(rdb(rdb(body, f"body.{b}.rdb1"), f"body.{b}.rdb2"), f"body.{b}.rdb3")
            body = out * 0.2 + body
            b += 1
        feat = feat + self._conv(body, "conv_body")
        feat = lrelu(self._conv(self._up(feat, 2), "conv_up1"))
        feat = lrelu(self._conv(self._up(feat, 2), "conv_up2"))
        return self._conv(lrelu(self._conv(feat, "conv_hr")), "conv_last")

    def forward(self, x):
        """``x``: (1, H, W, 3) in [0, 1] -> (1, 4H, 4W, 3)."""
        return self._srvgg(x) if self.quality == "fast" else self._rrdbnet(x)

    # -- frames -------------------------------------------------------------------

    def _run(self, rgb: np.ndarray) -> np.ndarray:
        mx = self.mx
        x = mx.array(rgb[None].astype(np.float32) / 255.0).astype(self.dtype)
        y = mx.clip(self.forward(x), 0.0, 1.0)
        mx.eval(y)
        return np.asarray(y[0].astype(mx.float32))



# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


def _encoder_args(width: int, height: int) -> list[str]:
    encoders = subprocess.run([media.tool("ffmpeg"), "-hide_banner", "-encoders"],
                              capture_output=True, text=True, **media.NO_WINDOW).stdout
    four_k = width * height > 2_500_000
    if four_k and "hevc_videotoolbox" in encoders:  # the Mac's hardware encoder
        return ["-c:v", "hevc_videotoolbox", "-b:v", "45M", "-tag:v", "hvc1", "-pix_fmt", "yuv420p"]
    if not four_k and "h264_videotoolbox" in encoders:
        return ["-c:v", "h264_videotoolbox", "-b:v", "16M", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p"]


def upscale_video(src: str | Path, dst: str | Path, width: int, height: int, quality: str = "fast",
                  report=lambda done, total: None) -> Path:
    src, dst = Path(src), Path(dst)
    info = media.probe(src)
    model = make_upscaler(quality)
    sw, sh = info.width * SCALE, info.height * SCALE
    ffmpeg = media.tool("ffmpeg")
    reader = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(src), "-map", "0:v:0", "-fps_mode", "passthrough",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE, **media.NO_WINDOW)
    tmp = dst.with_name(f".{dst.stem}_tmp{dst.suffix}")
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height},setsar=1"
    writer = subprocess.Popen(
        [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{sw}x{sh}",
         "-r", f"{info.fps:g}", "-i", "-", "-i", str(src), "-map", "0:v", "-map", "1:a?", "-vf", vf,
         *_encoder_args(width, height), "-c:a", "copy", "-movflags", "+faststart", str(tmp)],
        stdin=subprocess.PIPE, **media.NO_WINDOW)
    frame_bytes = info.width * info.height * 3
    done = 0
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(info.height, info.width, 3)
            writer.stdin.write(model.frame(frame).tobytes())
            done += 1
            report(done, info.frames)
    finally:
        writer.stdin.close()
        reader.stdout.close()
        reader.wait()
        code = writer.wait()
    if code != 0 or done == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Upscaling failed (ffmpeg exit code {code}, {done} frames).")
    tmp.replace(dst)
    return dst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tdplay.upscale", description=__doc__.split("\n")[0])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--quality", choices=sorted(WEIGHTS), default="fast")
    args = parser.parse_args(argv)
    print(f"Upscaling to {args.width}x{args.height} with the {args.quality} model…", flush=True)

    def report(done: int, total: int) -> None:
        print(f"PROGRESS {done}/{max(total, done)}", flush=True)

    upscale_video(args.input, args.output, args.width, args.height, args.quality, report)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
