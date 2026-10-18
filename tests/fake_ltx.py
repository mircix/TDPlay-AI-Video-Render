"""Stand-in for ``ltx-2-mlx`` so the engine and UI can be tested without a Mac or model weights.

It accepts the same ``generate`` / ``extend`` flags TDPlay uses and writes videos that make seams
measurable:

* every frame carries its global frame number as 16 black/white squares along the top edge;
* the audio is a 440 Hz sine whose phase follows that global frame number;
* ``extend`` behaves like the real model: its output starts with a re-decode of the source (with a
  small red color bias, like a VAE round trip) and continues with ``8 * --extend-frames`` new frames.

Environment knobs: ``FAKE_LTX_FAIL_ON=<substring of --output>`` makes that run die with SIGKILL
(as macOS does on out-of-memory); ``FAKE_LTX_SLEEP=<seconds>`` slows each step down.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

import numpy as np

SR = 48000
BITS = 16
BLOCK = 32
BIAS = np.array([4, 0, -2])


def base_color(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(60, 160, size=3)


def render(index: int, width: int, height: int, base: np.ndarray) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = base
    x = (index * 6) % width
    frame[height // 2 :, max(0, x - 20) : x + 20] = (230, 230, 40)
    for b in range(BITS):
        on = (index >> b) & 1
        frame[:BLOCK, b * BLOCK : (b + 1) * BLOCK] = 255 if on else 0
    return frame


def read_index(frame: np.ndarray) -> int:
    value = 0
    for b in range(BITS):
        if frame[4 : BLOCK - 4, b * BLOCK + 4 : (b + 1) * BLOCK - 4].mean() > 127:
            value |= 1 << b
    return value


def tone(start_frame: int, frames: int, fps: int) -> np.ndarray:
    t = (np.arange(frames * SR // fps) + start_frame * SR // fps) / SR
    wave = 0.3 * np.sin(2 * np.pi * 440 * t)
    return np.stack([wave, wave], axis=1).astype(np.float32)


def write(path: str, frames: np.ndarray, audio: np.ndarray, fps: int) -> None:
    import tempfile
    import wave as wavmod

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    with wavmod.open(wav_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    h, w_ = frames.shape[1:3]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w_}x{h}", "-r", str(fps),
         "-i", "-", "-i", wav_path, "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", path],
        input=frames.tobytes(), check=True,
    )
    os.unlink(wav_path)


def read(path: str) -> tuple[np.ndarray, int, int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    num, den = probe[2].split("/")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), w, h, round(int(num) / int(den))


def steps(total: int) -> None:
    delay = float(os.environ.get("FAKE_LTX_SLEEP", "0"))
    for i in range(total + 1):
        print(f"\rDenoising: {100 * i // total:3d}%|####| {i}/{total} [00:00<00:00, 9.9it/s]", end="", file=sys.stderr)
        sys.stderr.flush()
        time.sleep(delay)
    print(file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "extend"):
        p = sub.add_parser(name)
        p.add_argument("--distilled", action="store_true", required=True)
        p.add_argument("--prompt", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--gemma")
        p.add_argument("--low-ram", action="store_true")
        p.add_argument("--image", nargs="+", action="append")
    gen = sub.choices["generate"]
    gen.add_argument("--height", type=int, required=True)
    gen.add_argument("--width", type=int, required=True)
    gen.add_argument("--frames", type=int, required=True)
    gen.add_argument("--frame-rate", type=int, required=True)
    ext = sub.choices["extend"]
    ext.add_argument("--video", required=True)
    ext.add_argument("--extend-frames", type=int, required=True)
    args = parser.parse_args()

    print("[Loading text encoder (Gemma)] ...", file=sys.stderr, flush=True)
    print(f"prompt: {args.prompt}", file=sys.stderr, flush=True)
    fail = os.environ.get("FAKE_LTX_FAIL_ON")
    if fail and fail in args.output:
        if hasattr(signal, "SIGKILL"):
            os.kill(os.getpid(), signal.SIGKILL)
        os._exit(137)  # Windows has no SIGKILL: exit with the code a killed process gets

    if args.command == "generate":
        assert (args.frames - 1) % 8 == 0, "frames must be on the 8k+1 grid"
        steps(8)
        steps(3)
        base = base_color(args.seed)
        frames = np.stack([render(i, args.width, args.height, base) for i in range(args.frames)])
        audio = tone(0, args.frames, args.frame_rate)
    else:
        source, w, h, fps = read(args.video)
        assert len(source) == 25, f"extend got {len(source)} source frames, TDPlay should send 25"
        steps(8)
        steps(3)
        last = read_index(source[-1])
        base = source[-1][h // 4, w - 1].astype(int)  # continue the scene's background color
        redecoded = np.clip(source.astype(int) + BIAS, 0, 255).astype(np.uint8)
        new = np.stack([render(last + 1 + i, w, h, base) for i in range(8 * args.extend_frames)])
        new = np.clip(new.astype(int) + BIAS, 0, 255).astype(np.uint8)
        frames = np.concatenate([redecoded, new])
        audio = tone(last + 1 - len(source), len(frames), fps)
    print("[Decoding video + audio + muxing] ...", file=sys.stderr, flush=True)
    write(args.output, frames, audio, args.frame_rate if args.command == "generate" else fps)


if __name__ == "__main__":
    main()
