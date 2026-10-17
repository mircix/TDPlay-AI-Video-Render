"""ffmpeg helpers: read and write frames and audio, build tail clips, assemble the master video.

Everything here works on plain numpy arrays: video is ``uint8 (frames, height, width, 3)`` RGB and
audio is ``float32 (samples, 2)`` stereo at :data:`SAMPLE_RATE`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000
CHANNELS = 2
#: Quality of the chunks the master video is assembled from (visually lossless).
CHUNK_CRF = 14
#: Quality of the tail clip handed to the model (near-lossless; it is only read back).
TAIL_CRF = 8
#: Keeps every ffmpeg / model process from opening a console window on Windows.
NO_WINDOW: dict = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _bin(name: str) -> str:
    path = shutil.which(name)
    if path is None and os.name == "nt":
        # winget puts ffmpeg here; until the next sign-in, the app may not have it on its PATH yet.
        links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
        if (links / f"{name}.exe").exists():
            os.environ["PATH"] = f"{links}{os.pathsep}{os.environ.get('PATH', '')}"  # child processes too
            path = str(links / f"{name}.exe")
    if path is None:
        install = "winget install Gyan.FFmpeg" if os.name == "nt" else "brew install ffmpeg"
        raise RuntimeError(f"{name} was not found. Install it with:  {install}")
    return path


def _run(cmd: list[str], *, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(cmd, input=stdin, capture_output=True, check=False, **NO_WINDOW)
    if result.returncode != 0:
        tail = result.stderr.decode(errors="replace")[-2000:]
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({result.returncode}):\n{tail}")
    return result.stdout


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: float
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    """Read size, frame count, frame rate and whether there is an audio track."""
    out = _run([_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)])
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"No video stream in {path}")
    num, den = video.get("r_frame_rate", "24/1").split("/")
    fps = float(num) / float(den or 1)
    frames = int(video.get("nb_frames") or 0)
    if frames == 0:
        counted = _run(
            [_bin("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)]
        )
        frames = int(counted.decode().strip() or 0)
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        frames=frames,
        fps=fps,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def read_frames(path: str | Path, width: int, height: int) -> np.ndarray:
    """Decode every frame of ``path`` as RGB, scaled to ``width x height`` if it is not already."""
    raw = _run(
        [_bin("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
         "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy()


def read_audio(path: str | Path) -> np.ndarray:
    """Decode the first audio track as stereo float32 at :data:`SAMPLE_RATE` (empty if there is none)."""
    if not probe(path).has_audio:
        return np.zeros((0, CHANNELS), dtype=np.float32)
    raw = _run(
        [_bin("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", str(CHANNELS),
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    )
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, CHANNELS).copy()


def samples_for_frames(frames: int, fps: int) -> int:
    """Audio samples that cover ``frames`` video frames (exact for 24/25/30 fps at 48 kHz)."""
    return frames * SAMPLE_RATE // fps


def fit_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    """Trim, or pad with silence, to exactly ``samples`` samples."""
    if len(audio) >= samples:
        return audio[:samples]
    pad = np.zeros((samples - len(audio), CHANNELS), dtype=np.float32)
    return np.concatenate([audio, pad])


def write_wav(path: str | Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32767.0).reshape(-1, CHANNELS)


def write_video(path: str | Path, frames: Iterable[np.ndarray] | np.ndarray, fps: int, *, crf: int = CHUNK_CRF,
                audio: np.ndarray | None = None) -> None:
    """Encode RGB frames to an H.264 mp4, optionally muxing ``audio`` as AAC."""
    frames = np.asarray(frames)
    height, width = frames.shape[1:3]
    cmd = [_bin("ffmpeg"), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
    # Every chunk shares these settings so the concat demuxer can join them without re-encoding.
    codec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", str(fps)]
    wav_path = None
    if audio is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        write_wav(wav_path, audio)
        cmd += ["-i", wav_path, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "256k"]
    cmd += codec + ["-movflags", "+faststart", str(path)]
    try:
        _run(cmd, stdin=np.ascontiguousarray(frames).tobytes())
    finally:
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)


def normalize_clip(src: str | Path, dst: str | Path, width: int, height: int, fps: int) -> None:
    """Scale + center-crop a user video to the project size and frame rate, with a stereo 48 kHz track."""
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps},setsar=1"
    has_audio = probe(src).has_audio
    cmd = [_bin("ffmpeg"), "-v", "error", "-y", "-i", str(src)]
    if not has_audio:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo"]
    cmd += ["-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0", "-vf", vf,
            "-c:v", "libx264", "-crf", str(TAIL_CRF), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-shortest", str(dst)]
    _run(cmd)


def assemble(video_chunks: list[Path], audio_chunks: list[Path], out_path: str | Path) -> None:
    """Join chunk mp4s (stream copy, no quality loss) and their wavs into one mp4 with AAC audio."""
    out_path = Path(out_path)
    work = out_path.parent
    list_file = work / f".{out_path.stem}_concat.txt"
    wav_file = work / f".{out_path.stem}_audio.wav"
    # concat-demuxer quoting: a ' inside the path is written as '\''
    quoted = (p.resolve().as_posix().replace("'", "'\\''") for p in video_chunks)
    list_file.write_text("".join(f"file '{q}'\n" for q in quoted))
    with wave.open(str(wav_file), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        for chunk in audio_chunks:
            with wave.open(str(chunk), "rb") as r:
                w.writeframes(r.readframes(r.getnframes()))
    tmp_out = work / f".{out_path.stem}_tmp.mp4"
    try:
        _run(
            [_bin("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-i", str(wav_file),
             "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
             str(tmp_out)]
        )
        tmp_out.replace(out_path)
    finally:
        list_file.unlink(missing_ok=True)
        wav_file.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)
