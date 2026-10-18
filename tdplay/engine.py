"""TDPlay engine: chains LTX-2 segments into one continuous shot.

Segment 1 is generated from its prompt (and the character photo, when there is one). Every later
segment is an ``ltx-2-mlx extend --distilled`` run on the last 25 frames of the shot so far: the
model gets those frames *and their sound* pinned at the start of its window and continues them.
This is the "motion context" idea behind RunningHub's MiniMax H3 workflow, on open weights.

Each model run happens in its own process. macOS kills a process that runs out of memory, and this
way that costs one segment, not the app; the project can be resumed from where it stopped.

The shot is stored as numbered chunks (H.264 video + WAV audio) plus a 25-frame "carry" tail that
the next segment continues from. ``project.json`` records how far the run got, so an interrupted
run resumes at the next segment.
"""

from __future__ import annotations

import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from tdplay import media

#: Pixel frames ``extend --distilled`` pins from the source (ltx-2-mlx ``EXTEND_CARRY_PIXEL_FRAMES``).
CARRY_FRAMES = 25
FPS = 24
#: Expected denoising steps per segment on the distilled path (8 at half resolution + 3 refine).
DISTILLED_STEPS = 11
SEAM_AUDIO_SECONDS = 0.05
#: Largest per-channel color correction applied at a seam (0-255 scale).
MAX_COLOR_SHIFT = 8.0
MIN_SEGMENT_SECONDS = 1
MAX_SEGMENT_SECONDS = 10
#: The app folder. Helper processes run inside the project folder, so ``-m tdplay.*`` needs it on PYTHONPATH.
APP_ROOT = Path(__file__).resolve().parent.parent


def first_segment_frames(seconds: int, fps: int = FPS) -> int:
    """Frame count for a generated (not extended) segment: on the 8k+1 grid, at least 2 s."""
    latent = max(2 * fps // 8, round(seconds * fps / 8))
    return 8 * latent + 1


def extend_latent_frames(seconds: int, fps: int = FPS) -> int:
    """Latent frames to append for an extended segment (each one is 8 pixel frames)."""
    return max(1, round(seconds * fps / 8))


# -- seams: joining a model output to the frames it continued from ------------------------------------------


def color_lock_luts(original: np.ndarray, redecoded: np.ndarray) -> list[np.ndarray]:
    """Per-channel lookup tables that cancel the color bias of a VAE round trip.

    ``original`` and ``redecoded`` show the same frames, so the gap between their means is the bias.
    """
    delta = original.reshape(-1, 3).mean(axis=0) - redecoded.reshape(-1, 3).mean(axis=0)
    delta = np.clip(delta, -MAX_COLOR_SHIFT, MAX_COLOR_SHIFT)
    return [np.clip(np.round(np.arange(256) + d), 0, 255).astype(np.uint8) for d in delta]


def apply_luts(frames: np.ndarray, luts: list[np.ndarray]) -> None:
    """Apply :func:`color_lock_luts` to ``frames`` in place."""
    for frame in frames:
        for c in range(3):
            frame[..., c] = luts[c][frame[..., c]]


def splice(carry: tuple[np.ndarray, np.ndarray], out: np.ndarray, out_audio: np.ndarray, *, fps: int,
           seam_frames: int, luts: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Splice a model output onto the carry tail it continued from.

    The output starts with the model's own re-decode of the carried frames; the new frames follow
    it. The carry's last ``seam_frames`` frames are cross-faded into that re-decode, and its audio is
    cross-faded into the re-decoded audio, so the cut is invisible and silent. ``luts`` (from
    :func:`color_lock_luts`) are applied to the output from the cross-fade on.
    """
    carry_frames, carry_audio = carry[0].copy(), carry[1].copy()
    t = len(carry_frames)
    k = min(seam_frames, t)
    if luts is not None:
        apply_luts(out[t - k:], luts)
    for j in range(k):
        a = (j + 1) / (k + 1)
        idx = t - k + j
        mixed = (1 - a) * carry_frames[idx].astype(np.float32) + a * out[idx].astype(np.float32)
        carry_frames[idx] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)

    new_frames = out[t:]
    cut = media.samples_for_frames(t, fps)
    new_audio = media.fit_audio(out_audio[cut:], media.samples_for_frames(len(new_frames), fps))
    xf = min(int(SEAM_AUDIO_SECONDS * media.SAMPLE_RATE), cut, len(carry_audio), len(out_audio))
    if xf > 0 and len(out_audio) >= cut:
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
        carry_audio[-xf:] = carry_audio[-xf:] * (1 - ramp) + out_audio[cut - xf:cut] * ramp

    return np.concatenate([carry_frames, new_frames]), np.concatenate([carry_audio, new_audio])


@dataclass
class Segment:
    prompt: str
    seconds: int


@dataclass
class Settings:
    width: int
    height: int
    model_dir: str
    gemma_dir: str | None = None
    low_ram: bool = True
    fast_math: bool = True
    seed: int = -1
    #: Strength of the character photo re-applied at the end of every extended segment (0 = off).
    anchor_strength: float = 0.0
    #: torch backend: how closely the first frame follows the photo (under 1 lets the model repaint it a little).
    image_strength: float = 1.0
    #: Frames over which the end of the previous segment is blended into the model's re-decode.
    seam_frames: int = 8
    #: Cancel the small color shift the VAE round trip adds at every seam.
    color_lock: bool = True
    #: Relax the macOS 26 GPU watchdog for the model process (``AGX_RELAX_CDM_CTXSTORE_TIMEOUT``).
    gpu_watchdog_fix: bool = False
    #: Upscale the finished shot: "off", "1080p" or "4k" (see ``tdplay.upscale``).
    upscale: str = "off"
    #: Upscaler model: "fast" or "best".
    upscale_quality: str = "fast"
    #: "mlx" (Mac, segment by segment) or "torch" (Windows/Linux, whole shot in one LTX-2 run).
    backend: str = "mlx"
    #: torch backend: LTX-2's latent upscaler checkpoint, and where weights live while rendering
    #: ("disk": stream from disk, ~5 GB RAM; "cpu": keep in RAM, ~36 GB RAM).
    upsampler: str | None = None
    offload: str = "disk"
    #: torch backend: how much of the graphics card the model may use ("balanced" or "max").
    gpu_memory: str = "balanced"
    fps: int = FPS


@dataclass
class Project:
    dir: Path
    title: str
    settings: Settings
    segments: list[Segment]
    seeds: list[int]
    reference_image: str | None = None
    start_video: str | None = None
    #: Who is in the shot and what they wear, one line per character: added to every segment's prompt.
    characters: str = ""
    #: Segments finished so far.
    done: int = 0
    #: Chunk files that make up the shot so far (the carry tail is not one of them).
    chunks: int = 0
    #: Whether the start video (if any) has been turned into chunks yet.
    started: bool = False
    status: str = "ready"
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # -- persistence ---------------------------------------------------------

    def save(self) -> None:
        data = asdict(self)
        data.pop("dir")
        tmp = self.dir / "project.json.tmp"
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.dir / "project.json")

    @classmethod
    def load(cls, directory: str | Path) -> Project:
        directory = Path(directory)
        data = json.loads((directory / "project.json").read_text(encoding="utf-8"))
        data["settings"] = Settings(**data["settings"])
        data["segments"] = [Segment(**s) for s in data["segments"]]
        return cls(dir=directory, **data)

    # -- layout ----------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    def prompt(self, i: int) -> str:
        """Segment ``i``'s prompt for the model: the user's text, then the character notes.

        LTX-2 sees a character photo only as the first frame; after that it knows only the prompt and
        the end of the previous segment. Repeating who wears what in every segment keeps the outfits.
        """
        text = self.segments[i].prompt.strip()
        notes = [line.strip() for line in self.characters.splitlines() if line.strip()]
        if not notes:
            return text
        return " ".join(part if part.endswith((".", "!", "?", '"')) else part + "." for part in [text, *notes])

    def chunk(self, index: int, ext: str) -> Path:
        return self.path("chunks", f"chunk_{index:03d}.{ext}")

    def carry(self, step: int, kind: str) -> Path:
        """Carry files after ``step`` finished steps: ``av.mp4`` (model input), ``v.mp4`` and ``wav``."""
        return self.path("carry", f"carry_{step:03d}_{kind}")

    @property
    def carry_step(self) -> int:
        """How many times a carry has been written: once for the start video, once per segment."""
        return self.done + (1 if self.started else 0)

    @property
    def total_seconds(self) -> float:
        fps = self.settings.fps
        total = 0.0
        for i, seg in enumerate(self.segments):
            if i == 0 and not self.start_video:
                total += first_segment_frames(seg.seconds, fps) / fps
            else:
                total += 8 * extend_latent_frames(seg.seconds, fps) / fps
        return total

    @property
    def output(self) -> Path:
        return self.path("render.mp4")

    @property
    def preview(self) -> Path:
        return self.path("preview.mp4")

    def upscaled(self, size: str) -> Path:
        return self.path(f"render_{size}.mp4")


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "shot"


def create_project(
    root: str | Path,
    settings: Settings,
    segments: list[Segment],
    *,
    reference_image: str | None = None,
    start_video: str | None = None,
    characters: str = "",
) -> Project:
    """Make a new project folder under ``root`` (copies the inputs in, so the project is self-contained)."""
    if not segments:
        raise ValueError("Add at least one segment prompt.")
    for i, seg in enumerate(segments, 1):
        if not seg.prompt.strip():
            raise ValueError(f"Segment {i} has no prompt.")
        if not MIN_SEGMENT_SECONDS <= seg.seconds <= MAX_SEGMENT_SECONDS:
            raise ValueError(f"Segment {i}: length must be {MIN_SEGMENT_SECONDS}-{MAX_SEGMENT_SECONDS} s.")
    if segments[0].seconds < 2 and not start_video:
        raise ValueError("The first segment must be at least 2 s (the next one continues its last second).")
    if settings.width % 64 or settings.height % 64:
        raise ValueError("Width and height must be multiples of 64.")

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    title = segments[0].prompt.strip().split("\n")[0][:80]
    directory = Path(root) / f"{stamp}_{_slug(title)}"
    for sub in ("segments", "chunks", "carry"):
        (directory / sub).mkdir(parents=True, exist_ok=True)

    base = settings.seed if settings.seed >= 0 else random.randrange(2**31 - 1000)
    settings.seed = base
    project = Project(
        dir=directory,
        title=title,
        settings=settings,
        segments=segments,
        seeds=[base + i for i in range(len(segments))],
        characters=characters.strip(),
    )
    if reference_image:
        from PIL import Image, ImageOps

        with Image.open(reference_image) as img:
            ImageOps.exif_transpose(img).convert("RGB").save(directory / "reference.png")
        project.reference_image = "reference.png"
    if start_video:
        media.normalize_clip(start_video, directory / "start.mp4", settings.width, settings.height, settings.fps)
        if media.probe(directory / "start.mp4").frames < CARRY_FRAMES:
            raise ValueError(f"The start video needs at least {CARRY_FRAMES} frames (about 1 s).")
        project.start_video = "start.mp4"
    project.save()
    return project


@dataclass
class Event:
    """Something the UI should show: ``log`` line, ``progress``, a ``preview`` file, ``done``, ``error``, ``stopped``."""

    kind: str
    text: str = ""
    fraction: float = 0.0
    path: str | None = None
    #: When it happened (``time.time()``); set when the event is written to a job's ``events.jsonl``.
    at: float = 0.0


class _Stopped(Exception):
    pass


class _SegmentFailed(Exception):
    pass


_TQDM = re.compile(r"(\d+)/(\d+) \[")
_OUT_OF_MEMORY = re.compile(r"out of memory|OutOfMemoryError|HIP error: out|insufficient memory", re.IGNORECASE)


class _StepCounter:
    """Turns tqdm denoising bars (8 half-resolution + 3 refine steps per segment) into progress."""

    def __init__(self) -> None:
        self.finished, self.total, self.n = 0, 0, 0

    def reset(self) -> None:
        self.finished, self.total, self.n = 0, 0, 0

    def update(self, line: str) -> int | None:
        """Steps done in the current segment, or ``None`` if ``line`` is not a tqdm update."""
        if not (m := _TQDM.search(line)):
            return None
        step, total = int(m.group(1)), int(m.group(2))
        if total != self.total or step < self.n:  # a new bar started
            self.finished += self.total if self.n == self.total else self.n
            self.total = total
        self.n = step
        return min(DISTILLED_STEPS, self.finished + step)


def default_ltx_command() -> list[str]:
    override = os.environ.get("TDPLAY_LTX_CMD")
    return shlex.split(override) if override else [sys.executable, "-m", "ltx_pipelines_mlx"]


def default_upscale_command() -> list[str]:
    override = os.environ.get("TDPLAY_UPSCALE_CMD")
    return shlex.split(override) if override else [sys.executable, "-m", "tdplay.upscale"]


_UPSCALE_PROGRESS = re.compile(r"PROGRESS (\d+)/(\d+)")


class Runner:
    """Runs (or resumes) one project. ``run()`` yields :class:`Event` objects; ``cancel()`` stops it."""

    def __init__(self, project: Project, ltx_command: list[str] | None = None,
                 upscale_command: list[str] | None = None):
        self.project = project
        self.ltx_command = ltx_command or default_ltx_command()
        self.upscale_command = upscale_command or default_upscale_command()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None

    # -- public ------------------------------------------------------------------

    def cancel(self) -> None:
        self._stop.set()
        self._kill()

    def run(self) -> Iterator[Event]:
        p = self.project
        p.status = "running"
        p.save()
        try:
            final = yield from self._render()
            seconds = media.probe(final).frames / p.settings.fps
            if p.settings.upscale != "off":
                yield Event("preview", f"Shot finished ({seconds:.1f} s). Upscaling…", fraction=1.0, path=str(final))
                final = yield from self._upscale(final, p.settings.upscale, p.settings.upscale_quality)
            p.status = "complete"
            p.save()
            yield Event("done", f"Finished: {seconds:.1f} s in one continuous shot.", fraction=1.0, path=str(final))
        except _Stopped:
            p.status = "stopped"
            p.save()
            yield Event("stopped", self._stopped_text())
        except GeneratorExit:
            # Whoever was reading the events stopped (closed the stream): don't leave the model running.
            self._stop.set()
            self._kill()
            p.status = "stopped"
            p.save()
            raise
        except _SegmentFailed as exc:
            p.status = "failed"
            p.save()
            yield Event("error", str(exc))
        except Exception as exc:
            p.status = "failed"
            p.save()
            yield Event("error", f"{type(exc).__name__}: {exc}")

    def _stopped_text(self) -> str:
        return f"Stopped after {self.project.done} of {len(self.project.segments)} segments. Resume any time."

    def _render(self):
        """Render the remaining segments (Mac: one ltx-2-mlx run per segment); return the shot's path."""
        p = self.project
        n = len(p.segments)
        carry = self._load_carry()
        if p.start_video and not p.started:
            yield Event("log", "Preparing the start video…")
            frames = media.read_frames(p.path(p.start_video), p.settings.width, p.settings.height)
            audio = media.fit_audio(media.read_audio(p.path(p.start_video)), self._samples(len(frames)))
            carry = self._commit(frames, audio, finished_segment=False)
            yield Event("preview", "Start video loaded.", path=str(self._assemble(p.preview)))

        for i in range(p.done, n):
            started = time.time()
            yield Event("progress", f"Segment {i + 1} of {n}", fraction=i / n)
            raw = p.path("segments", f"seg_{i + 1:02d}.mp4")
            if i == 0 and not p.start_video:
                yield from self._cli(self._generate_args(i, raw), i, n)
                frames = media.read_frames(raw, p.settings.width, p.settings.height)
                audio = media.fit_audio(media.read_audio(raw), self._samples(len(frames)))
            else:
                yield from self._cli(self._extend_args(i, raw, p.carry(p.carry_step, "av.mp4")), i, n)
                frames, audio = self._join(carry, raw)
            carry = self._commit(frames, audio)
            minutes = (time.time() - started) / 60
            yield Event("log", f"Segment {i + 1} done in {minutes:.1f} min.")
            yield Event("preview", f"Segment {i + 1} of {n} done.", fraction=(i + 1) / n,
                        path=str(self._assemble(p.preview)))

        return self._assemble(p.output)

    def upscale(self, size: str, quality: str = "fast") -> Iterator[Event]:
        """Upscale an already finished project (Projects tab)."""
        p = self.project
        if not p.output.exists():
            yield Event("error", "This project isn't finished yet. Resume it first, then upscale.")
            return
        try:
            out = yield from self._upscale(p.output, size, quality)
            yield Event("done", f"Upscaled to {size.upper()}.", fraction=1.0, path=str(out))
        except _Stopped:
            yield Event("stopped", "Upscaling stopped. Start it again any time.")
        except _SegmentFailed as exc:
            yield Event("error", str(exc))
        except Exception as exc:
            yield Event("error", f"{type(exc).__name__}: {exc}")

    def _upscale(self, src: Path, size: str, quality: str):
        from tdplay import upscale

        s = self.project.settings
        width, height = upscale.target_size(size, s.width, s.height)
        dst = self.project.upscaled(size)
        label = f"Upscaling to {size.upper()} ({width}×{height})"
        yield Event("progress", label + "…", fraction=0.0)
        cmd = self.upscale_command + [str(src), str(dst), "--width", str(width), "--height", str(height),
                                      "--quality", quality]
        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1")
        started = time.time()
        for line in self._process(cmd, env, "Upscaling"):
            if m := _UPSCALE_PROGRESS.match(line):
                done, total = int(m.group(1)), int(m.group(2))
                left = (time.time() - started) / done * (total - done)
                eta = f" · about {left / 60:.0f} min left" if left >= 90 else ""
                yield Event("progress", f"{label} · frame {done}/{total}{eta}", fraction=done / total)
            else:
                yield Event("log", line)
        return dst

    # -- model runs ----------------------------------------------------------------

    def _common_args(self, i: int, raw: Path) -> list[str]:
        s = self.project.settings
        # "--prompt=..." so a prompt that starts with "-" is not taken for an option.
        args = ["--distilled", f"--prompt={self.project.prompt(i)}", "--output", str(raw),
                "--model", s.model_dir, "--seed", str(self.project.seeds[i])]
        if s.gemma_dir:
            args += ["--gemma", s.gemma_dir]
        if s.low_ram:
            args.append("--low-ram")
        return args

    def _generate_args(self, i: int, raw: Path) -> list[str]:
        p, s = self.project, self.project.settings
        args = ["generate"] + self._common_args(i, raw)
        args += ["--height", str(s.height), "--width", str(s.width),
                 "--frames", str(first_segment_frames(p.segments[i].seconds, s.fps)), "--frame-rate", str(s.fps)]
        if p.reference_image:
            args += ["--image", str(p.path(p.reference_image))]
        return args

    def _extend_args(self, i: int, raw: Path, carry_av: Path) -> list[str]:
        p, s = self.project, self.project.settings
        args = ["extend"] + self._common_args(i, raw)
        args += ["--video", str(carry_av), "--extend-frames", str(extend_latent_frames(p.segments[i].seconds, s.fps))]
        if p.reference_image and s.anchor_strength > 0:
            args += ["--image", str(p.path(p.reference_image)), "last", f"{s.anchor_strength:.2f}"]
        return args

    def _cli(self, args: list[str], i: int, n: int) -> Iterator[Event]:
        p = self.project
        env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        if p.settings.fast_math:
            env["LTX2_COMPUTE_DTYPE"] = "float16"
        if p.settings.gpu_watchdog_fix:
            env["AGX_RELAX_CDM_CTXSTORE_TIMEOUT"] = "1"
        steps = _StepCounter()
        for line in self._process(self.ltx_command + args, env, f"Segment {i + 1}"):
            if (done := steps.update(line)) is not None:
                within = 0.05 + 0.85 * done / DISTILLED_STEPS
                yield Event("progress", f"Segment {i + 1} of {n} · denoising step {done}/{DISTILLED_STEPS}",
                            fraction=(i + within) / n)
            elif line.startswith("[Decoding"):
                yield Event("progress", f"Segment {i + 1} of {n} · decoding video + sound", fraction=(i + 0.93) / n)
            else:
                yield Event("log", line)

    def _process(self, cmd: list[str], env: dict, what: str) -> Iterator[str]:
        """Run ``cmd`` in its own process group, logging it, and yield its output line by line.

        Raises ``_Stopped`` after :meth:`cancel` and ``_SegmentFailed`` (with a hint) on a non-zero exit.
        """
        if self._stop.is_set():
            raise _Stopped
        tail: list[str] = []
        env = dict(env, PYTHONPATH=os.pathsep.join(filter(None, [str(APP_ROOT), env.get("PYTHONPATH")])),
                   # UTF-8 for the helper's output and files, not Windows' legacy code page: progress bars and
                   # prompts aren't in it.
                   PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        # AMD's MIOpen (convolutions on Radeon cards): only errors in the log, not its harmless workspace
        # warnings, and no benchmarking of every convolution method the first time a size comes up, which
        # takes long and was running when a PC's display driver crashed. Settings already in the environment win.
        env = {"MIOPEN_LOG_LEVEL": "3", "MIOPEN_FIND_MODE": "FAST", **env}
        with self.project.path("log.txt").open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n$ {shlex.join(cmd)}\n")
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                          cwd=self.project.dir, start_new_session=os.name != "nt", **media.NO_WINDOW)
            # tqdm redraws with \r; split on both so every update becomes a line.
            fd = self._proc.stdout.fileno()
            buffer = b""
            while chunk := os.read(fd, 65536):
                buffer += chunk
                *lines, buffer = re.split(rb"[\r\n]", buffer)
                for raw_line in lines:
                    line = raw_line.decode(errors="replace").strip()
                    if line:
                        log.write(line + "\n")
                        tail = (tail + [line])[-15:]
                        yield line
                log.flush()
            code = self._proc.wait()
        self._proc = None
        if self._stop.is_set():
            raise _Stopped
        if code != 0:
            hint = ""
            if any("watchdog" in line.lower() or "ImpactingInteractivity" in line for line in tail):
                hint = ("\n\nThe macOS GPU watchdog stopped the render. Turn on \"GPU watchdog workaround\" in "
                        "Settings (and keep the display awake), then press Resume.")
            elif code in (-getattr(signal, "SIGKILL", 9), 137) or any(_OUT_OF_MEMORY.search(line) for line in tail):
                hint = ("\n\nIt ran out of memory. Try a lower resolution or shorter segments, use the low-memory "
                        "setting (Low-RAM mode / stream from disk), close other apps, then press Resume.")
            raise _SegmentFailed(f"{what} failed (exit code {code}).{hint}\n\n" + "\n".join(tail))

    def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if os.name == "nt":  # no process groups: end the whole tree (the model may have started ffmpeg)
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, **media.NO_WINDOW)
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    # -- stitching -------------------------------------------------------------------

    def _samples(self, frames: int) -> int:
        return media.samples_for_frames(frames, self.project.settings.fps)

    def _load_carry(self) -> tuple[np.ndarray, np.ndarray] | None:
        p = self.project
        if p.carry_step == 0:
            return None
        s = p.settings
        frames = media.read_frames(p.carry(p.carry_step, "v.mp4"), s.width, s.height)
        audio = media.read_wav(p.carry(p.carry_step, "wav"))
        return frames, audio

    def _join(self, carry: tuple[np.ndarray, np.ndarray], raw: Path) -> tuple[np.ndarray, np.ndarray]:
        """Splice an extend output onto the carry tail it continued from (see :func:`splice`)."""
        s = self.project.settings
        t = len(carry[0])
        out = media.read_frames(raw, s.width, s.height)
        if len(out) <= t:
            raise _SegmentFailed(f"The model returned {len(out)} frames, expected more than {t}.")
        luts = color_lock_luts(carry[0], out[:t]) if s.color_lock else None
        return splice(carry, out, media.read_audio(raw), fps=s.fps, seam_frames=s.seam_frames, luts=luts)

    def _commit(self, frames: np.ndarray, audio: np.ndarray, *, finished_segment: bool = True) -> tuple:
        """Write everything but the last 25 frames as a chunk, keep those as the new carry, save state.

        Files are written first and ``project.json`` last, so a crash part-way leaves the previous
        state (and its carry files) intact and the segment simply runs again on resume.
        """
        p, fps = self.project, self.project.settings.fps
        keep = max(0, len(frames) - CARRY_FRAMES)
        split = self._samples(keep)
        if keep:
            tmp_v, tmp_a = p.chunk(p.chunks, "tmp.mp4"), p.chunk(p.chunks, "tmp.wav")
            media.write_video(tmp_v, frames[:keep], fps)
            media.write_wav(tmp_a, audio[:split])
            tmp_v.replace(p.chunk(p.chunks, "mp4"))
            tmp_a.replace(p.chunk(p.chunks, "wav"))
        carry_frames, carry_audio = frames[keep:], audio[split:]
        old_step = p.carry_step
        step = old_step + 1
        media.write_video(p.carry(step, "v.mp4"), carry_frames, fps)
        media.write_wav(p.carry(step, "wav"), carry_audio)
        media.write_video(p.carry(step, "av.mp4"), carry_frames, fps, crf=media.TAIL_CRF, audio=carry_audio)
        if keep:
            p.chunks += 1
        if finished_segment:
            p.done += 1
        else:
            p.started = True
        p.save()
        if old_step:
            for kind in ("v.mp4", "wav", "av.mp4"):
                p.carry(old_step, kind).unlink(missing_ok=True)
        return carry_frames, carry_audio

    def _assemble(self, out: Path) -> Path:
        p = self.project
        videos = [p.chunk(i, "mp4") for i in range(p.chunks)] + [p.carry(p.carry_step, "v.mp4")]
        wavs = [p.chunk(i, "wav") for i in range(p.chunks)] + [p.carry(p.carry_step, "wav")]
        media.assemble(videos, wavs, out)
        return out


def list_projects(root: str | Path) -> list[Project]:
    root = Path(root)
    projects = []
    for path in sorted(root.glob("*/project.json"), reverse=True):
        try:
            projects.append(Project.load(path.parent))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return projects
