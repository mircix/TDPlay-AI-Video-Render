"""TDPlay AI Video Render web UI.

Start it with ``./start.command`` (or ``python -m tdplay``); it opens in your browser.
"""

from __future__ import annotations

import html
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import gradio as gr

from tdplay import __version__, engine, models, scene, worker

MAX_SEGMENTS = 20
PROJECTS_DIR = models.HOME / "projects"

# Short side -> long side; every size is a multiple of 64, as the two-stage pipeline needs.
QUALITIES = {"512p": (512, 896), "576p": (576, 1024), "704p": (704, 1280)}
ASPECTS = ["16:9 landscape", "9:16 portrait", "1:1 square"]
OUTPUT_SIZES = [("Original", "off"), ("1080p", "1080p"), ("4K", "4k")]
UPSCALERS = [("Fast", "fast"), ("Best detail (slower)", "best")]

EXAMPLE = [
    ("A young woman in a yellow raincoat walks toward the camera down a rainy neon-lit street at night. "
     "Handheld tracking shot moving backwards in front of her. Rain hits the puddles, traffic hums in the distance.", 4),
    ("She keeps walking at the same pace while the camera keeps tracking backwards. She glances at a glowing "
     "shop window as she passes; colored reflections slide across her face. Rain patters on her hood.", 4),
    ("Still walking, same camera move: she pulls her hood back, looks into the lens and smiles. "
     "A tram bell rings somewhere behind her.", 4),
]

TIPS = """
**How to write segments.** Each segment continues the last second of the one before it, picture *and* sound.
Start every prompt after the first by continuing the action: *same pace, same camera*. Describe sound too
(footsteps, rain, music) — the model makes audio. Put joins mid-movement (mid-walk, mid-pan), never on a cut or a doorway.
"""

SCENE_HELP = """
LTX-2 takes a picture only as the video's first frame, so this builds one: each character is cut out of their
photo and placed on the location. *Left ↔ right* moves a character across, *Feet* sets where they stand (lower is
nearer the camera), *Size* is their height in the frame. Full-body photos work best for people who stand in the
scene. *Name and outfit* goes into the Characters box above, which reminds the model in every segment what everyone
wears. The first build downloads the cut-out model (180 MB).
"""

RAM_GB = models.system_ram_gb()
BACKEND = models.BACKEND
ON_MAC = BACKEND == "mlx"
PRESETS = models.presets_for(BACKEND)
REC = models.recommend(RAM_GB, BACKEND)
IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"
ENGINE_LABEL = "LTX-2 via MLX" if ON_MAC else "LTX-2 via PyTorch (AMD ROCm / NVIDIA CUDA)"
COMPUTER = "Mac" if sys.platform == "darwin" else "PC"

# job: (project folder, worker process) of the last job this app started.
_state: dict = {"job": None, "last_project": None, "demo": None}
BACKGROUND_NOTE = "It keeps going if you close this window."
#: How long after a job ended a newly opened window still shows how it ended.
RECENT_SECONDS = 24 * 3600

THEMES = [("Match system", "system"), ("Light", "light"), ("Dark", "dark")]
# Gradio's dark theme is the "dark" class on <body>. "system" follows the OS, also when it changes later.
APPLY_THEME_JS = """
(mode) => {
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const apply = (m) => document.body.classList.toggle('dark', m === 'dark' || (m === 'system' && media.matches));
  window.__tdplayTheme = mode;
  apply(mode);
  if (!window.__tdplayThemeWatch) {
    window.__tdplayThemeWatch = true;
    media.addEventListener('change', () => apply(window.__tdplayTheme));
  }
  return mode;
}
"""


def _prefs_path() -> Path:
    return models.HOME / "settings.json"


def load_prefs() -> dict:
    try:
        return json.loads(_prefs_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_pref(key: str, value) -> None:
    prefs = load_prefs()
    prefs[key] = value
    _prefs_path().parent.mkdir(parents=True, exist_ok=True)
    _prefs_path().write_text(json.dumps(prefs, indent=2), encoding="utf-8")


def projects_dir() -> Path:
    """Where new projects are saved: the folder chosen in the Projects tab, else ``PROJECTS_DIR``."""
    chosen = load_prefs().get("projects_dir")
    return Path(chosen) if chosen else PROJECTS_DIR


def project_folders() -> list[Path]:
    """Every folder projects were saved in: the current one, the ones chosen before, the default."""
    folders: list[Path] = []
    for folder in [projects_dir(), *map(Path, load_prefs().get("earlier_projects_dirs", [])), PROJECTS_DIR]:
        if folder not in folders:
            folders.append(folder)
    return folders


def theme_pref() -> str:
    mode = load_prefs().get("theme", "system")
    return mode if mode in {v for _, v in THEMES} else "system"


GPU_MEMORY_CHOICES = [("Balanced (80%)", "balanced"), ("Maximum (90%)", "max")]


def gpu_memory_pref() -> str:
    setting = load_prefs().get("gpu_memory", "balanced")
    return setting if setting in {v for _, v in GPU_MEMORY_CHOICES} else "balanced"
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def resolution(quality: str, aspect: str) -> tuple[int, int]:
    short, long = QUALITIES[quality]
    if aspect.startswith("9:16"):
        return short, long
    if aspect.startswith("1:1"):
        return short, short
    return long, short


def model_paths(preset_key: str) -> tuple[str, str | None, str | None]:
    """Model folder, Gemma folder and (PyTorch) upscaler file; ``TDPLAY_MODEL_DIR`` etc. override them."""
    if os.environ.get("TDPLAY_MODEL_DIR"):
        return (os.environ["TDPLAY_MODEL_DIR"], os.environ.get("TDPLAY_GEMMA_DIR") or None,
                os.environ.get("TDPLAY_UPSAMPLER") or None)
    preset = models.PRESETS[preset_key]
    gemma = str(models.local_dir(preset.gemma_repo)) if preset.gemma_repo else None
    upsampler = str(models.local_dir(preset.upsampler[0]) / preset.upsampler[1]) if preset.upsampler else None
    return str(models.local_dir(preset.repo)), gemma, upsampler


def model_ready(preset_key: str) -> bool:
    return bool(os.environ.get("TDPLAY_MODEL_DIR")) or models.preset_ready(models.PRESETS[preset_key])


def segment_seconds(index: int, seconds: int, has_start: bool) -> float:
    if index == 0 and not has_start:
        return engine.first_segment_frames(int(seconds)) / engine.FPS
    return 8 * engine.extend_latent_frames(int(seconds)) / engine.FPS


def total_text(start_video, n, *seconds) -> str:
    n = int(n)
    secs = [int(s) for s in seconds[:n]]
    total = sum(segment_seconds(i, s, bool(start_video)) for i, s in enumerate(secs))
    text = f"**{total:.1f} s** of new video in {n} segment{'s' if n != 1 else ''}"
    if start_video:
        text += ", after your start video"
    if any(s > REC.max_segment_seconds for s in secs):
        text += (f"  \n⚠️ Segments longer than {REC.max_segment_seconds} s may run out of memory on a "
                 f"{RAM_GB} GB Mac. Use more, shorter segments instead.")
    return text


def spread_characters(*pictures):
    """Spread the characters evenly across the frame whenever one is added or removed."""
    present = [i for i, picture in enumerate(pictures) if picture]
    across = {i: round(100 * scene.default_x(k, len(present))) for k, i in enumerate(present)}
    return [gr.update(value=across[i]) if i in across else gr.skip() for i in range(len(pictures))]


_cutter: list = []  # the cut-out model, loaded on the first build


def build_scene(location, *rest):
    """Build the first frame from the location and characters; it replaces the picture above."""
    n = scene.MAX_CHARACTERS
    pictures, across, down, sizes = rest[:n], rest[n:2 * n], rest[2 * n:3 * n], rest[3 * n:4 * n]
    notes, (quality, aspect) = rest[4 * n:5 * n], rest[5 * n:5 * n + 2]
    chosen = [i for i, picture in enumerate(pictures) if picture]
    written = [" ".join(str(notes[i] or "").split()) for i in chosen if str(notes[i] or "").strip()]
    characters = "\n".join(written) if written else gr.skip()  # the notes go into every segment's prompt
    try:
        if chosen and not _cutter:
            if not scene.cutout_model_path().exists():
                yield gr.skip(), gr.skip(), gr.skip(), "Downloading the cut-out model (180 MB, only once)…"
            _cutter.append(scene.Cutter())
        width, height = resolution(quality, aspect)
        placements = [scene.Placement(x=across[i] / 100, y=down[i] / 100, size=sizes[i] / 100) for i in chosen]
        path = scene.build(location or None, [pictures[i] for i in chosen], placements, width, height,
                           models.HOME / "scenes", cutter=_cutter[0] if _cutter else None)
    except ImportError:
        fix = "the repair setup in the Start menu" if os.name == "nt" else "the app's setup again"
        yield (gr.skip(), gr.skip(), gr.skip(),
               f"⚠️ The cut-out model needs ONNX Runtime, which isn't installed: run {fix}.")
        return
    except (ValueError, RuntimeError, OSError) as exc:
        yield gr.skip(), gr.skip(), gr.skip(), f"⚠️ {exc}"
        return
    yield (str(path), gr.skip() if ON_MAC else scene.SCENE_STRENGTH, characters,
           "Built: it's now the first frame above. Say what the characters do in segment 1, "
           "or move them with the sliders and build again.")


def status_html(title: str, fraction: float | None = None, detail: str = "", tone: str = "info") -> str:
    bar = ""
    if fraction is not None:
        pct = max(0.0, min(1.0, fraction)) * 100
        bar = f'<div class="lt-bar"><div style="width:{pct:.1f}%"></div></div>'
    detail_html = f'<div class="lt-detail">{html.escape(detail)}</div>' if detail else ""
    return f'<div class="lt-status lt-{tone}"><b>{html.escape(title)}</b>{bar}{detail_html}</div>'


def log_html(lines) -> str:
    text = "\n".join(lines) or "The log appears here while generating."
    return f'<div class="lt-log"><pre>{html.escape(text)}</pre></div>'


def _eta(started: float, fraction: float) -> str:
    if fraction < 0.03:
        return ""
    left = (time.time() - started) / fraction * (1 - fraction)
    return f"about {left / 60:.0f} min left" if left >= 90 else "less than 2 min left"


def open_folder(path: str | None = None) -> None:
    """Show a project's folder in Finder / File Explorer: ``path``, else the latest project, else all of them."""
    path = path or _state["last_project"] or latest_project()
    if not path or not Path(path).exists():
        projects_dir().mkdir(parents=True, exist_ok=True)
        path = str(projects_dir())
    if os.name == "nt":
        os.startfile(path)  # Windows has no "open" command
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def active_job() -> str | None:
    """The project a background job is working on right now (renders run in their own process)."""
    job = _state["job"]
    if job is not None and job[1].poll() is None:  # started by this app and still running (or starting)
        return job[0]
    for folder in project_folders():  # started earlier, e.g. before the app was reopened
        for lock in folder.glob(f"*/{worker.LOCK}"):
            if worker.is_running(lock.parent):
                return str(lock.parent)
    return None


def _start(project: engine.Project, job: list[str], resume_hint: bool = True):
    """Start ``job`` on ``project`` in the background and follow it."""
    with _lock:
        busy = active_job()
        if busy:
            yield (status_html("Already working on another video — wait for it to finish or press Stop.",
                               tone="warn", detail=Path(busy).name), gr.skip(), gr.skip())
            return
        if not ON_MAC:  # the card's memory setting applies to whatever runs next, a resumed render too
            project.settings.gpu_memory = gpu_memory_pref()
            project.save()
        proc = worker.start(project.dir, job)
        _state["job"], _state["last_project"] = (str(project.dir), proc), str(project.dir)
    yield from _watch(str(project.dir), proc, resume_hint)


def _watch(project_dir: str, proc=None, resume_hint: bool = True):
    """Follow a background job and turn its events into (status, video, log) updates."""
    log: deque[str] = deque(maxlen=400)
    log.append(f"Project folder: {project_dir}")
    started = time.time()
    title, fraction, eta, video = "Starting…", 0.0, "", gr.skip()
    changed, last_push = True, 0.0
    for event in worker.follow(project_dir, proc):
        if event is None:
            pass
        elif event.kind == "queued":
            started = event.at or started
        elif event.kind == "log":
            log.append(event.text)
        elif event.kind == "progress":
            title, fraction = event.text, event.fraction
            eta = _eta(started, fraction)
        elif event.kind == "preview":
            log.append(event.text)
            fraction = event.fraction or fraction
            video = event.path
        elif event.kind == "done":
            log.append(event.text)
            yield (status_html("Done! " + event.text, 1.0, f"Saved to {event.path}", tone="ok"),
                   event.path, log_html(log))
            return
        elif event.kind in ("error", "stopped"):
            log.append(event.text)
            tone = "warn" if event.kind == "stopped" else "error"
            head = "Stopped." if event.kind == "stopped" else "Something went wrong."
            detail = event.text.split("\n\n")[0] + ("  Press Resume to continue." if resume_hint else "")
            yield status_html(head, fraction, detail, tone), video, log_html(log)
            return
        changed = changed or event is not None
        # tqdm lines arrive many times a second; refresh the page at most twice a second.
        now = time.time()
        if changed and now - last_push >= 0.5:
            detail = (eta + " · " if eta else "") + BACKGROUND_NOTE
            yield status_html(title, fraction, detail), video, log_html(log)
            video, changed, last_push = gr.skip(), False, now


def generate(image, start_video, characters, n, *rest):
    prompts, seconds = rest[:MAX_SEGMENTS], rest[MAX_SEGMENTS : 2 * MAX_SEGMENTS]
    (preset_key, quality, aspect, low_ram, fast_math, watchdog, anchor, seam, color_lock, seed, out_size,
     up_quality, image_strength) = rest[2 * MAX_SEGMENTS :]
    n = int(n)
    if ON_MAC and not IS_APPLE_SILICON and not os.environ.get("TDPLAY_LTX_CMD"):
        yield (status_html("TDPlay AI Video Render needs a Mac with Apple Silicon (M1 or newer) to generate.",
                           tone="error"), None, gr.skip())
        return
    if not model_ready(preset_key):
        yield (status_html("Download the model first: open the Models tab and press Download.", tone="warn"),
               None, gr.skip())
        return
    if active_job():
        yield (status_html("Already working on another video — wait for it to finish or press Stop.", tone="warn"),
               gr.skip(), gr.skip())
        return
    width, height = resolution(quality, aspect)
    model_dir, gemma_dir, upsampler = model_paths(preset_key)
    settings = engine.Settings(
        width=width, height=height, model_dir=model_dir, gemma_dir=gemma_dir, low_ram=bool(low_ram),
        backend=BACKEND, upsampler=upsampler, offload="disk" if low_ram else "cpu",
        fast_math=bool(fast_math), seed=int(seed if seed is not None else -1), anchor_strength=float(anchor),
        seam_frames=int(seam), color_lock=bool(color_lock), gpu_watchdog_fix=bool(watchdog),
        upscale=out_size or "off", upscale_quality=up_quality or "fast", image_strength=float(image_strength),
    )
    segments = [engine.Segment(str(p or ""), int(s)) for p, s in zip(prompts[:n], seconds[:n])]
    try:
        project = engine.create_project(projects_dir(), settings, segments, reference_image=image or None,
                                        start_video=start_video or None, characters=characters or "")
    except OSError as exc:  # the projects folder is on a drive that's gone, say
        yield status_html(f"Can't save the project in {projects_dir()}: {exc.strerror or exc}. Choose another "
                          "folder in the Projects tab.", tone="error"), None, gr.skip()
        return
    except (ValueError, RuntimeError) as exc:
        yield status_html(str(exc), tone="error"), None, gr.skip()
        return
    yield from _start(project, ["render"])


def all_projects() -> list[engine.Project]:
    """The projects in every projects folder, newest first (their folder names start with the date)."""
    projects = [p for folder in project_folders() for p in engine.list_projects(folder)]
    return sorted(projects, key=lambda p: p.dir.name, reverse=True)


def latest_project() -> str | None:
    projects = all_projects()
    return str(projects[0].dir) if projects else None


def resume(project_dir, low_ram, fast_math, watchdog):
    project_dir = project_dir or _state["last_project"] or latest_project()
    if not project_dir:
        yield status_html("Nothing to resume yet.", tone="warn"), gr.skip(), gr.skip()
        return
    if active_job() == str(project_dir):  # still running in the background: just show it
        yield from _watch(str(project_dir))
        return
    project = engine.Project.load(project_dir)
    if project.status == "complete":
        yield (status_html("That project is already finished.", 1.0, str(project.output), "ok"),
               str(project.output), gr.skip())
        return
    # Memory/GPU settings may change between attempts; everything that shapes the shot stays as it was.
    s = project.settings
    s.low_ram, s.fast_math, s.gpu_watchdog_fix = bool(low_ram), bool(fast_math), bool(watchdog)
    s.offload = "disk" if low_ram else "cpu"
    project.save()
    yield from _start(project, ["render"])


def upscale_project(project_dir, size, quality):
    if not project_dir:
        yield status_html("Choose a project in the Projects tab first.", tone="warn"), gr.skip(), gr.skip()
        return
    project = engine.Project.load(project_dir)
    yield from _start(project, ["upscale", "--size", size, "--quality", quality], resume_hint=False)


def attach():
    """On page load: show the job running in the background, or how the last one ended if that was recent."""
    project_dir = active_job()
    if project_dir:
        _state["last_project"] = project_dir
        job = _state["job"]
        yield from _watch(project_dir, job[1] if job is not None and job[0] == project_dir else None)
        return
    events = sorted((path for folder in project_folders() for path in folder.glob(f"*/{worker.EVENTS}")),
                    key=lambda path: path.stat().st_mtime)
    if events and time.time() - events[-1].stat().st_mtime < RECENT_SECONDS:
        _state["last_project"] = str(events[-1].parent)
        yield from _watch(str(events[-1].parent))
        return
    yield gr.skip(), gr.skip(), gr.skip()


def is_busy() -> bool:
    """True while a model download is running: quitting stops it (renders keep going on their own)."""
    return bool(_state.get("downloading"))


def resume_last(low_ram, fast_math, watchdog):
    yield from resume(None, low_ram, fast_math, watchdog)


def stop():
    project_dir = active_job()
    if project_dir is None:
        return status_html("Nothing is running.", tone="warn")
    worker.request_stop(project_dir)
    return status_html("Stopping…", tone="warn", detail=(
        "The current segment is discarded; finished ones are kept." if ON_MAC else
        "Resume renders the shot again from the start."))


# ---------------------------------------------------------------------------
# projects + models tabs
# ---------------------------------------------------------------------------


def project_choices() -> list[tuple[str, str]]:
    choices = []
    for p in all_projects():
        label = (f"{p.created.replace('T', ' ')[:16]} · {p.title[:60]} · {project_status(p)} "
                 f"({p.done}/{len(p.segments)})")
        choices.append((label, str(p.dir)))
    return choices


def project_status(p: engine.Project) -> str:
    if p.status == "running" and not worker.is_running(p.dir):
        return "interrupted"  # the render process ended without saying so (a crash or a restart)
    return p.status


def refresh_projects():
    choices = project_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def _serve(folder: Path) -> None:
    """Let the window show videos from ``folder`` (Gradio serves files only from folders it was allowed)."""
    demo = _state["demo"]
    if demo is not None and str(folder) not in demo.allowed_paths:
        demo.allowed_paths.append(str(folder))


def use_projects_dir(text):
    """Save new projects in the folder ``text`` (made if it doesn't exist); empty means the default folder."""
    raw = str(text or "").strip().strip('"')
    folder = Path(raw).expanduser() if raw else PROJECTS_DIR
    if not folder.is_absolute():
        example = r"D:\Videos\TDPlay" if os.name == "nt" else "/Users/you/Movies/TDPlay"
        return gr.skip(), f"⚠️ Give the folder's full path, like `{example}`.", gr.skip()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".tdplay-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return gr.skip(), f"⚠️ Can't save projects in `{folder}`: {exc.strerror or exc}", gr.skip()
    old = projects_dir()
    if folder != old:  # projects already made stay where they are: keep listing that folder
        earlier = [d for d in load_prefs().get("earlier_projects_dirs", []) if Path(d) != folder]
        if old != PROJECTS_DIR and str(old) not in earlier:
            earlier.append(str(old))
        save_pref("earlier_projects_dirs", earlier)
    save_pref("projects_dir", "" if folder == PROJECTS_DIR else str(folder))
    _serve(folder)
    return (str(folder), f"New projects are saved in `{folder}`. Projects made before stay where they are and "
                         "are still listed here.", refresh_projects())


def pick_projects_dir(current):
    """The app window's own folder picker; in a browser the path is typed in instead."""
    try:
        import webview

        window = webview.windows[0] if webview.windows else None
    except ImportError:
        window = None
    if window is None:
        return gr.skip(), "Type or paste the folder's path, then press **Use this folder**.", gr.skip()
    chosen = window.create_file_dialog(webview.FileDialog.FOLDER, directory=str(current or projects_dir()))
    if not chosen:  # cancelled
        return gr.skip(), "", gr.skip()
    return use_projects_dir(chosen[0] if isinstance(chosen, (list, tuple)) else chosen)


def show_project(project_dir):
    if not project_dir:
        return None, ""
    p = engine.Project.load(project_dir)
    head = (f"**{p.title}**\n\nStatus: {project_status(p)} · {p.done} of {len(p.segments)} segments · "
            f"{p.settings.width}×{p.settings.height} · seed {p.settings.seed}\n")
    lines = [head] + [f"{i + 1}. ({s.seconds} s) {s.prompt}" for i, s in enumerate(p.segments)]
    if p.characters:
        lines.append("\nCharacters (in every segment): " + "; ".join(p.characters.splitlines()))
    files = [f.name for f in (p.output, p.upscaled("1080p"), p.upscaled("4k")) if f.exists()]
    if files:
        lines.append("\nFiles: " + ", ".join(f"`{f}`" for f in files))
    video = p.output if p.output.exists() else (p.preview if p.preview.exists() else None)
    return (str(video) if video else None), "\n".join(lines)


def preset_status(preset_key):
    if model_ready(preset_key):
        return status_html("Ready.")
    return status_html("This model isn't downloaded yet — open the Models tab and press Download.", tone="warn")


def models_table() -> str:
    rows = ["| Model | Download | Status |", "|---|---|---|"]
    for key, p in PRESETS.items():
        ready = "✅ ready" if models.preset_ready(p) else "not downloaded"
        mark = f" ← recommended for this {COMPUTER}" if key == REC.preset else ""
        rows.append(f"| {p.label}{mark} | {_SIZES.get(key, '?')} | {ready} |")
    free = shutil.disk_usage(models.HOME if models.HOME.exists() else Path.home()).free / 1e9
    rows.append(f"\nFree disk space: **{free:.0f} GB** · models are stored in `{models.MODELS_DIR}`")
    return "\n".join(rows)


_SIZES = {"ltx-2.5-q8": "≈ 45 GB", "ltx-2.3-q8": "≈ 38 GB", "ltx-2.3-q4": "≈ 29 GB", "ltx-2.3-fp8": "≈ 55 GB"}


def download_models(preset_key):
    _state["downloading"] = True
    try:
        yield from _download_models(preset_key)
    finally:
        _state["downloading"] = False


def _download_models(preset_key):
    preset = models.PRESETS[preset_key]
    try:
        needed = models.download_size_gb(preset, missing_only=True)
    except Exception as exc:
        yield status_html("Could not reach Hugging Face.", tone="error", detail=f"{type(exc).__name__}: {exc}"), gr.skip()
        return
    models.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(models.MODELS_DIR).free / 1e9
    if free < needed + 5:
        yield (status_html("Not enough free disk space.", tone="error",
                           detail=f"Needs about {needed:.0f} GB plus room for videos; {free:.0f} GB free."), gr.skip())
        return
    for repo, patterns in models.downloads_for(preset):
        if models.is_downloaded(repo):
            continue
        started = time.time()
        try:
            for progress in models.download(repo, patterns):
                msg = models.wait_message(progress, started)
                yield status_html(f"Downloading {preset.label}", progress.fraction, msg), gr.skip()
        except Exception as exc:  # network errors, disk full, ...
            yield status_html("Download failed — press Download again to continue where it stopped.",
                              tone="error", detail=f"{type(exc).__name__}: {exc}"), models_table()
            return
    yield status_html(f"{preset.label} is ready.", 1.0, tone="ok"), models_table()


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

CSS = """
.lt-status {padding: 12px 14px; border-radius: 10px; border: 1px solid var(--border-color-primary);
            background: var(--block-background-fill);}
.lt-status b {font-size: 1.02em;}
.lt-ok {border-color: #2e9e5b;} .lt-warn {border-color: #d9a400;} .lt-error {border-color: #d64545;}
.lt-bar {height: 8px; border-radius: 4px; background: var(--border-color-primary); margin-top: 8px; overflow: hidden;}
.lt-bar div {height: 100%; background: var(--color-accent); transition: width .4s;}
.lt-detail {margin-top: 6px; opacity: .8; font-size: .92em; white-space: pre-wrap;}
.lt-seg {padding: 6px 10px !important;}
.lt-log {height: 260px; overflow-y: auto; display: flex; flex-direction: column-reverse; padding: 8px 10px;
         border: 1px solid var(--border-color-primary); border-radius: 8px; background: var(--block-background-fill);}
.lt-log pre {margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; font-family: var(--font-mono);}
"""


def build() -> gr.Blocks:
    chip = models.chip_name()
    with gr.Blocks(title="TDPlay AI Video Render", delete_cache=(1800, 3600)) as demo:
        with gr.Row(equal_height=True):
            gr.Markdown(
                f"# 🎬 TDPlay AI Video Render\nOne continuous AI video shot, built segment by segment with "
                f"picture-and-sound continuity — free and private, on your computer.  \n<small>{chip} · {RAM_GB} GB · "
                f"{ENGINE_LABEL} · v{__version__}</small>",
                scale=5,
            )
            appearance = gr.Radio(THEMES, value=theme_pref(), label="Appearance", scale=1, min_width=260)
        with gr.Tabs() as tabs:
            with gr.Tab("Create", id="create"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image = gr.Image(label="Character / first frame (optional)", type="filepath",
                                         sources=["upload", "clipboard"], height=240)
                        characters = gr.Textbox(
                            label="Characters (added to every segment)", lines=2, max_lines=6,
                            placeholder="Mia: black lace slip dress with thin straps, choker, long black hair\n"
                                        "Leo: grey suit, white shirt, short brown hair",
                            info="One line per character: name, clothes, hair. The model sees your photo only as "
                                 "the first frame; these lines remind it in every segment, so outfits stay the same.")
                        with gr.Accordion("…or continue a video you already have", open=False):
                            start_video = gr.Video(label="Start video (any size; it is cropped to the shot size)",
                                                   sources=["upload"], height=200)
                        with gr.Accordion("…or build it from a location and characters", open=False):
                            gr.Markdown(SCENE_HELP)
                            scene_location = gr.Image(label="Location (optional)", type="filepath",
                                                      sources=["upload", "clipboard"], height=180)
                            scene_pictures, scene_across, scene_down, scene_size, scene_notes = [], [], [], [], []
                            with gr.Row():
                                for i in range(scene.MAX_CHARACTERS):
                                    with gr.Column(min_width=150):
                                        scene_pictures.append(gr.Image(
                                            label=f"Character {i + 1}", type="filepath",
                                            sources=["upload", "clipboard"], height=150))
                                        scene_across.append(gr.Slider(
                                            0, 100, value=round(100 * scene.default_x(i, scene.MAX_CHARACTERS)),
                                            step=1, label="Left ↔ right"))
                                        scene_down.append(gr.Slider(20, 100, value=95, step=1,
                                                                    label="Feet: up ↔ down"))
                                        scene_size.append(gr.Slider(10, 100, value=70, step=1, label="Size"))
                                        scene_notes.append(gr.Textbox(
                                            label="Name and outfit", lines=2, max_lines=4,
                                            placeholder="Mia: black lace slip dress, thin straps, choker"))
                            scene_btn = gr.Button("Build the first frame")
                            scene_status = gr.Markdown()
                        gr.Markdown(TIPS)
                        n_segments = gr.Slider(1, MAX_SEGMENTS, value=3, step=1, label="Segments")
                        groups, prompts, seconds = [], [], []
                        for i in range(MAX_SEGMENTS):
                            with gr.Group(visible=i < 3, elem_classes="lt-seg") as group:
                                prompts.append(gr.Textbox(
                                    label=f"Segment {i + 1}" + (" — the shot" if i == 0 else " — continues"),
                                    lines=2, max_lines=6,
                                    placeholder="Describe the action, camera and sound for this part of the shot…"))
                                seconds.append(gr.Slider(engine.MIN_SEGMENT_SECONDS, engine.MAX_SEGMENT_SECONDS,
                                                         value=min(4, REC.max_segment_seconds), step=1,
                                                         label="Seconds"))
                            groups.append(group)
                        total = gr.Markdown(total_text(None, 3, *[4] * 3))
                        example_btn = gr.Button("Load an example", size="sm")
                        with gr.Accordion("Settings", open=False):
                            preset = gr.Dropdown([(p.label, k) for k, p in PRESETS.items()], value=REC.preset,
                                                 label="Model")
                            with gr.Row():
                                quality = gr.Radio(list(QUALITIES), value=REC.quality, label="Resolution")
                                aspect = gr.Radio(ASPECTS, value=ASPECTS[0], label="Shape")
                            low_ram = gr.Checkbox(
                                REC.low_ram, label="Low-RAM mode",
                                info=("Streams the model from disk: ~5% slower, needs far less memory." if ON_MAC else
                                      "Streams the model from disk (~5 GB RAM). Off keeps it in RAM: faster, but "
                                      "needs about 36 GB free."))
                            gpu_memory = gr.Radio(
                                GPU_MEMORY_CHOICES, value=gpu_memory_pref(), label="Graphics card memory",
                                visible=not ON_MAC,
                                info="How much of the card the model may use. Maximum is faster; close games "
                                     "and other GPU-heavy apps while it renders. Back to Balanced if the "
                                     "screen flickers or goes black.")
                            fast_math = gr.Checkbox(True, label="Faster math (float16)", visible=ON_MAC,
                                                    info="About 20% faster. Turn off if you see glitches.")
                            watchdog = gr.Checkbox(False, label="GPU watchdog workaround (macOS 26)", visible=ON_MAC,
                                                   info="Turn on if a render dies with a GPU timeout / watchdog "
                                                        "error. The Mac may feel less responsive while rendering.")
                            anchor = gr.Slider(0, 1, value=0, step=0.05,
                                               label="Keep the character's face: re-anchor the photo every segment",
                                               info="0 = off. 0.5–0.7 stops face drift, but pulls each segment's "
                                                    "last frame toward the photo's pose and framing.")
                            image_strength = gr.Slider(
                                0.5, 1, value=1.0, step=0.05, label="First frame strength", visible=not ON_MAC,
                                info="1 = the video starts exactly on the picture. About 0.85 lets the model repaint "
                                     "it a little, which blends the light of a built scene.")
                            # The PyTorch backend blends the joins inside LTX-2, so these are Mac-only.
                            seam = gr.Slider(0, 16, value=8, step=1, label="Seam blend (frames)", visible=ON_MAC)
                            color_lock = gr.Checkbox(True, label="Color lock at seams", visible=ON_MAC,
                                                     info="Cancels the small color shift each continuation adds.")
                            seed = gr.Number(-1, precision=0, label="Seed (-1 = random)")
                        with gr.Row():
                            out_size = gr.Radio(OUTPUT_SIZES, value="off", label="Final output size",
                                                info="1080p and 4K are made by AI upscaling (Real-ESRGAN) after "
                                                     "the last segment. The original-size video is kept too.")
                            up_quality = gr.Radio(UPSCALERS, value="fast", label="Upscaler")
                        with gr.Row():
                            go = gr.Button("Generate", variant="primary", scale=3)
                            stop_btn = gr.Button("Stop", variant="stop", scale=1)
                            resume_btn = gr.Button("Resume", scale=1)
                    with gr.Column(scale=6):
                        ready = model_ready(REC.preset)
                        status = gr.HTML(status_html(
                            "Ready." if ready else "First, download the model in the Models tab.",
                            tone="info" if ready else "warn",
                            detail="" if IS_APPLE_SILICON or not ON_MAC else "Generation needs a Mac with Apple Silicon."))
                        video = gr.Video(label="Your shot (updates after every segment)", interactive=False,
                                         height=420, loop=False)
                        with gr.Row():
                            folder_btn = gr.Button("Open project folder", size="sm")
                        log = gr.HTML(log_html([]))

            with gr.Tab("Projects", id="projects") as projects_tab:
                with gr.Row():
                    folder_box = gr.Textbox(str(projects_dir()), label="New projects are saved in", scale=5)
                    with gr.Column(scale=1, min_width=140):
                        pick_folder_btn = gr.Button("Choose…")
                        use_folder_btn = gr.Button("Use this folder")
                folder_note = gr.Markdown()
                with gr.Row():
                    project = gr.Dropdown(project_choices(), label="Your projects", scale=5)
                    refresh_btn = gr.Button("Refresh", scale=1)
                with gr.Row():
                    with gr.Column(scale=5):
                        project_info = gr.Markdown()
                        with gr.Row():
                            resume_project_btn = gr.Button("Resume this project", variant="primary")
                            project_folder_btn = gr.Button("Open folder")
                        with gr.Group():
                            with gr.Row():
                                p_size = gr.Radio(OUTPUT_SIZES[1:], value="1080p", label="Upscale to")
                                p_quality = gr.Radio(UPSCALERS, value="fast", label="Upscaler")
                            upscale_btn = gr.Button("Upscale this project")
                    project_video = gr.Video(label="Video", interactive=False, height=360, scale=6)

            with gr.Tab("Models", id="models") as models_tab:
                gr.Markdown(
                    "TDPlay AI Video Render runs **LTX-2** (Lightricks, open weights) through "
                    "[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), natively on Apple Silicon. Download once; "
                    "after that everything runs offline. Only the files the fast *distilled* pipeline needs are "
                    "downloaded (about half of each pack)." if ON_MAC else
                    "TDPlay AI Video Render runs **LTX-2.3** (Lightricks, open weights) with Lightricks' own "
                    "[LTX-2 code](https://github.com/Lightricks/LTX-2) on PyTorch: AMD Radeon through ROCm, or NVIDIA "
                    "through CUDA. Download once; after that everything runs offline. The download is the 8-bit "
                    "distilled model, its 2× upscaler and the Gemma 3 text encoder."
                )
                table = gr.Markdown(models_table())
                with gr.Row():
                    dl_preset = gr.Dropdown([(p.label, k) for k, p in PRESETS.items()], value=REC.preset,
                                            label="Model to download", scale=4)
                    dl_btn = gr.Button("Download", variant="primary", scale=1)
                dl_status = gr.HTML()
                gr.Markdown(
                    f"This {COMPUTER}: **{chip}, {RAM_GB} GB**. Recommended: **{models.PRESETS[REC.preset].label}**, "
                    f"{REC.quality}, segments up to {REC.max_segment_seconds} s"
                    f"{', Low-RAM mode on' if REC.low_ram else ''}.  \nThe weights are under the "
                    "[LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE): free to use, "
                    "but organizations with $10M+ annual revenue need a paid commercial license from Lightricks, "
                    "and it has an acceptable-use policy. Read it before commercial use."
                )

            with gr.Tab("Help", id="help"):
                gr.Markdown(HELP)

        # -- wiring -----------------------------------------------------------------
        def show_segments(n, *secs):
            n = int(n)
            return [gr.update(visible=i < n) for i in range(MAX_SEGMENTS)]

        total_inputs = [start_video, n_segments, *seconds]
        n_segments.change(show_segments, [n_segments], groups, api_visibility="private")
        for comp in [n_segments, start_video, *seconds]:
            comp.change(total_text, total_inputs, total, api_visibility="private")

        def load_example():
            values = [gr.update(value=EXAMPLE[i][0] if i < len(EXAMPLE) else "") for i in range(MAX_SEGMENTS)]
            secs = [gr.update(value=EXAMPLE[i][1] if i < len(EXAMPLE) else 4) for i in range(MAX_SEGMENTS)]
            return [len(EXAMPLE), *values, *secs]

        example_btn.click(load_example, None, [n_segments, *prompts, *seconds], api_visibility="private")

        outputs = [status, video, log]
        # The jobs run in their own processes; these handlers only start and follow them, so any number of
        # windows can follow at once (concurrency_limit=None). Starting is guarded by active_job().
        follow = dict(api_visibility="private", show_progress="hidden", concurrency_limit=None)
        go.click(generate,
                 [image, start_video, characters, n_segments, *prompts, *seconds, preset, quality, aspect, low_ram,
                  fast_math, watchdog, anchor, seam, color_lock, seed, out_size, up_quality, image_strength],
                 outputs, **follow)
        for picture in scene_pictures:
            picture.change(spread_characters, scene_pictures, scene_across, api_visibility="private", queue=False)
        scene_btn.click(build_scene, [scene_location, *scene_pictures, *scene_across, *scene_down, *scene_size,
                                      *scene_notes, quality, aspect],
                        [image, image_strength, characters, scene_status], api_visibility="private")
        stop_btn.click(stop, None, status, api_visibility="private", queue=False)
        resume_btn.click(resume_last, [low_ram, fast_math, watchdog], outputs, **follow)
        demo.load(attach, None, outputs, **follow)
        folder_btn.click(open_folder, None, None, api_visibility="private")

        refresh_btn.click(refresh_projects, None, project, api_visibility="private")
        project.change(show_project, project, [project_video, project_info], api_visibility="private")
        project_folder_btn.click(open_folder, project, None, api_visibility="private")
        resume_project_btn.click(lambda: gr.Tabs(selected="create"), None, tabs, api_visibility="private").then(
            resume, [project, low_ram, fast_math, watchdog], outputs, **follow)

        upscale_btn.click(lambda: gr.Tabs(selected="create"), None, tabs, api_visibility="private").then(
            upscale_project, [project, p_size, p_quality], outputs, **follow)

        dl_btn.click(download_models, dl_preset, [dl_status, table], api_visibility="private",
                     show_progress="hidden")
        preset.change(preset_status, preset, status, api_visibility="private")
        projects_tab.select(refresh_projects, None, project, api_visibility="private")
        projects_tab.select(lambda: str(projects_dir()), None, folder_box, api_visibility="private", queue=False)
        pick_folder_btn.click(pick_projects_dir, folder_box, [folder_box, folder_note, project],
                              api_visibility="private")
        use_folder_btn.click(use_projects_dir, folder_box, [folder_box, folder_note, project],
                             api_visibility="private")
        models_tab.select(models_table, None, table, api_visibility="private")
        demo.load(models_table, None, table, api_visibility="private")
        # The saved choice is read on every page load, so all windows and browsers agree.
        appearance.input(lambda mode: save_pref("theme", mode), appearance, None, js=APPLY_THEME_JS,
                         api_visibility="private", queue=False)
        demo.load(theme_pref, None, appearance, api_visibility="private", queue=False).then(
            None, appearance, None, js=APPLY_THEME_JS)
        gpu_memory.input(lambda setting: save_pref("gpu_memory", setting), gpu_memory, None,
                         api_visibility="private", queue=False)
        demo.load(gpu_memory_pref, None, gpu_memory, api_visibility="private", queue=False)
    return demo


if ON_MAC:
    HOW_STEPS = """2. **Every next segment** is generated by LTX-2's *extend* mode: the last 25 frames of the shot so far
   (about one second of picture **and sound**) are pinned at the start of the new clip, so motion, camera and audio
   carry straight on.
3. The joins are cross-faded (picture and sound) and color-locked, then everything is assembled into one mp4."""
    RESUME_NOTE = "finished segments are kept."
    PLATFORM_NOTE = """
On macOS 26 a GPU watchdog bug can abort long renders. If a segment fails with a GPU timeout or watchdog error, turn
on **GPU watchdog workaround** in Settings, keep the display awake, and press Resume.
"""
else:
    HOW_STEPS = """2. **Every next segment** is one more chunk of LTX-2's own long-video mode: the last 25 frames of
   the segment before (about one second of picture **and sound**) are pinned at its start, so motion, camera and
   audio carry straight on. The whole shot is made in one model run, and LTX-2 cross-fades the joins itself.
3. The decoded pieces are assembled into one mp4."""
    RESUME_NOTE = ("the shot is rendered again from the start, with the same seed (on Windows the segments are made "
                   "in one run, so they can't be kept separately yet).")
    PLATFORM_NOTE = """
With Low-RAM mode on, the model's weights stream from the SSD instead of being held in memory: this is what lets
32 GB of RAM work. If the GPU isn't found, update the AMD Adrenalin driver and restart Windows.
"""

HELP = f"""
### How it works
TDPlay AI Video Render rebuilds RunningHub's *MINIMAX H3 Motion Context* workflow with open weights:

1. **Segment 1** is generated from its prompt — and from your character photo, if you add one.
{HOW_STEPS}

Optionally, the finished shot is then **upscaled to 1080p or 4K** with Real-ESRGAN on the {COMPUTER}'s GPU (choose
*Final output size* next to Generate, or upscale any finished project from the Projects tab). *Fast* takes minutes;
*Best detail* gives sharper texture but takes several times longer. The original-size video is always kept.

After each segment you can watch the shot so far. If anything stops it (you, an error, the {COMPUTER} running out of
memory), press **Resume**: {RESUME_NOTE}

**Renders run in the background.** You can close the window or quit the app while a video is being made: it keeps
going, and you get a notification when it's done. Open the app again to watch the progress or see the result.

### Speed and memory
Everything runs on your {COMPUTER}'s GPU. A segment typically takes a few minutes; long, high-resolution segments take
longer. If a segment fails with "ran out of memory": turn on Low-RAM mode, lower the resolution, use shorter
segments, close other apps, then press Resume.
{PLATFORM_NOTE}
### Where things are
- Projects (videos, each segment, a log): the folder shown in the Projects tab (`{PROJECTS_DIR}` unless you
  choose another one there)
- Models: `{models.MODELS_DIR}`

### Differences from MiniMax H3
H3 segments can be 15 s each; on a {COMPUTER}, LTX-2 segments of 3–6 s work best (use more of them). H3's "shared
reference" keeps a face from the photo; here the closest match is the *re-anchor* setting, which works best when the
character stays in roughly the same framing.
"""


def launch(*, port: int | None = None, open_browser: bool = True, block: bool = True) -> gr.Blocks:
    """Start the local web server (127.0.0.1 only). With ``block=False`` it returns once the server is up."""
    try:
        projects_dir().mkdir(parents=True, exist_ok=True)
    except OSError:  # a chosen folder that isn't there now (an unplugged drive): the Projects tab says so
        pass
    demo = build()
    _state["demo"] = demo
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=open_browser,
        prevent_thread_lock=not block,
        allowed_paths=[str(models.HOME), *map(str, project_folders())],
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=CSS,
        footer_links=[],
    )
    return demo


def main() -> None:
    launch(port=int(os.environ["TDPLAY_PORT"]) if os.environ.get("TDPLAY_PORT") else None,
           open_browser=not os.environ.get("TDPLAY_NO_BROWSER"))
