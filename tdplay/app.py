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

from tdplay import __version__, engine, models

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

RAM_GB = models.system_ram_gb()
REC = models.recommend(RAM_GB)
IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

_state: dict = {"runner": None, "last_project": None}

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
        return json.loads(_prefs_path().read_text())
    except (OSError, ValueError):
        return {}


def save_pref(key: str, value) -> None:
    prefs = load_prefs()
    prefs[key] = value
    _prefs_path().parent.mkdir(parents=True, exist_ok=True)
    _prefs_path().write_text(json.dumps(prefs, indent=2))


def theme_pref() -> str:
    mode = load_prefs().get("theme", "system")
    return mode if mode in {v for _, v in THEMES} else "system"
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


def model_paths(preset_key: str) -> tuple[str, str | None]:
    """Model and Gemma folders; ``TDPLAY_MODEL_DIR`` / ``TDPLAY_GEMMA_DIR`` override them."""
    if os.environ.get("TDPLAY_MODEL_DIR"):
        return os.environ["TDPLAY_MODEL_DIR"], os.environ.get("TDPLAY_GEMMA_DIR") or None
    preset = models.PRESETS[preset_key]
    gemma = str(models.local_dir(preset.gemma_repo)) if preset.gemma_repo else None
    return str(models.local_dir(preset.repo)), gemma


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


def open_folder(path: str | None) -> None:
    if path and Path(path).exists():
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _stream(project: engine.Project, job=lambda runner: runner.run(), resume_hint: bool = True):
    """Run a job on ``project`` (by default: generate it) and turn its events into (status, video, log) updates."""
    with _lock:
        if _state["runner"] is not None:
            yield (status_html("Already working — wait for it to finish or press Stop.", tone="warn"),
                   gr.skip(), gr.skip())
            return
        runner = engine.Runner(project)
        _state["runner"], _state["last_project"] = runner, str(project.dir)

    log: deque[str] = deque(maxlen=400)
    started = time.time()
    last_push = 0.0
    title, fraction, detail = "Starting…", 0.0, ""
    log.append(f"Project folder: {project.dir}")
    try:
        for event in job(runner):
            video = gr.skip()
            if event.kind == "log":
                log.append(event.text)
            elif event.kind == "progress":
                title, fraction = event.text, event.fraction
                detail = _eta(started, fraction)
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
                yield status_html(head, fraction, detail, tone), gr.skip(), log_html(log)
                return
            # tqdm lines arrive many times a second; refresh the page at most twice a second.
            now = time.time()
            if event.kind != "log" or now - last_push > 0.5:
                last_push = now
                yield status_html(title, fraction, detail), video, log_html(log)
    finally:
        with _lock:
            _state["runner"] = None


def generate(image, start_video, n, *rest):
    prompts, seconds = rest[:MAX_SEGMENTS], rest[MAX_SEGMENTS : 2 * MAX_SEGMENTS]
    (preset_key, quality, aspect, low_ram, fast_math, watchdog, anchor, seam, color_lock, seed, out_size,
     up_quality) = rest[2 * MAX_SEGMENTS :]
    n = int(n)
    if not IS_APPLE_SILICON and not os.environ.get("TDPLAY_LTX_CMD"):
        yield (status_html("TDPlay AI Video Render needs a Mac with Apple Silicon (M1 or newer) to generate.",
                           tone="error"), None, gr.skip())
        return
    if not model_ready(preset_key):
        yield (status_html("Download the model first: open the Models tab and press Download.", tone="warn"),
               None, gr.skip())
        return
    width, height = resolution(quality, aspect)
    model_dir, gemma_dir = model_paths(preset_key)
    settings = engine.Settings(
        width=width, height=height, model_dir=model_dir, gemma_dir=gemma_dir, low_ram=bool(low_ram),
        fast_math=bool(fast_math), seed=int(seed if seed is not None else -1), anchor_strength=float(anchor),
        seam_frames=int(seam), color_lock=bool(color_lock), gpu_watchdog_fix=bool(watchdog),
        upscale=out_size or "off", upscale_quality=up_quality or "fast",
    )
    segments = [engine.Segment(str(p or ""), int(s)) for p, s in zip(prompts[:n], seconds[:n])]
    try:
        project = engine.create_project(PROJECTS_DIR, settings, segments, reference_image=image or None,
                                        start_video=start_video or None)
    except (ValueError, RuntimeError) as exc:
        yield status_html(str(exc), tone="error"), None, gr.skip()
        return
    yield from _stream(project)


def resume(project_dir, low_ram, fast_math, watchdog):
    project_dir = project_dir or _state["last_project"]
    if not project_dir:
        yield status_html("Nothing to resume yet.", tone="warn"), gr.skip(), gr.skip()
        return
    project = engine.Project.load(project_dir)
    if project.status == "complete":
        yield (status_html("That project is already finished.", 1.0, str(project.output), "ok"),
               str(project.output), gr.skip())
        return
    # Memory/GPU settings may change between attempts; everything that shapes the shot stays as it was.
    s = project.settings
    s.low_ram, s.fast_math, s.gpu_watchdog_fix = bool(low_ram), bool(fast_math), bool(watchdog)
    yield from _stream(project)


def upscale_project(project_dir, size, quality):
    if not project_dir:
        yield status_html("Choose a project in the Projects tab first.", tone="warn"), gr.skip(), gr.skip()
        return
    project = engine.Project.load(project_dir)
    yield from _stream(project, lambda runner: runner.upscale(size, quality), resume_hint=False)


def is_busy() -> bool:
    """True while a render, an upscale or a model download is running (the Mac app asks before quitting)."""
    return _state["runner"] is not None or bool(_state.get("downloading"))


def shutdown() -> None:
    """Stop a running job; called when the app quits."""
    runner = _state["runner"]
    if runner is not None:
        runner.cancel()


def resume_last(low_ram, fast_math, watchdog):
    yield from resume(None, low_ram, fast_math, watchdog)


def stop():
    runner = _state["runner"]
    if runner is None:
        return status_html("Nothing is running.", tone="warn")
    threading.Thread(target=runner.cancel, daemon=True).start()
    return status_html("Stopping…", tone="warn", detail="The current segment is discarded; finished ones are kept.")


# ---------------------------------------------------------------------------
# projects + models tabs
# ---------------------------------------------------------------------------


def project_choices() -> list[tuple[str, str]]:
    choices = []
    for p in engine.list_projects(PROJECTS_DIR):
        label = f"{p.created.replace('T', ' ')[:16]} · {p.title[:60]} · {p.status} ({p.done}/{len(p.segments)})"
        choices.append((label, str(p.dir)))
    return choices


def refresh_projects():
    choices = project_choices()
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def show_project(project_dir):
    if not project_dir:
        return None, ""
    p = engine.Project.load(project_dir)
    head = (f"**{p.title}**\n\nStatus: {p.status} · {p.done} of {len(p.segments)} segments · "
            f"{p.settings.width}×{p.settings.height} · seed {p.settings.seed}\n")
    lines = [head] + [f"{i + 1}. ({s.seconds} s) {s.prompt}" for i, s in enumerate(p.segments)]
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
    for key, p in models.PRESETS.items():
        ready = "✅ ready" if models.preset_ready(p) else "not downloaded"
        mark = " ← recommended for this Mac" if key == REC.preset else ""
        rows.append(f"| {p.label}{mark} | {_SIZES.get(key, '?')} | {ready} |")
    free = shutil.disk_usage(models.HOME if models.HOME.exists() else Path.home()).free / 1e9
    rows.append(f"\nFree disk space: **{free:.0f} GB** · models are stored in `{models.MODELS_DIR}`")
    return "\n".join(rows)


_SIZES = {"ltx-2.5-q8": "≈ 45 GB", "ltx-2.3-q8": "≈ 38 GB", "ltx-2.3-q4": "≈ 29 GB"}


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
    for repo in models.repos_for(preset):
        if models.is_downloaded(repo):
            continue
        started = time.time()
        try:
            for progress in models.download(repo):
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
                f"LTX-2 via MLX · v{__version__}</small>",
                scale=5,
            )
            appearance = gr.Radio(THEMES, value=theme_pref(), label="Appearance", scale=1, min_width=260)
        with gr.Tabs() as tabs:
            with gr.Tab("Create", id="create"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        image = gr.Image(label="Character / first frame (optional)", type="filepath",
                                         sources=["upload", "clipboard"], height=240)
                        with gr.Accordion("…or continue a video you already have", open=False):
                            start_video = gr.Video(label="Start video (any size; it is cropped to the shot size)",
                                                   sources=["upload"], height=200)
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
                            preset = gr.Dropdown([(p.label, k) for k, p in models.PRESETS.items()], value=REC.preset,
                                                 label="Model")
                            with gr.Row():
                                quality = gr.Radio(list(QUALITIES), value=REC.quality, label="Resolution")
                                aspect = gr.Radio(ASPECTS, value=ASPECTS[0], label="Shape")
                            low_ram = gr.Checkbox(REC.low_ram, label="Low-RAM mode",
                                                  info="Streams the model from disk: ~5% slower, needs far less memory.")
                            fast_math = gr.Checkbox(True, label="Faster math (float16)",
                                                    info="About 20% faster. Turn off if you see glitches.")
                            watchdog = gr.Checkbox(False, label="GPU watchdog workaround (macOS 26)",
                                                   info="Turn on if a render dies with a GPU timeout / watchdog "
                                                        "error. The Mac may feel less responsive while rendering.")
                            anchor = gr.Slider(0, 1, value=0, step=0.05,
                                               label="Keep the character's face: re-anchor the photo every segment",
                                               info="0 = off. 0.5–0.7 stops face drift, but pulls each segment's "
                                                    "last frame toward the photo's pose and framing.")
                            seam = gr.Slider(0, 16, value=8, step=1, label="Seam blend (frames)")
                            color_lock = gr.Checkbox(True, label="Color lock at seams",
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
                            detail="" if IS_APPLE_SILICON else "Generation needs a Mac with Apple Silicon."))
                        video = gr.Video(label="Your shot (updates after every segment)", interactive=False,
                                         height=420, loop=False)
                        with gr.Row():
                            folder_btn = gr.Button("Open project folder", size="sm")
                        log = gr.HTML(log_html([]))

            with gr.Tab("Projects", id="projects") as projects_tab:
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
                    "downloaded (about half of each pack)."
                )
                table = gr.Markdown(models_table())
                with gr.Row():
                    dl_preset = gr.Dropdown([(p.label, k) for k, p in models.PRESETS.items()], value=REC.preset,
                                            label="Model to download", scale=4)
                    dl_btn = gr.Button("Download", variant="primary", scale=1)
                dl_status = gr.HTML()
                gr.Markdown(
                    f"This Mac: **{chip}, {RAM_GB} GB**. Recommended: **{models.PRESETS[REC.preset].label}**, "
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
        go.click(generate,
                 [image, start_video, n_segments, *prompts, *seconds, preset, quality, aspect, low_ram, fast_math,
                  watchdog, anchor, seam, color_lock, seed, out_size, up_quality],
                 outputs, api_visibility="private", show_progress="hidden")
        stop_btn.click(stop, None, status, api_visibility="private", queue=False)
        resume_btn.click(resume_last, [low_ram, fast_math, watchdog], outputs, api_visibility="private",
                         show_progress="hidden")
        folder_btn.click(lambda: open_folder(_state["last_project"] or str(PROJECTS_DIR)), None, None,
                         api_visibility="private")

        refresh_btn.click(refresh_projects, None, project, api_visibility="private")
        project.change(show_project, project, [project_video, project_info], api_visibility="private")
        project_folder_btn.click(open_folder, project, None, api_visibility="private")
        resume_project_btn.click(lambda: gr.Tabs(selected="create"), None, tabs, api_visibility="private").then(
            resume, [project, low_ram, fast_math, watchdog], outputs, api_visibility="private",
            show_progress="hidden")

        upscale_btn.click(lambda: gr.Tabs(selected="create"), None, tabs, api_visibility="private").then(
            upscale_project, [project, p_size, p_quality], outputs, api_visibility="private", show_progress="hidden")

        dl_btn.click(download_models, dl_preset, [dl_status, table], api_visibility="private",
                     show_progress="hidden")
        preset.change(preset_status, preset, status, api_visibility="private")
        projects_tab.select(refresh_projects, None, project, api_visibility="private")
        models_tab.select(models_table, None, table, api_visibility="private")
        demo.load(models_table, None, table, api_visibility="private")
        # The saved choice is read on every page load, so all windows and browsers agree.
        appearance.input(lambda mode: save_pref("theme", mode), appearance, None, js=APPLY_THEME_JS,
                         api_visibility="private", queue=False)
        demo.load(theme_pref, None, appearance, api_visibility="private", queue=False).then(
            None, appearance, None, js=APPLY_THEME_JS)
    return demo


HELP = f"""
### How it works
TDPlay AI Video Render rebuilds RunningHub's *MINIMAX H3 Motion Context* workflow with open weights:

1. **Segment 1** is generated from its prompt — and from your character photo, if you add one.
2. **Every next segment** is generated by LTX-2's *extend* mode: the last 25 frames of the shot so far
   (about one second of picture **and sound**) are pinned at the start of the new clip, so motion, camera and audio
   carry straight on.
3. The joins are cross-faded (picture and sound) and color-locked, then everything is assembled into one mp4.

Optionally, the finished shot is then **upscaled to 1080p or 4K** with Real-ESRGAN on the Mac's GPU (choose
*Final output size* next to Generate, or upscale any finished project from the Projects tab). *Fast* takes minutes;
*Best detail* gives sharper texture but takes several times longer. The original-size video is always kept.

After each segment you can watch the shot so far. If anything stops it (you, an error, the Mac running out of memory),
press **Resume**: finished segments are kept.

### Speed and memory
Everything runs on your Mac's GPU. A segment typically takes a few minutes; long, high-resolution segments take longer.
If a segment fails with "ran out of memory": turn on Low-RAM mode, lower the resolution, use shorter segments, close
other apps, then press Resume.

On macOS 26 a GPU watchdog bug can abort long renders. If a segment fails with a GPU timeout or watchdog error, turn
on **GPU watchdog workaround** in Settings, keep the display awake, and press Resume.

### Where things are
- Projects (videos, each segment, a log): `{PROJECTS_DIR}`
- Models: `{models.MODELS_DIR}`

### Differences from MiniMax H3
H3 segments can be 15 s each; on a Mac, LTX-2 segments of 3–6 s work best (use more of them). H3's "shared
reference" keeps a face from the photo; here the closest match is the *re-anchor* setting, which works best when the
character stays in roughly the same framing.
"""


def launch(*, port: int | None = None, open_browser: bool = True, block: bool = True) -> gr.Blocks:
    """Start the local web server (127.0.0.1 only). With ``block=False`` it returns once the server is up."""
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    demo = build()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=open_browser,
        prevent_thread_lock=not block,
        allowed_paths=[str(models.HOME)],
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
        css=CSS,
        footer_links=[],
    )
    return demo


def main() -> None:
    launch(port=int(os.environ["TDPLAY_PORT"]) if os.environ.get("TDPLAY_PORT") else None,
           open_browser=not os.environ.get("TDPLAY_NO_BROWSER"))
