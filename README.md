# TDPlay AI Video Render 🎬

**One continuous AI video shot, built segment by segment — free, private, on your Mac or Windows PC.**

TDPlay AI Video Render is a local take on RunningHub's *MINIMAX H3 Motion Context* workflow. You write one prompt per
segment. Each new segment continues the last second of the one before it, **picture and sound together**,
so the result plays as a single unbroken take of 30, 45, 60+ seconds instead of a set of cuts.

It runs the open-weights **LTX-2** video model from Lightricks (which makes video *with* audio) natively on Apple
Silicon through [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), and on Windows PCs with an AMD Radeon RX 7000 /
RX 9000 card through Lightricks' own PyTorch code and AMD ROCm. No account, no credits, no upload: after the
one-time model download everything runs offline on your computer.

> **Windows is new (beta).** The Mac version is the main one. The Windows version has been tested against the real
> LTX-2 code but not yet on a real Radeon card; see [Windows](#windows-amd-radeon-rx-7000--rx-9000).

---

## What you need

| | |
|---|---|
| Mac | Apple Silicon: M1 or newer. Built with **M4 and M5** in mind. |
| macOS | 14 Sonoma or newer |
| Memory | 16 GB minimum, **32 GB or more recommended** |
| Disk | 30–45 GB for the model, plus space for your videos |
| Internet | only for installing and the one-time model download |

## Install on a Mac (once)

1. Download **`TDPlay-AI-Video-Render-<version>.dmg`** from the
   [Releases](https://github.com/mircix/TDPlay-AI-Video-Render/releases) page, open it and drag
   **TDPlay AI Video Render** onto **Applications**.
2. Open it from Applications. The app is free and not registered with Apple, so the first time macOS says it can't
   check it for malicious software. Click **Done**, open **System Settings → Privacy & Security**, scroll down and
   click **Open Anyway**. You only do this once.
3. On its first launch the app sets itself up: it downloads Python, Apple's MLX, the LTX-2 code and ffmpeg (about
   1 GB, a few minutes) into `~/TDPlay/runtime`, with a small window while it works. No Terminal, Homebrew or
   password needed. Then it opens; download the model in its **Models** tab (30–45 GB).

It opens in its own window; drag its icon to the Dock to keep it there. Quit it like any app (⌘Q): a video that is
being made keeps going.

### …or from the source folder

1. Get the `TDPlay-AI-Video-Render` folder (clone it from GitHub, or unzip `TDPlay-AI-Video-Render.zip`) and keep it
   somewhere permanent, for example your home folder or your Claude Workspace folder.
2. Double-click **`install.command`**.
   * If macOS says it "cannot be opened because it is from an unidentified developer": right-click the file →
     **Open** → **Open**. Or open Terminal and run `bash ~/TDPlay-AI-Video-Render/install.command` (adjust the path).
   * The installer sets up [Homebrew](https://brew.sh) (it asks first), ffmpeg, Python and the model code, and adds
     **TDPlay AI Video Render** to your Applications folder. At the end it offers to download the recommended model
     for your Mac (30–45 GB). You can also do that later in the app's **Models** tab.
3. Open **TDPlay AI Video Render** from Applications, Launchpad or Spotlight. It opens in its own window; drag its
   icon to the Dock to keep it there. Quit it like any app (⌘Q): a video that is being made keeps going.
   * `start.command` still works too: it opens the same app in your web browser, with a Terminal window showing the
     log.
   * If you move the `TDPlay-AI-Video-Render` folder, run `install.command` again so the app finds it.

## Make a long take

1. *(Optional)* Drop in a **character / first-frame photo**. Segment 1 starts from it.
   Or open *"…or continue a video you already have"* to extend one of your own clips.
2. Pick the number of **segments** and write **one prompt per segment**, each with a length in seconds.
3. Press **Generate**. The shot so far appears after every segment, so you can watch it grow.
4. The finished `render.mp4` is in the project folder (**Open project folder**).

**Renders run in the background.** Close the window or quit the app whenever you like: the video keeps being made,
and you get a notification when it's done. Open the app again to see the progress or the result. Only **Stop** stops
it.

### 1080p and 4K

Choose **Final output size** next to Generate: *1080p* or *4K*. After the last segment the finished shot is upscaled
with **Real-ESRGAN** on the Mac's GPU, and saved as `render_1080p.mp4` or `render_4k.mp4` next to the original
`render.mp4` (which is kept). You can also upscale any finished project later: **Projects** tab → choose it →
*Upscale this project*.

* **Fast** (realesr-general-x4v3) is made for real-world video and takes minutes for a short shot.
* **Best detail** (RealESRGAN_x4plus) gives sharper texture but takes several times longer. Use it for short shots
  or let it run overnight.

The shot is rendered at the size your Mac can handle (512p–704p) and enlarged 4×. Upscaling sharpens and cleans
what is there; it cannot invent detail the way rendering natively at 4K would. 4K files use HEVC (H.265), which
every current Mac, iPhone and browser plays.

If a run stops for any reason (you pressed Stop, the Mac ran out of memory, it restarted), press **Resume**.
Finished segments are kept. Older projects are in the **Projects** tab.

### Writing good segments

* **Segment 1 is the shot.** Describe who, where, the camera, the light and the sound.
* **Every later segment continues the action.** Start with it: *"She keeps walking at the same pace, the camera
  keeps tracking…"*. Same pace, same camera.
* **Put the joins mid-movement** (mid-walk, mid-pan). Avoid making a join coincide with a cut, a doorway or
  something entering the frame.
* **Describe the sound** (footsteps, rain, crowd, music). LTX-2 generates audio, and it carries across the joins too.
* **Use shorter segments.** On a Mac, 3–6 s segments work best. For 60 s, use 12–15 segments.

## Which settings for my Mac?

The app picks these automatically from your memory size. You can change them under **Settings**.

| Memory | Model | Resolution | Segment length | Low-RAM mode |
|---|---|---|---|---|
| 16–24 GB | LTX-2.3 · 8-bit (≈ 38 GB download) | 512p | up to 4 s | on |
| 32 GB | LTX-2.5 · 8-bit (≈ 45 GB download) | 576p | up to 5 s | on |
| 48–64 GB | LTX-2.5 · 8-bit | 704p (1280×704) | up to 6 s | on |
| 96 GB+ | LTX-2.5 · 8-bit | 704p | up to 8 s | off (faster) |

*LTX-2.3 · 4-bit* (≈ 29 GB) is the smallest download if disk space is tight.

**Speed.** Expect several minutes per segment at 512p, and more at 704p and on longer segments. It varies a lot
by chip, so treat your first run as the benchmark. For reference, the ltx-2-mlx authors measured on an M1 Max
with LTX-2.5 8-bit: about 100 s to add ⅔ s at 768×512, and about 10 min to add 4 s at 1280×704. M4 and M5
chips should be faster. The app keeps the Mac from idle-sleeping while it is open.

### Settings explained

* **Low-RAM mode** streams the model from disk instead of holding it all in memory (~5 % slower).
* **Faster math (float16)** is about 20 % faster. Turn it off if you see glitches.
* **GPU watchdog workaround (macOS 26)**: turn this on if a render dies with a GPU timeout or watchdog error.
* **Keep the character's face**: LTX-2 only sees the last second of the shot when it continues it. If a face turns
  away and comes back, it can come back as a different face. This setting re-applies your photo as a soft target
  at the end of every segment (0.5–0.7 works well). The trade-off is that each segment drifts back toward the
  photo's pose and framing, so it suits shots where the character stays roughly in frame.
* **Seam blend** and **Color lock** hide the joins (cross-fade + color correction). Leave them on.
* **Seed**: −1 for a random one. The seed of every project is saved, so you can repeat a run.

## Windows (AMD Radeon RX 7000 / RX 9000)

| | |
|---|---|
| Windows | **Windows 11** (AMD's PyTorch for Windows is made for it), with a current **AMD Adrenalin** driver |
| Graphics card | **Radeon RX 7900 XTX / XT / GRE**, RX 7800 XT, RX 7700 XT, RX 7600 · **RX 9070 XT / 9070**, RX 9060 XT, Radeon AI PRO R9700. NVIDIA RTX cards work too (CUDA). |
| CPU | any recent AMD Ryzen or Intel (the GPU does the work) |
| Memory | 16 GB minimum, **32 GB recommended** |
| Disk | about **70 GB** (55 GB model + PyTorch), plus space for your videos. An SSD helps a lot. |

A 24 GB card (RX 7900 XTX, AI PRO R9700) is the best fit. 16 GB cards (RX 7800 XT, RX 9070 XT, RX 9060 XT) work at
the lower resolutions. 8–12 GB cards are likely too small.

### Install on Windows (once)

1. Download **`TDPlay-AI-Video-Render-Setup-<version>.exe`** from the
   [Releases](https://github.com/mircix/TDPlay-AI-Video-Render/releases) page and run it. If Windows says "Windows
   protected your PC": **More info** → **Run anyway**. It installs for your user only (no administrator rights),
   into `%LOCALAPPDATA%\Programs\TDPlay AI Video Render`.
2. After copying the app, a setup window finds your graphics card and installs
   Python 3.13 (python.org's build), [uv](https://docs.astral.sh/uv/), ffmpeg, Lightricks' LTX-2 code, and **AMD's ROCm
   PyTorch for your card** (RX 7900 → `gfx1100`, RX 7800/7700 → `gfx1101`, RX 7600 → `gfx1102`, RX 9060 →
   `gfx1200`, RX 9070 / R9700 → `gfx1201`).
   It then checks that PyTorch can see the GPU. Keep that window open until it closes by itself. If it stops (or was
   closed), open **TDPlay AI Video Render - repair setup** from the Start menu: it continues where it stopped, and
   its log is `setup.log` in the install folder.
3. Open **TDPlay AI Video Render** from the Start menu or the desktop and download the model (about 55 GB) in its
   **Models** tab. It opens in its own window, in dark or light mode like Windows. Closing it doesn't stop a video
   that is being made.
   * `start.bat` (in the install folder) opens the same app in your web browser, with a window showing the log.

**From the source folder instead:** get the `TDPlay-AI-Video-Render` folder, keep it somewhere permanent (not in a
OneDrive-synced folder) and double-click **`install.bat`**. It does the same setup and makes the shortcuts.

### Recommended settings on Windows

The app picks these for you: **LTX-2.3 distilled, 8-bit** (official Lightricks weights, ≈ 55 GB with the Gemma 3
text encoder), **576p**, segments **up to 5 s**. With less than 48 GB of RAM, *Low-RAM mode* is on: the model's
weights stream from the SSD instead of being held in memory, which is what makes 32 GB of RAM work. As much of the model
as fits stays on the graphics card for the whole render, and only the rest streams from the SSD: the log says
how much ("GPU: keeping 26 of 48 transformer blocks on the card"). The more of it stays on the card, the faster. The
model never takes more than 80 % of the card's memory: Windows, the display and your other apps need the rest, and
when a card runs out, Windows makes everything crawl and the display driver can crash. *Graphics card memory:
Maximum* in Settings raises that to 90 % for more speed; close games and other GPU-heavy apps while it renders, and
go back to Balanced if the screen flickers or goes black. 704p (1280×704) is worth trying on a 24 GB card.

### How the Windows version differs

* **The whole shot is one model run.** Every segment is one chunk of LTX-2's own long-video mode: the last 25 frames
  of a segment and the sound under them are pinned at the start of the next one, and LTX-2 cross-fades the overlap.
  It is the same continuation the Mac version does segment by segment. You still see the shot grow after every
  segment.
* **Resume starts the shot over.** The carry between segments lives inside the model run, so a stopped shot is
  rendered again from the beginning (with the same seed).
* **No "continue a video you already have"** yet. A start photo works.
* **1080p / 4K upscaling** works the same way, with Real-ESRGAN on the GPU through PyTorch.
* **Speed is not measured yet.** Your first run is the benchmark; the log shows every step.

### Troubleshooting on Windows

* **"The GPU check failed"** → update the AMD Adrenalin driver, restart Windows, run `install.bat` again.
* **"ran out of memory"** → keep Low-RAM mode on, lower the resolution, use shorter or fewer segments, close other
  apps (games and browsers especially), then **Resume**.
* **"ffmpeg was installed but Windows can't find it yet"** → sign out and back in, run `install.bat` again.
* **The full log** is `log.txt` in each project folder. The app's own log is `%USERPROFILE%\TDPlay\logs\app.log`.
* **Moved the folder?** Run `install.bat` again so the shortcuts point to it.

## How it works

```
 photo ─► [ segment 1: generate ]──► ... last 25 frames + their sound ...
                                                  │ pinned at the start of
                                                  ▼ the next window
                          [ segment 2: extend ]──► ... last 25 frames + sound ...
                                                                   ▼
                                           [ segment 3: extend ] ─► ...
 all segments ─► cross-fade picture + audio at each join, color-lock ─► render.mp4
```

* Segment 1 runs `ltx-2-mlx generate --distilled` (image-to-video when you give a photo).
* Every later segment runs `ltx-2-mlx extend --distilled`. This is Lightricks' own *chunk continuation*: the previous
  clip's last 25 frames and the matching audio are pinned at the start of a new window, and the model fills in the
  rest. The model's re-render of those 25 frames is used to blend the join, then dropped.
* Each model run is a separate process. If macOS kills one for using too much memory, the app survives and you
  can press Resume.
* The whole job runs in a background process of its own (`tdplay/worker.py`), not in the app: it writes its
  progress to `events.jsonl` in the project folder, and any app window, also one opened later, follows that file.
* Everything for a project lives in its own folder: the raw segments, the assembled chunks, a log and
  `project.json` (prompts, settings, seeds, progress).

### Compared with MiniMax H3 on RunningHub

| | MiniMax H3 on RunningHub | TDPlay AI Video Render |
|---|---|---|
| Cost | paid per run | free |
| Where it runs | cloud | your Mac, offline |
| Model | MiniMax H3 (closed) | LTX-2.3 / LTX-2.5 (open weights) |
| Continuity | last frames, picture + sound | last 25 frames (~1 s), picture + sound |
| Segment length | up to 15 s | best at 3–6 s (more segments instead) |
| Face reference on every segment | "shared reference" | "keep the character's face" re-anchor |
| Resolution | 1.2 MP | up to 1280×704 (~0.9 MP) |

H3 is a larger cloud model. Expect LTX-2 on a laptop to be good but not identical in quality.

## Troubleshooting

* **"ran out of memory"** → turn on Low-RAM mode, lower the resolution, use shorter segments, close other apps
  (browsers with many tabs especially), then **Resume**.
* **GPU timeout / watchdog error** (macOS 26) → Settings → *GPU watchdog workaround*, keep the screen on, **Resume**.
* **Download stopped** → press **Download** again. It continues where it stopped.
* **Faces change over a long shot** → raise *Keep the character's face* to 0.5–0.7, and avoid prompts where the
  character turns away at a join.
* **Nothing happens on double-click** → see the right-click → Open tip under Install.
* **The full log** is `log.txt` in each project folder. The app's own log is `~/TDPlay/logs/app.log`.
* **"Connection to the server was lost"** → the window lost its connection to the app (for example the Terminal
  window of `start.command` was closed). The render itself keeps going: open the app again to follow it.
* **The app doesn't open, or says its folder was moved** → run `install.command` again from the folder's current
  location.

## Where things are

* Projects: `~/TDPlay/projects/<date>_<title>/render.mp4` (plus `render_1080p.mp4` / `render_4k.mp4` if upscaled)
* Models: `~/TDPlay/models/` (delete a folder there to free its space)
* Logs: `~/TDPlay/logs/` (`app.log`; `setup.log` for the Mac app's first-launch setup)
* The Mac app from the disk image: `/Applications/TDPlay AI Video Render.app`, which sets up Python and the packages
  in `~/TDPlay/runtime`.
* The Windows app from Setup.exe: `%LOCALAPPDATA%\Programs\TDPlay AI Video Render` (`.venv\` and `vendor\` in it are
  made by the setup).
* From the source folder: the app is a launcher for this folder, which holds `.venv/` and `vendor/`.
* On Windows, `~` is your user folder (`C:\Users\<you>`).

## Update or uninstall

* **Update**: install the new version over the old one: drag the new app from the disk image onto Applications
  (**Replace**), or run the new Setup.exe. The app sets itself up again on the next launch, which is quick. From
  the source folder: `git pull`, then run `install.command` / `install.bat` again. Models and projects stay in
  `~/TDPlay`.
* **Uninstall on a Mac**: drag the app to the Trash and delete `~/TDPlay` (models, videos and the runtime). From the
  source folder, also delete the folder; ffmpeg and uv can be removed with `brew uninstall ffmpeg uv`.
* **Uninstall on Windows**: **Settings → Apps → Installed apps → TDPlay AI Video Render → Uninstall**. It asks whether
  to delete `%USERPROFILE%\TDPlay` (models and videos) too. From the source folder: delete the shortcuts, the folder
  and `%USERPROFILE%\TDPlay`.

## Credits and licenses

* **LTX-2** by [Lightricks](https://huggingface.co/Lightricks), under the
  [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE). Free to use, but
  organizations with annual revenue of $10M or more need a paid commercial license from Lightricks. It also includes
  an acceptable-use policy. Read it before any commercial use.
* **[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)** (MIT) by dgrauet: the MLX port and the pre-converted weights
  (`dgrauet/ltx-2.x-mlx-*` on Hugging Face). TDPlay AI Video Render pins release `v0.16.3`.
* **[LTX-2](https://github.com/Lightricks/LTX-2)** code by Lightricks, release `v1.4.2`, runs the Windows version, with
  the official `Lightricks/LTX-2.3-fp8` weights. Code and weights are under the same LTX-2 Community License.
* **Gemma 3** (`mlx-community/gemma-3-12b-it-4bit` on Mac, `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized` on
  Windows) is the text encoder, under the Gemma terms.
* **[PyTorch](https://pytorch.org)** (BSD-3-Clause) with **AMD ROCm** on Windows, and
  **[spandrel](https://github.com/chaiNNer-org/spandrel)** (MIT) for Real-ESRGAN there.
* **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)** (BSD-3-Clause) by Xintao Wang et al. for upscaling,
  using the MLX conversions `mlx-community/Real-ESRGAN-general-x4v3` and `mlx-community/Real-ESRGAN-x4plus`.
* **[Gradio](https://gradio.app)** (Apache-2.0) for the interface, **[pywebview](https://pywebview.flowrl.com)**
  (BSD-3-Clause) for the app window, **[ffmpeg](https://ffmpeg.org)** for video.
* "RunningHub", "MiniMax" and "H3" are named only to describe the workflow TDPlay AI Video Render recreates. This project is not
  affiliated with them.

## For developers

```bash
.venv/bin/python -m tdplay                 # start the UI in the browser (TDPLAY_PORT, TDPLAY_NO_BROWSER, TDPLAY_HOME)
.venv/bin/python -m tdplay.desktop_app    # the same UI in its own window (what the .app runs)
.venv/bin/python -m tdplay download        # download the recommended model (or: download ltx-2.3-q4)
.venv/bin/python -m tdplay.upscale IN.mp4 OUT.mp4 --width 3840 --height 2160 --quality fast
.venv/bin/python -m tdplay.macos_bundle    # (re)build the app in /Applications
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest tests             # runs the whole pipeline against a fake ltx-2-mlx (no GPU needed)
```

`tests/fake_ltx.py` stands in for the `ltx-2-mlx` CLI. It writes frame numbers into the pixels and a continuous
tone into the audio, so the tests can prove there are no dropped or duplicated frames and no audio gaps at the
joins. The tests also check every command TDPlay AI Video Render builds against the real `ltx-2-mlx` argument parser, and cover
resuming after an out-of-memory kill, stop/resume and continuing an existing video. The upscaler is checked against
the original PyTorch Real-ESRGAN (`tests/test_upscale.py`), and the app launcher is run under bash 3.2, the version
macOS uses (`tests/test_macos_bundle.py`; set `TDPLAY_BASH32` to a bash 3.2 binary to include it).

The Windows backend has its own stand-in, `tests/fake_ltx_torch.py`, and `tests/test_ltx_torch.py` runs TDPlay's
chunk plan through the real `ltx_pipelines` classes on the CPU (it skips if `ltx_pipelines` isn't installed).
`tests/test_windows_scripts.py` checks that the `.bat`/`.ps1` files are ASCII with CRLF line endings and, with
`TDPLAY_PWSH` set to a PowerShell binary, parses the installer and tests its graphics card detection.
`tests/test_worker.py` covers background jobs: a render outliving the app that started it, Stop, and a window
reopening onto a running job.

**Installers.** `bash macos/build_dmg.sh` builds the Mac disk image (on a Mac), and
`iscc /DAppVersion=1.2.0 windows\installer\TDPlay.iss` the Windows Setup.exe ([Inno Setup 6](https://jrsoftware.org/isinfo.php)).
The *Installers* GitHub workflow (`.github/workflows/installers.yml`) builds both, installs each one the way a user
would (the Windows one with CPU PyTorch, as the runner has no GPU), runs `tests/smoke_installed.py` and the test
suite against the installed packages. **Releasing:** change `__version__` in `tdplay/__init__.py` and push to `main`
(or push a `v*` tag): if that version has no release yet, the workflow publishes one with both installers once
everything passed. *Run workflow* on GitHub builds and tests them without releasing.

Layout: `tdplay/engine.py` (Mac segment chaining, joins, resume) · `tdplay/engine_torch.py` and
`tdplay/ltx_torch.py` (Windows: the whole shot as one LTX-2 run) · `tdplay/media.py` (ffmpeg I/O) ·
`tdplay/models.py` (presets, downloads) · `tdplay/upscale.py` (Real-ESRGAN on MLX or PyTorch) · `tdplay/app.py`
(Gradio UI) · `tdplay/worker.py` (background jobs) · `tdplay/desktop_app.py` (app window, Mac and Windows) ·
`tdplay/macos_bundle.py` (the .app) · `macos/` (disk image and its first-launch setup) · `windows/install.ps1`
(Windows setup) · `windows/installer/` (Setup.exe) · `scripts/make_icon.py` (the icon).
