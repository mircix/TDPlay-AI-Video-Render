# TDPlay AI Video Render 🎬

**One continuous AI video shot, built segment by segment — free, private, on your Mac.**

TDPlay AI Video Render is a local take on RunningHub's *MINIMAX H3 Motion Context* workflow. You write one prompt per
segment. Each new segment continues the last second of the one before it, **picture and sound together**,
so the result plays as a single unbroken take of 30, 45, 60+ seconds instead of a set of cuts.

It runs the open-weights **LTX-2** video model from Lightricks (which makes video *with* audio) natively on Apple
Silicon through [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx). No account, no credits, no upload: after the
one-time model download everything runs offline on your Mac.

---

## What you need

| | |
|---|---|
| Mac | Apple Silicon: M1 or newer. Built with **M4 and M5** in mind. |
| macOS | 14 Sonoma or newer |
| Memory | 16 GB minimum, **32 GB or more recommended** |
| Disk | 30–45 GB for the model, plus space for your videos |
| Internet | only for installing and the one-time model download |

## Install (once)

1. Unzip `TDPlay-AI-Video-Render.zip` and move the `TDPlay-AI-Video-Render` folder somewhere permanent (for example your home folder).
2. Double-click **`install.command`**.
   * If macOS says it "cannot be opened because it is from an unidentified developer": right-click the file →
     **Open** → **Open**. Or open Terminal and run `bash ~/TDPlay-AI-Video-Render/install.command` (adjust the path).
   * The installer sets up [Homebrew](https://brew.sh) (it asks first), ffmpeg, Python and the model code. At the
     end it offers to download the recommended model for your Mac (30–45 GB). You can also do that later in
     the app's **Models** tab.
3. Double-click **`start.command`**. TDPlay AI Video Render opens in your browser at `http://127.0.0.1:7860`.
   Keep the Terminal window open while you use it; close it to quit.

## Make a long take

1. *(Optional)* Drop in a **character / first-frame photo**. Segment 1 starts from it.
   Or open *"…or continue a video you already have"* to extend one of your own clips.
2. Pick the number of **segments** and write **one prompt per segment**, each with a length in seconds.
3. Press **Generate**. The shot so far appears after every segment, so you can watch it grow.
4. The finished `render.mp4` is in the project folder (**Open project folder**).

If a run stops for any reason (you pressed Stop, the Mac ran out of memory, you quit), press **Resume**.
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
chips should be faster. `start.command` keeps the Mac from idle-sleeping while the app is open.

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
* **The full log** is `log.txt` in each project folder.

## Where things are

* Projects: `~/TDPlay/projects/<date>_<title>/render.mp4`
* Models: `~/TDPlay/models/` (delete a folder there to free its space)
* App code: this folder (`.venv/` and `vendor/` are created by the installer)

## Update or uninstall

* **Update**: replace this folder with a new version, then run `install.command` again. Models and projects stay in
  `~/TDPlay`.
* **Uninstall**: delete this folder and `~/TDPlay`. ffmpeg and uv can be removed with
  `brew uninstall ffmpeg uv`.

## Credits and licenses

* **LTX-2** by [Lightricks](https://huggingface.co/Lightricks), under the
  [LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE). Free to use, but
  organizations with annual revenue of $10M or more need a paid commercial license from Lightricks. It also includes
  an acceptable-use policy. Read it before any commercial use.
* **[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)** (MIT) by dgrauet: the MLX port and the pre-converted weights
  (`dgrauet/ltx-2.x-mlx-*` on Hugging Face). TDPlay AI Video Render pins release `v0.16.3`.
* **Gemma 3** (`mlx-community/gemma-3-12b-it-4bit`) is the text encoder for the LTX-2.3 packs, under the Gemma terms.
* **[Gradio](https://gradio.app)** (Apache-2.0) for the interface, **[ffmpeg](https://ffmpeg.org)** for video.
* "RunningHub", "MiniMax" and "H3" are named only to describe the workflow TDPlay AI Video Render recreates. This project is not
  affiliated with them.

## For developers

```bash
.venv/bin/python -m tdplay                 # start the UI (TDPLAY_PORT, TDPLAY_NO_BROWSER, TDPLAY_HOME)
.venv/bin/python -m tdplay download        # download the recommended model (or: download ltx-2.3-q4)
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest tests             # runs the whole pipeline against a fake ltx-2-mlx (no GPU needed)
```

`tests/fake_ltx.py` stands in for the `ltx-2-mlx` CLI. It writes frame numbers into the pixels and a continuous
tone into the audio, so the tests can prove there are no dropped or duplicated frames and no audio gaps at the
joins. The tests also check every command TDPlay AI Video Render builds against the real `ltx-2-mlx` argument parser, and cover
resuming after an out-of-memory kill, stop/resume and continuing an existing video.

Layout: `tdplay/engine.py` (segment chaining, joins, resume) · `tdplay/media.py` (ffmpeg I/O) ·
`tdplay/models.py` (presets, downloads) · `tdplay/app.py` (Gradio UI).
