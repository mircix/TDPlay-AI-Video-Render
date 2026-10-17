"""Stand-in for ``python -m tdplay.ltx_torch JOB.json``: same progress lines, frame-numbered output pieces.

Like the real decoder, a segment can arrive as two pieces (body, then the blended tail), so the
runner must follow frames, not pieces. ``FAKE_LTX_FAIL_ON=<segment number>`` makes that segment
crash; ``FAKE_LTX_SLEEP`` slows the steps down.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_ltx  # noqa: E402
from tdplay import media  # noqa: E402

job = json.loads(Path(sys.argv[1]).read_text())
out = Path(job["chunks_dir"])
fps, w, h = job["fps"], job["width"], job["height"]
sizes = [job["first_frames"]] + job["new_frames"]
total = sum(sizes)
base = fake_ltx.base_color(job["seed"])
delay = float(os.environ.get("FAKE_LTX_SLEEP", "0"))
print("GPU: Fake Radeon RX 7900 XTX", flush=True)
done, piece = 0, 0
for i, size in enumerate(sizes):
    print(f"SEGMENT {i + 1}/{len(sizes)}", flush=True)
    for total_steps in (8, 3):
        for s in range(total_steps + 1):
            print(f"\r{100 * s // total_steps:3d}%|##| {s}/{total_steps} [00:00<00:00]", end="", file=sys.stderr, flush=True)
            time.sleep(delay)
        print(file=sys.stderr)
    if os.environ.get("FAKE_LTX_FAIL_ON") == str(i + 1):
        raise SystemExit("HIP out of memory. Tried to allocate 2.00 GiB")
    parts = [size] if size < 40 else [size - 24, 24]  # body + blended tail
    for part in parts:
        frames = np.stack([fake_ltx.render(done + k, w, h, base) for k in range(part)])
        audio = fake_ltx.tone(done, part, fps)
        media.write_video(out / f"piece_{piece:03d}.mp4", frames, fps)
        media.write_wav(out / f"piece_{piece:03d}.wav", audio)
        done += part
        piece += 1
        print(f"FRAMES {done}/{total}", flush=True)
