"""Stand-in for ``python -m tdplay.ltx_torch JOB.json``: same progress lines, frame-numbered output pieces.

Like the real decoder, a segment can arrive as two pieces (body, then the blended tail), so the
runner must follow frames, not pieces. With a ``start_tail`` (a shot that continues the user's
video), the first chunk starts with a re-decode of those frames, color-shifted like a VAE round
trip, and every new frame has the same shift; the pieces then go through the real
``tdplay.ltx_torch.write_pieces``, which blends the seam. ``FAKE_LTX_FAIL_ON=<segment number>``
makes that segment crash; ``FAKE_LTX_SLEEP`` slows the steps down.
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
from tdplay import ltx_torch  # noqa: E402

job = json.loads(Path(sys.argv[1]).read_text())
fps, w, h = job["fps"], job["width"], job["height"]
sizes = [job["first_frames"]] + job["new_frames"]
total = sum(sizes)
base, first, bias, source = fake_ltx.base_color(job["seed"]), 0, 0, None
if job.get("start_tail"):  # continue the user's video: number on from its last frame, in its colors
    source = fake_ltx.read(job["start_tail"])[0]
    first = fake_ltx.read_index(source[-1]) + 1 - len(source)
    base, bias = source[-1][h // 4, w - 1].astype(int), fake_ltx.BIAS
delay = float(os.environ.get("FAKE_LTX_SLEEP", "0"))
print("GPU: Fake Radeon RX 7900 XTX", flush=True)


def frame(index: int) -> np.ndarray:
    """Frame ``index`` of the chunk timeline: a re-decode of the start video's frames, then new ones."""
    image = source[index] if source is not None and index < len(source) else fake_ltx.render(first + index, w, h, base)
    return np.clip(image.astype(int) + bias, 0, 255).astype(np.uint8)


def pieces():
    done = 0
    for i, size in enumerate(sizes):
        print(f"SEGMENT {i + 1}/{len(sizes)}", flush=True)
        for total_steps in (8, 3):
            for s in range(total_steps + 1):
                bar = "█" * (10 * s // total_steps)  # tqdm's own characters when its output is UTF-8
                print(f"\r{100 * s // total_steps:3d}%|{bar:<10}| {s}/{total_steps} [00:00<00:00]", end="",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
            print(file=sys.stderr)
        if os.environ.get("FAKE_LTX_FAIL_ON") == str(i + 1):
            raise SystemExit("HIP out of memory. Tried to allocate 2.00 GiB")
        parts = [size] if size < 40 else [size - 24, 24]  # body + blended tail
        for part in parts:
            yield np.stack([frame(done + k) for k in range(part)]), fake_ltx.tone(first + done, part, fps)
            done += part


ltx_torch.write_pieces(pieces(), job, total)
