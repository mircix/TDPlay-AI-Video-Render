"""Shot runner for the PyTorch backend (Windows/Linux): the whole shot in one LTX-2 run.

``tdplay.ltx_torch`` renders every segment as one chunk of LTX-2's own long-video mode, so the
picture-and-sound carry between segments happens inside the model run. This runner writes the
job, follows the progress, previews the shot after every segment and assembles the result. It
shares stopping, upscaling and error handling with the Mac runner (:class:`tdplay.engine.Runner`).

Resume starts the shot again from the beginning: the carry between segments lives inside the
model run, so there is nothing on disk to continue from.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

from tdplay import media
from tdplay.engine import (
    DISTILLED_STEPS,
    Event,
    Runner,
    _StepCounter,
    extend_latent_frames,
    first_segment_frames,
)
from tdplay.ltx_torch import segment_ends

_SEGMENT = re.compile(r"SEGMENT (\d+)/(\d+)")
_FRAMES = re.compile(r"FRAMES (\d+)/(\d+)")


def default_torch_command() -> list[str]:
    override = os.environ.get("TDPLAY_TORCH_CMD")
    return shlex.split(override) if override else [sys.executable, "-m", "tdplay.ltx_torch"]


def find_checkpoint(model_dir: str) -> str:
    """The distilled LTX-2 checkpoint in ``model_dir`` (the newest if there are several)."""
    found = sorted(Path(model_dir).glob("*distilled*.safetensors"))
    if not found:
        raise FileNotFoundError(f"No distilled LTX-2 checkpoint in {model_dir}. Download the model in the Models tab.")
    return str(found[-1])


class TorchRunner(Runner):
    def __init__(self, project, torch_command: list[str] | None = None, upscale_command: list[str] | None = None):
        super().__init__(project, upscale_command=upscale_command)
        self.torch_command = torch_command or default_torch_command()

    def job(self) -> dict:
        p, s = self.project, self.project.settings
        first = first_segment_frames(p.segments[0].seconds, s.fps)
        new = [8 * extend_latent_frames(seg.seconds, s.fps) for seg in p.segments[1:]]
        images = []
        if p.reference_image:
            images.append({"path": str(p.path(p.reference_image)), "frame": 0, "strength": 1.0})
            if s.anchor_strength > 0:  # re-anchor the photo on the last frame of every later segment
                images += [{"path": str(p.path(p.reference_image)), "frame": end - 1, "strength": s.anchor_strength}
                           for end in segment_ends(first, new)[1:]]
        return {
            "chunks_dir": str(p.path("chunks")),
            "fps": s.fps, "width": s.width, "height": s.height, "seed": p.seeds[0],
            "prompts": [seg.prompt.strip() for seg in p.segments],
            "first_frames": first, "new_frames": new, "images": images,
            "checkpoint": find_checkpoint(s.model_dir) if s.model_dir else "",
            "gemma_root": s.gemma_dir, "upsampler": s.upsampler, "offload": s.offload,
        }

    def _stopped_text(self) -> str:
        return "Stopped. Resume renders the shot again from the start (finished segments can't be kept yet on Windows)."

    def _pieces(self) -> list[Path]:
        return sorted(self.project.path("chunks").glob("piece_[0-9][0-9][0-9].mp4"))

    def _assemble_pieces(self, out: Path) -> Path:
        videos = self._pieces()
        media.assemble(videos, [v.with_suffix(".wav") for v in videos], out)
        return out

    def _render(self):
        p = self.project
        n = len(p.segments)
        if p.done < n:  # a fresh run, or an interrupted one: start the shot over
            for old in p.path("chunks").glob("piece_*"):
                old.unlink()
            p.done = 0
            p.save()
            job = self.job()
            job_path = p.path("job.json")
            job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
            ends = segment_ends(job["first_frames"], job["new_frames"])
            env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       PYTORCH_HIP_ALLOC_CONF="expandable_segments:True",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            steps, current = _StepCounter(), 0
            yield Event("progress", f"Segment 1 of {n} · loading the model", fraction=0.0)
            for line in self._process(self.torch_command + [str(job_path)], env, "Rendering"):
                if m := _SEGMENT.match(line):
                    current = int(m.group(1)) - 1
                    steps.reset()
                    yield Event("progress", f"Segment {current + 1} of {n}", fraction=current / n)
                elif (done := steps.update(line)) is not None:
                    within = 0.05 + 0.85 * done / DISTILLED_STEPS
                    yield Event("progress", f"Segment {current + 1} of {n} · denoising step {done}/{DISTILLED_STEPS}",
                                fraction=(current + within) / n)
                elif m := _FRAMES.match(line):
                    decoded = int(m.group(1))
                    finished = sum(1 for end in ends if decoded >= end)
                    if finished > p.done:
                        p.done = finished
                        p.save()
                        yield Event("preview", f"Segment {finished} of {n} done.", fraction=finished / n,
                                    path=str(self._assemble_pieces(p.preview)))
                else:
                    yield Event("log", line)
            if p.done < n:  # the model run ended without decoding the whole shot
                p.done = n
                p.save()
        return self._assemble_pieces(p.output)
