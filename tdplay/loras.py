"""Add-ons (LoRAs) for the Windows version: small extra weights that change how LTX-2 renders.

A LoRA adds a low-rank change to some of the transformer's weights. LTX-2 folds it into each block as
the block is copied to the GPU (``ltx_core.block_streaming.provider``), so it costs little memory and no
extra steps. The app offers two from Hugging Face to download, and uses any other LTX-2.3 LoRA file the
user adds (the ComfyUI format that Civitai and Hugging Face use: ``diffusion_model.transformer_blocks.N.
<layer>.lora_A/lora_B.weight``). The Mac version has none: ltx-2-mlx takes LoRAs only for a shot's first
segment (``generate``), not for the segments that continue it (``extend``).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

from tdplay import models

#: How many LoRAs a shot can use at once.
MAX_LORAS = 4
#: LTX-2.3's transformer blocks.
BLOCKS = 48


@dataclass(frozen=True)
class Addon:
    """A LoRA the app knows and can download."""

    name: str
    repo: str
    file: str
    #: The strength to start from.
    strength: float
    what: str
    megabytes: int

    @property
    def path(self) -> Path:
        return models.local_dir(self.repo) / self.file


ADDONS = [
    Addon("OmniNFT RL", "Kijai/LTX2.3_comfy", "loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors", 0.9,
          "sound in sync with the picture, lip sync, steadier video", 617),
    Addon("VBVR I2V", "chfm/Ltx2.3-VBVR-lora-I2V", "Ltx2.3-Licon-VBVR-I2V-390K-R32.safetensors", 0.7,
          "livelier motion, following longer prompts", 554),
]


def folder() -> Path:
    """Where the user's own LoRA files can go (any other file can be added from where it is, too)."""
    return models.HOME / "models" / "loras"


def suggested_strength(path: str | Path) -> float:
    return next((a.strength for a in ADDONS if a.path == Path(path)), 1.0)


def choices(extra: list[str] = ()) -> list[tuple[str, str]]:
    """``(label, path)`` of every LoRA there is: downloaded add-ons, the LoRA folder, files added elsewhere."""
    found: list[tuple[str, str]] = []
    for addon in ADDONS:
        if addon.path.exists():
            found.append((f"{addon.name} ({addon.what})", str(addon.path)))
    files = sorted(folder().glob("*.safetensors")) if folder().is_dir() else []
    files += [Path(p) for p in extra if Path(p).is_file()]
    for path in files:
        if str(path) not in (p for _, p in found):
            found.append((path.name, str(path)))
    return found


@dataclass(frozen=True)
class Info:
    """What a LoRA file holds, from its header (no weights are read)."""

    megabytes: int
    #: lora_A/lora_B pairs on the transformer's blocks, after LTX-2's ComfyUI renaming.
    pairs: int
    blocks: int
    rank: int
    problem: str = ""

    def describe(self) -> str:
        if self.problem:
            return self.problem
        return f"{self.megabytes} MB, rank {self.rank}, changes {self.pairs} layers in {self.blocks} of {BLOCKS} blocks"


def inspect(path: str | Path) -> Info:
    """Check that LTX-2 can load ``path`` as an LTX-2.3 LoRA, from the safetensors header."""
    path = Path(path)
    if "distilled-lora" in path.name.lower() or "distilled_lora" in path.name.lower():
        return Info(round(path.stat().st_size / 1e6) if path.exists() else 0, 0, 0, 0,
                    f"⚠️ {path.name} isn't needed: the app already runs the distilled model, so it would be applied "
                    "twice.")
    try:
        with open(path, "rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            if not 2 <= length <= min(path.stat().st_size - 8, 100_000_000):
                raise ValueError("it isn't a safetensors file")
            header = json.loads(f.read(length))
    except (OSError, ValueError, struct.error) as exc:
        return Info(0, 0, 0, 0, f"⚠️ Can't read {path.name}: {exc}")
    size = round(path.stat().st_size / 1e6)
    header.pop("__metadata__", None)
    a, b, ranks, blocks = set(), set(), set(), set()
    for key, value in header.items():
        name = key.removeprefix("diffusion_model.")  # LTX-2's LTXV_LORA_COMFY_RENAMING_MAP
        if not name.startswith("transformer_blocks."):
            continue
        if name.endswith(".lora_A.weight"):
            a.add(name.removesuffix(".lora_A.weight"))
            ranks.add(value["shape"][0])
        elif name.endswith(".lora_B.weight"):
            b.add(name.removesuffix(".lora_B.weight"))
    pairs = a & b
    blocks = {int(name.split(".")[1]) for name in pairs if name.split(".")[1].isdigit()}
    if not pairs:
        kohya = any("lora_down" in k or "lora_up" in k for k in header)
        why = ("it uses the lora_down/lora_up naming, which LTX-2 doesn't load" if kohya else
               "it has no LTX-2 transformer layers in it")
        return Info(size, 0, 0, 0, f"⚠️ {path.name} can't be used: {why}.")
    if max(blocks, default=0) >= BLOCKS:
        return Info(size, len(pairs), len(blocks), max(ranks), f"⚠️ {path.name} is for a bigger model than LTX-2.3.")
    return Info(size, len(pairs), len(blocks), max(ranks))
