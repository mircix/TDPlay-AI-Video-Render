"""LoRA add-ons (``tdplay.loras``): which files can be used, and where they come from."""

from __future__ import annotations

from pathlib import Path

import pytest

from tdplay import loras, models

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402


def lora_file(path: Path, *, blocks=range(48), rank=32, prefix="diffusion_model.", naming=("lora_A", "lora_B")):
    """A small LoRA in the given layout (ComfyUI's by default) on two layers of each block."""
    tensors = {}
    for i in blocks:
        for layer in ("attn1.to_q", "ff.net.2"):
            tensors[f"{prefix}transformer_blocks.{i}.{layer}.{naming[0]}.weight"] = torch.zeros(rank, 8)
            tensors[f"{prefix}transformer_blocks.{i}.{layer}.{naming[1]}.weight"] = torch.zeros(8, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return path


def test_lora_files_are_checked_before_use(tmp_path):
    good = loras.inspect(lora_file(tmp_path / "LTX2.3-IC-LORA-Dual-Character.safetensors"))
    assert not good.problem and (good.pairs, good.blocks, good.rank) == (96, 48, 32)
    assert "rank 32, changes 96 layers in 48 of 48 blocks" in good.describe()
    assert not loras.inspect(lora_file(tmp_path / "plain.safetensors", prefix="")).problem  # no ComfyUI prefix
    kohya = loras.inspect(lora_file(tmp_path / "kohya.safetensors", naming=("lora_down", "lora_up")))
    assert "lora_down/lora_up" in kohya.problem
    assert "bigger model" in loras.inspect(lora_file(tmp_path / "big.safetensors", blocks=range(60))).problem
    distilled = loras.inspect(lora_file(tmp_path / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"))
    assert "isn't needed" in distilled.problem
    (tmp_path / "broken.safetensors").write_bytes(b"not a model")
    assert "Can't read" in loras.inspect(tmp_path / "broken.safetensors").problem


def test_the_loras_to_pick_from(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HOME", tmp_path / "home")
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path / "home" / "models")
    assert loras.choices() == []
    omni = loras.ADDONS[0]
    lora_file(omni.path)  # downloaded with the app
    mine = lora_file(loras.folder() / "mine.safetensors")  # copied into the LoRA folder
    elsewhere = lora_file(tmp_path / "ComfyUI" / "models" / "loras" / "ltx2" / "dual.safetensors")  # added in place
    picked = loras.choices([str(elsewhere), str(tmp_path / "gone.safetensors")])
    assert [path for _, path in picked] == [str(omni.path), str(mine), str(elsewhere)]
    assert picked[0][0].startswith("OmniNFT RL") and picked[2][0] == "dual.safetensors"
    assert loras.suggested_strength(omni.path) == 0.9 and loras.suggested_strength(mine) == 1.0
