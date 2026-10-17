"""Model packs: which LTX-2 weights suit which Mac, and downloading only the files this app uses.

The MLX packs on Hugging Face bundle every transformer variant (dev, distilled, LoRAs, extra
upscalers). TDPlay only runs the distilled pipeline, so it downloads the distilled transformer,
the text encoder, the VAEs, the audio decoder and the 2x spatial upscaler, and skips the rest. That
roughly halves the download.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

HOME = Path(os.environ.get("TDPLAY_HOME", Path.home() / "TDPlay"))
MODELS_DIR = HOME / "models"

GEMMA3 = "mlx-community/gemma-3-12b-it-4bit"


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    repo: str
    #: LTX-2.3 packs need a separate Gemma 3 text encoder; LTX-2.5 packs bundle their own Gemma 4.
    gemma_repo: str | None
    min_ram_gb: int


PRESETS: dict[str, Preset] = {
    p.key: p
    for p in [
        Preset("ltx-2.5-q8", "LTX-2.5 · 8-bit — best quality (32 GB+ Macs)", "dgrauet/ltx-2.5-mlx-q8", None, 32),
        Preset("ltx-2.3-q8", "LTX-2.3 · 8-bit — for 16–24 GB Macs", "dgrauet/ltx-2.3-mlx-q8", GEMMA3, 16),
        Preset("ltx-2.3-q4", "LTX-2.3 · 4-bit — smallest download", "dgrauet/ltx-2.3-mlx-q4", GEMMA3, 16),
    ]
}

# Big weight files the distilled pipeline never loads.
_UNUSED_WEIGHTS = ("transformer-dev", "lora", "temporal_upscaler", "spatial_upscaler_x1_5")


# ---------------------------------------------------------------------------
# This Mac
# ---------------------------------------------------------------------------


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def system_ram_gb() -> int:
    if sys.platform == "darwin" and (mem := _sysctl("hw.memsize")):
        return round(int(mem) / 2**30)
    return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)


def chip_name() -> str:
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    return f"{sys.platform} (not a Mac — generation needs Apple Silicon)"


@dataclass(frozen=True)
class Recommendation:
    preset: str
    low_ram: bool
    quality: str
    max_segment_seconds: int


def recommend(ram_gb: int) -> Recommendation:
    """Defaults by unified memory, from the ltx-2-mlx measurements (see README)."""
    if ram_gb >= 96:
        return Recommendation("ltx-2.5-q8", low_ram=False, quality="704p", max_segment_seconds=8)
    if ram_gb >= 48:
        return Recommendation("ltx-2.5-q8", low_ram=True, quality="704p", max_segment_seconds=6)
    if ram_gb >= 32:
        return Recommendation("ltx-2.5-q8", low_ram=True, quality="576p", max_segment_seconds=5)
    return Recommendation("ltx-2.3-q8", low_ram=True, quality="512p", max_segment_seconds=4)


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def local_dir(repo: str) -> Path:
    return MODELS_DIR / repo.replace("/", "--")


def _manifest(repo: str) -> Path:
    return MODELS_DIR / f"{repo.replace('/', '--')}.manifest"


def plan_files(repo: str) -> tuple[str, list[tuple[str, int]]]:
    """The revision and ``(filename, size)`` list TDPlay needs from ``repo`` (asks the Hub)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings if not s.rfilename.startswith(".")]
    names = [name for name, _ in files]
    # ltx-2-mlx loads the newest versioned distilled transformer when there is one.
    versioned = sorted(n for n in names if re.fullmatch(r"transformer-distilled-.+\.safetensors", n))
    unused = set(versioned[:-1])
    if versioned:
        unused.add("transformer-distilled.safetensors")
    keep = [
        (name, size)
        for name, size in files
        if name not in unused and not (name.endswith(".safetensors") and any(k in name for k in _UNUSED_WEIGHTS))
    ]
    return info.sha, keep


def is_downloaded(repo: str) -> bool:
    manifest = _manifest(repo)
    if not manifest.exists():
        return False
    root = local_dir(repo)
    for line in manifest.read_text().splitlines()[1:]:
        name, _, size = line.rpartition("\t")
        path = root / name
        if not path.exists() or path.stat().st_size != int(size):
            return False
    return True


def repos_for(preset: Preset) -> list[str]:
    return [preset.repo] + ([preset.gemma_repo] if preset.gemma_repo else [])


def preset_ready(preset: Preset) -> bool:
    return all(is_downloaded(r) for r in repos_for(preset))


def _bytes_on_disk(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


@dataclass
class DownloadProgress:
    repo: str
    done_bytes: int
    total_bytes: int
    finished: bool = False
    #: Bytes already on disk when this download started (a resumed download).
    initial_bytes: int = 0

    @property
    def fraction(self) -> float:
        return min(1.0, self.done_bytes / self.total_bytes) if self.total_bytes else 0.0


def download(repo: str, *, poll_seconds: float = 2.0) -> Iterator[DownloadProgress]:
    """Download the files TDPlay needs from ``repo`` into :func:`local_dir`, yielding progress."""
    from huggingface_hub import snapshot_download

    revision, files = plan_files(repo)
    total = sum(size for _, size in files)
    dest = local_dir(repo)
    dest.mkdir(parents=True, exist_ok=True)
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            snapshot_download(repo, revision=revision, local_dir=dest, allow_patterns=[n for n, _ in files])
        except BaseException as exc:  # surfaced to the caller below
            error.append(exc)

    initial = min(_bytes_on_disk(dest), total)
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while thread.is_alive():
        yield DownloadProgress(repo, min(_bytes_on_disk(dest), total), total, initial_bytes=initial)
        thread.join(poll_seconds)
    if error:
        raise error[0]
    lines = [revision] + [f"{name}\t{(dest / name).stat().st_size}" for name, _ in files]
    _manifest(repo).write_text("\n".join(lines) + "\n")
    yield DownloadProgress(repo, total, total, finished=True, initial_bytes=initial)


def download_size_gb(preset: Preset, *, missing_only: bool = False) -> float:
    total = 0
    for repo in repos_for(preset):
        if missing_only and is_downloaded(repo):
            continue
        total += sum(size for _, size in plan_files(repo)[1])
    return total / 1e9


def cli_download(args: list[str]) -> None:
    """``python -m tdplay download [preset]``: download a model pack from the terminal."""
    key = args[0] if args else recommend(system_ram_gb()).preset
    if key not in PRESETS:
        raise SystemExit(f"Unknown model {key!r}. Choose one of: {', '.join(PRESETS)}")
    preset = PRESETS[key]
    print(f"Downloading {preset.label} (about {download_size_gb(preset, missing_only=True):.0f} GB) to {MODELS_DIR}")
    for repo in repos_for(preset):
        if is_downloaded(repo):
            print(f"  {repo}: already downloaded")
            continue
        started = time.time()
        for progress in download(repo):
            print(f"\r  {wait_message(progress, started)}   ", end="", flush=True)
        print()
    print("Done. Start TDPlay AI Video Render with start.command")


def wait_message(progress: DownloadProgress, started: float) -> str:
    gb_done, gb_total = progress.done_bytes / 1e9, progress.total_bytes / 1e9
    elapsed = max(1e-6, time.time() - started)
    rate = max(0, progress.done_bytes - progress.initial_bytes) / elapsed / 1e6
    return f"{progress.repo}: {gb_done:.1f} / {gb_total:.1f} GB ({rate:.0f} MB/s)"
