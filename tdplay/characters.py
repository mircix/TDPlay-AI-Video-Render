"""The character library: people saved once (name, photo, clothes and hair) and reused in any shot.

Each character is a folder in ``~/TDPlay/characters``: ``character.json`` (name and notes) and
``photo.png``. A character's notes become their line in a shot's Characters box ("Mia: black lace
slip dress..."), which reminds the model in every segment what they wear.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from tdplay import models

#: Photos are kept at most this big on their long side (they become a frame of a 1280x704 shot at most).
PHOTO_SIDE = 2048


@dataclass(frozen=True)
class Character:
    folder: Path
    name: str
    #: Clothes, hair, look: what the model should be reminded of.
    notes: str = ""

    @property
    def photo(self) -> Path:
        return self.folder / "photo.png"

    @property
    def line(self) -> str:
        """The character's line for a shot's Characters box."""
        return f"{self.name}: {self.notes}" if self.notes else self.name


def library() -> Path:
    return models.HOME / "characters"


def load(folder: str | Path) -> Character:
    folder = Path(folder)
    data = json.loads((folder / "character.json").read_text(encoding="utf-8"))
    return Character(folder, data["name"], data.get("notes", ""))


def everyone() -> list[Character]:
    """The saved characters, by name."""
    found = []
    for path in library().glob("*/character.json") if library().is_dir() else []:
        try:
            found.append(load(path.parent))
        except (OSError, ValueError, KeyError):
            continue
    return sorted(found, key=lambda c: c.name.lower())


def _folder_for(name: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "character"
    folder, n = library() / slug, 2
    while folder.exists():
        folder, n = library() / f"{slug}-{n}", n + 1
    return folder


def save(name: str, notes: str, photo: str | Path | None, folder: str | Path | None = None) -> Character:
    """Save a new character, or change the one in ``folder``. A new character needs a photo."""
    name, notes = " ".join(str(name or "").split()), " ".join(str(notes or "").split())
    if not name:
        raise ValueError("Give the character a name.")
    if ":" in name:
        raise ValueError("A name can't contain ':' (it separates the name from the notes in the Characters box).")
    folder = Path(folder) if folder else _folder_for(name)
    if not photo and not (folder / "photo.png").exists():
        raise ValueError("Add a photo of the character.")
    folder.mkdir(parents=True, exist_ok=True)
    if photo and Path(photo).resolve() != (folder / "photo.png").resolve():
        with Image.open(photo) as picture:
            picture = ImageOps.exif_transpose(picture).convert("RGB")
            picture.thumbnail((PHOTO_SIDE, PHOTO_SIDE), Image.Resampling.LANCZOS)
            picture.save(folder / "photo.png")
    (folder / "character.json").write_text(json.dumps({"name": name, "notes": notes}, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    return load(folder)


def delete(folder: str | Path) -> None:
    folder = Path(folder)
    if folder.parent == library() and (folder / "character.json").exists():
        shutil.rmtree(folder)


def with_line(box: str, character: Character) -> str:
    """The Characters box with ``character``'s line in it: replacing their earlier line, or added at the end."""
    lines = [line for line in str(box or "").splitlines() if line.strip()]
    mine = [i for i, line in enumerate(lines) if line.split(":", 1)[0].strip().lower() == character.name.lower()]
    if mine:
        lines[mine[0]] = character.line
    else:
        lines.append(character.line)
    return "\n".join(lines)
