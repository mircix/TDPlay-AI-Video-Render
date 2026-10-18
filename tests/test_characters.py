"""The character library (``tdplay.characters``) and its tab."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from tdplay import characters, models
from test_scene import BLUE, RED, photo


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "HOME", tmp_path / "home")


def test_save_edit_and_delete_characters(tmp_path):
    assert characters.everyone() == []
    big = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), RED).save(big)
    mia = characters.save("  Mia ", "black lace slip dress,\n choker", str(big))
    assert (mia.name, mia.notes, mia.line) == ("Mia", "black lace slip dress, choker", "Mia: black lace slip dress, choker")
    with Image.open(mia.photo) as kept:
        assert kept.size == (2048, 1536)  # kept no bigger than needed
    leo = characters.save("Leo", "", photo(tmp_path, "leo.png", BLUE))
    assert [c.name for c in characters.everyone()] == ["Leo", "Mia"] and leo.line == "Leo"
    another = characters.save("Mia", "red coat", photo(tmp_path, "mia2.png", RED))  # same name: its own folder
    assert another.folder != mia.folder and len(characters.everyone()) == 3

    changed = characters.save("Mia", "white sundress", None, mia.folder)  # new notes, same photo
    assert changed.folder == mia.folder and changed.notes == "white sundress" and changed.photo.exists()
    for bad, why in (("", "name"), ("Mia: x", "':'")):
        with pytest.raises(ValueError, match=why):
            characters.save(bad, "", photo(tmp_path, "x.png", RED))
    with pytest.raises(ValueError, match="photo"):
        characters.save("Nobody", "", None)
    characters.delete(another.folder)
    characters.delete(tmp_path)  # not a character folder: left alone
    assert [c.name for c in characters.everyone()] == ["Leo", "Mia"] and tmp_path.exists()


def test_a_characters_line_in_the_characters_box(tmp_path):
    mia = characters.save("Mia", "white sundress", photo(tmp_path, "mia.png", RED))
    assert characters.with_line("", mia) == "Mia: white sundress"
    assert characters.with_line("Leo: grey suit\n\nmia: old dress", mia) == "Leo: grey suit\nMia: white sundress"
    assert characters.with_line("Leo: grey suit", mia) == "Leo: grey suit\nMia: white sundress"


def test_the_characters_tab(tmp_path):
    from tdplay import app

    _, picture, folder, note = app.save_character("Mia", photo(tmp_path, "mia.png", RED), "white sundress", None)
    assert note == "Saved **Mia**." and picture.endswith("photo.png")
    app.save_character("Leo", photo(tmp_path, "leo.png", BLUE), "grey suit", None)
    gallery = app.character_gallery()
    assert [name for _, name in gallery] == ["Leo", "Mia"]
    name, picture, notes, picked, _ = app.pick_character(SimpleNamespace(index=1))
    assert (name, notes, picked) == ("Mia", "white sundress", folder)
    assert app.save_character("", None, "", None)[-1] == "⚠️ Give the character a name."

    picture, box, _, note = app.start_with_character(folder, "Leo: grey suit")
    assert picture == str(characters.load(folder).photo) and box == "Leo: grey suit\nMia: white sundress"
    leo = str(characters.everyone()[0].folder)
    slots = app.add_character_to_scene(leo, None, None, None, "", "", "")
    pictures, notes, note = slots[:3], slots[3:6], slots[-1]
    assert pictures[0].endswith("photo.png") and notes == ["Leo: grey suit", "", ""] and "character 1" in note
    full = app.add_character_to_scene(leo, "a", "b", "c", "", "", "")
    assert "3 characters already" in full[-1]
    *_, note = app.delete_character(folder)
    assert note == "Deleted **Mia**." and [name for _, name in app.character_gallery()] == ["Leo"]
    assert "first" in app.start_with_character(None, "")[-1]
