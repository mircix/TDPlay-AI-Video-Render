"""The .app bundle: structure, Info.plist, and a launcher that really starts the app (bash 3.2-safe)."""

import os
import plistlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tdplay import macos_bundle

ROOT = Path(__file__).resolve().parent.parent


def fake_install(tmp_path: Path) -> Path:
    """An install folder whose path has spaces and an apostrophe, with a stand-in Python."""
    root = tmp_path / "AI Video Workspace" / "Tom's TDPlay-AI-Video-Render"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / "tdplay" / "assets").mkdir(parents=True)
    shutil.copy2(ROOT / "tdplay" / "assets" / "AppIcon.icns", root / "tdplay" / "assets" / "AppIcon.icns")
    python = root / ".venv" / "bin" / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$(pwd)" "$@" > "${HOME}/launched.txt"\n')
    python.chmod(0o755)
    return root


def test_bundle_layout_and_plist(tmp_path):
    app = macos_bundle.build(tmp_path / "Applications", fake_install(tmp_path))
    assert app.name == "TDPlay AI Video Render.app"
    contents = app / "Contents"
    info = plistlib.loads((contents / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "TDPlay" and info["CFBundleIdentifier"] == macos_bundle.BUNDLE_ID
    assert info["CFBundleDisplayName"] == "TDPlay AI Video Render"
    assert (contents / "Resources" / "AppIcon.icns").stat().st_size > 10_000
    assert os.access(contents / "MacOS" / "TDPlay", os.X_OK)


def test_launcher_is_bash32_safe_and_starts_the_app(tmp_path):
    root = fake_install(tmp_path)
    app = macos_bundle.build(tmp_path / "Applications", root)
    launcher = app / "Contents" / "MacOS" / "TDPlay"
    text = launcher.read_text()
    assert text.isascii()
    assert not re.findall(r"\$[A-Za-z_]", text)  # every variable braced
    home = tmp_path / "home"
    home.mkdir()
    bashes = ["bash"] + ([os.environ["TDPLAY_BASH32"]] if os.environ.get("TDPLAY_BASH32") else [])
    for bash in bashes:
        subprocess.run([bash, "-n", str(launcher)], check=True)
        subprocess.run([bash, str(launcher)], check=True, env={"HOME": str(home), "PATH": os.environ["PATH"]})
        cwd, *args = (home / "launched.txt").read_text().splitlines()
        assert Path(cwd) == root and args == ["-m", "tdplay.desktop_app"]
        assert "---" in (home / "TDPlay" / "logs" / "app.log").read_text()


def test_launcher_explains_a_moved_folder(tmp_path):
    root = fake_install(tmp_path)
    app = macos_bundle.build(tmp_path / "Applications", root)
    shutil.rmtree(root / ".venv")
    result = subprocess.run(["bash", str(app / "Contents" / "MacOS" / "TDPlay")],
                            env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]})
    assert result.returncode == 1


def test_rebuild_replaces_only_our_own_bundle(tmp_path):
    dest = tmp_path / "Applications"
    root = fake_install(tmp_path)
    macos_bundle.build(dest, root)
    macos_bundle.build(dest, root)  # re-running install.command is fine
    other = dest / "TDPlay AI Video Render.app" / "Contents" / "Info.plist"
    other.write_bytes(plistlib.dumps({"CFBundleIdentifier": "com.someone.else"}))
    with pytest.raises(RuntimeError, match="not TDPlay's"):
        macos_bundle.build(dest, root)
