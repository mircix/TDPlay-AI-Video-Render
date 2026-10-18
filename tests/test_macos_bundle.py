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
#: The launchers show macOS alerts that wait for a click; the tests run without them.
NO_DIALOGS = {"TDPLAY_NO_DIALOGS": "1"}


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
        subprocess.run([bash, str(launcher)], check=True, env={"HOME": str(home), "PATH": os.environ["PATH"], **NO_DIALOGS})
        cwd, *args = (home / "launched.txt").read_text().splitlines()
        assert Path(cwd) == root and args == ["-m", "tdplay.desktop_app"]
        assert "---" in (home / "TDPlay" / "logs" / "app.log").read_text()


def test_launcher_explains_a_moved_folder(tmp_path):
    root = fake_install(tmp_path)
    app = macos_bundle.build(tmp_path / "Applications", root)
    shutil.rmtree(root / ".venv")
    result = subprocess.run(["bash", str(app / "Contents" / "MacOS" / "TDPlay")],
                            env={"HOME": str(tmp_path), "PATH": os.environ["PATH"], **NO_DIALOGS})
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


# -- the disk-image app ---------------------------------------------------------------------------------

FAKE_SETUP = """#!/bin/bash
# Stand-in for macos/setup.sh: makes a runtime with a stand-in Python, or fails like a dropped connection.
set -eu
echo "setup ${1} ${2}" >> "${HOME}/setup-runs.txt"
if [[ -f "${HOME}/fail" ]]; then echo "ERROR: Could not download uv. Check the internet connection."; exit 1; fi
mkdir -p "${2}/venv/bin" "${2}/app"
printf '#!/bin/bash\\nprintf "%%s\\\\n" "$(pwd)" "$@" > "${HOME}/launched.txt"\\n' > "${2}/venv/bin/python"
chmod 755 "${2}/venv/bin/python"
cp "${1}/VERSION" "${2}/VERSION"
"""


def standalone_app(tmp_path: Path) -> Path:
    app = macos_bundle.build(tmp_path / "Applications", ROOT, standalone=True, build_id="abc123")
    (app / "Contents" / "Resources" / "app" / "macos" / "setup.sh").write_text(FAKE_SETUP)
    return app


def test_standalone_app_carries_the_code(tmp_path):
    app = macos_bundle.build(tmp_path / "Applications", ROOT, standalone=True, build_id="abc123")
    res = app / "Contents" / "Resources"
    assert (res / "app" / "tdplay" / "worker.py").exists() and (res / "app" / "requirements.txt").exists()
    assert (res / "app" / "VERSION").read_text().strip() == f"{macos_bundle.__version__}+abc123"
    assert not list((res / "app").rglob("__pycache__"))
    assert (res / "AppIcon.icns").exists()
    for script in (app / "Contents" / "MacOS" / "TDPlay", ROOT / "macos" / "setup.sh", ROOT / "macos" / "build_dmg.sh"):
        text = script.read_text()
        assert text.isascii(), script
        assert not re.findall(r"\$[A-Za-z_]", text), script  # every variable braced (bash 3.2)


def test_standalone_app_sets_itself_up_once_then_starts(tmp_path):
    app = standalone_app(tmp_path)
    launcher = app / "Contents" / "MacOS" / "TDPlay"
    bashes = ["bash"] + ([os.environ["TDPLAY_BASH32"]] if os.environ.get("TDPLAY_BASH32") else [])
    for i, bash in enumerate(bashes):
        home = tmp_path / f"home{i}"
        home.mkdir()
        env = {"HOME": str(home), "PATH": os.environ["PATH"], **NO_DIALOGS}
        runtime = home / "TDPlay" / "runtime"
        for _ in range(2):  # the second launch goes straight to the app
            subprocess.run([bash, str(launcher)], check=True, env=env)
            cwd, *args = (home / "launched.txt").read_text().splitlines()
            assert Path(cwd) == runtime / "app" and args == ["-m", "tdplay.desktop_app"]
        assert len((home / "setup-runs.txt").read_text().splitlines()) == 1
        # An update (a new build of the app) sets up again.
        (app / "Contents" / "Resources" / "app" / "VERSION").write_text("9.9.9\n")
        subprocess.run([bash, str(launcher)], check=True, env=env)
        assert len((home / "setup-runs.txt").read_text().splitlines()) == 2
        assert (runtime / "VERSION").read_text() == "9.9.9\n"
        assert not (home / "TDPlay" / ".setup-lock").exists()
        (app / "Contents" / "Resources" / "app" / "VERSION").write_text(f"{macos_bundle.__version__}+abc123\n")


def test_standalone_app_reports_a_failed_setup(tmp_path):
    app = standalone_app(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "fail").touch()
    env = {"HOME": str(home), "PATH": os.environ["PATH"], **NO_DIALOGS}
    result = subprocess.run(["bash", str(app / "Contents" / "MacOS" / "TDPlay")], env=env)
    assert result.returncode == 1
    assert "ERROR: Could not download uv" in (home / "TDPlay" / "logs" / "setup.log").read_text()
    assert not (home / "launched.txt").exists() and not (home / "TDPlay" / ".setup-lock").exists()
    (home / "fail").unlink()  # the next launch tries again
    subprocess.run(["bash", str(app / "Contents" / "MacOS" / "TDPlay")], check=True, env=env)
    assert (home / "launched.txt").exists()
