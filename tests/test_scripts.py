"""The .command launchers run under macOS's bash 3.2, which is stricter than a modern bash."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [ROOT / "install.command", ROOT / "start.command"]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_ascii_only(script):
    # macOS bash 3.2 reads a non-ASCII byte right after $name as part of the variable name
    # ("tool…" -> "tool?: unbound variable"), so the launchers stay plain ASCII.
    bad = [(n, line) for n, line in enumerate(script.read_text().splitlines(), 1) if not line.isascii()]
    assert not bad, bad


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_variables_are_braced(script):
    assert not re.findall(r"\$[A-Za-z_]", script.read_text())


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_syntax(script):
    if not shutil.which("bash"):
        pytest.skip("no bash")
    subprocess.run(["bash", "-n", str(script)], check=True)
