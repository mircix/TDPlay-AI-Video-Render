"""Windows installer and launchers: plain ASCII with CRLF, and (with PowerShell available) parse + GPU picks."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [ROOT / "install.bat", ROOT / "start.bat", ROOT / "windows" / "install.ps1",
           ROOT / "windows" / "installer" / "TDPlay.iss", ROOT / "windows" / "installer" / "before-install.txt"]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_ascii_with_crlf(script):
    data = script.read_bytes()
    assert data.isascii(), "Windows PowerShell 5.1 reads scripts without a BOM as ANSI: keep them ASCII"
    assert b"\n" not in data.replace(b"\r\n", b""), "cmd.exe and PowerShell 5.1 expect CRLF line endings"


def _pwsh() -> str | None:
    return os.environ.get("TDPLAY_PWSH") or shutil.which("pwsh")


@pytest.mark.skipif(not _pwsh(), reason="PowerShell not installed (set TDPLAY_PWSH)")
def test_installer_parses_and_picks_the_right_rocm_build():
    script = ROOT / "windows" / "install.ps1"
    cases = {
        "AMD Radeon RX 7900 XTX": "amd gfx1100",
        "AMD Radeon RX 7900 GRE": "amd gfx1100",
        "AMD Radeon RX 7800 XT": "amd gfx1101",
        "AMD Radeon RX 7600": "amd gfx1102",
        "AMD Radeon RX 9070 XT": "amd gfx1201",
        "AMD Radeon RX 9060 XT": "amd gfx1200",
        "AMD Radeon(TM) Graphics|AMD Radeon RX 9070": "amd gfx1201",  # skips the Ryzen iGPU
        "NVIDIA GeForce RTX 4090": "nvidia ",
        "AMD Radeon RX 6800 XT": "amd all",
        "Intel(R) UHD Graphics": "none",
    }
    lines = [
        "$errors = $null; $tokens = $null",
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{script}', [ref]$tokens, [ref]$errors)",
        "if ($errors.Count) { 'PARSE ERRORS'; $errors | ForEach-Object { $_.Message } }",
        f". '{script}' -DefineOnly",
    ]
    for names in cases:
        quoted = ", ".join(f"'{n}'" for n in names.split("|"))
        lines.append(f"$t = Get-GpuTarget @({quoted}); if ($t) {{ '{{0}} {{1}}' -f $t.Kind, $t.Device }} else {{ 'none' }}")
    out = subprocess.run([_pwsh(), "-NoProfile", "-Command", "\n".join(lines)], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert "PARSE ERRORS" not in out
    assert out == list(cases.values())
