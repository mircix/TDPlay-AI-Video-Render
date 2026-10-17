# TDPlay AI Video Render installer for Windows: AMD Radeon RX 7000 / RX 9000 (ROCm), or NVIDIA (CUDA).
# Run it through install.bat. It is plain ASCII on purpose (Windows PowerShell 5.1 reads scripts as ANSI).
# -DefineOnly loads the functions without installing anything (used by the tests).
param([switch]$DefineOnly)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with its progress bar
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LtxTag = "v1.4.2"                                  # Lightricks LTX-2 release this version is built against
$RocmIndex = "https://stable.repo.amd.com/rocm/whl-next/"
$TorchRocm = "2.13.0+rocm10.0.0"; $VisionRocm = "0.28.0+rocm10.0.0"; $AudioRocm = "2.11.0.2+rocm10.0.0"
$PyVersion = "3.13"

function Say([string]$Text) { Write-Host ""; Write-Host $Text -ForegroundColor Cyan }
function Fail([string]$Text) { Write-Host ""; Write-Host $Text -ForegroundColor Red; exit 1 }
function Check([string]$What) {
    # Call right after an external program: stop with a message if it failed.
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit code $LASTEXITCODE). See the messages above." }
}
function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                (Join-Path $env:USERPROFILE ".local\bin") + ";" +
                (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links")
}

function Get-GpuTarget([string[]]$Names) {
    # AMD ships ROCm PyTorch per GPU family; pick the one for the first supported card.
    foreach ($g in $Names) {
        if     ($g -match "RX 9070|AI PRO R9700") { return @{ Kind = "amd"; Device = "gfx1201"; Name = $g } }
        elseif ($g -match "RX 9060")               { return @{ Kind = "amd"; Device = "gfx1200"; Name = $g } }
        elseif ($g -match "RX 7900")               { return @{ Kind = "amd"; Device = "gfx1100"; Name = $g } }
        elseif ($g -match "RX 7800|RX 7700")       { return @{ Kind = "amd"; Device = "gfx1101"; Name = $g } }
        elseif ($g -match "RX 7600")               { return @{ Kind = "amd"; Device = "gfx1102"; Name = $g } }
        elseif ($g -match "NVIDIA|GeForce|RTX")    { return @{ Kind = "nvidia"; Device = $null; Name = $g } }
    }
    # Another discrete Radeon (not the "Radeon(TM) Graphics" built into Ryzen CPUs): all of AMD's kernels.
    $radeon = $Names | Where-Object { $_ -match "Radeon" -and $_ -notmatch "Radeon\(TM\) Graphics" } | Select-Object -First 1
    if ($radeon) { return @{ Kind = "amd"; Device = "all"; Name = $radeon } }
    return $null
}

if ($DefineOnly) { return }

Say "TDPlay AI Video Render installer"

# --- This PC ------------------------------------------------------------------------------
$build = [int](Get-CimInstance Win32_OperatingSystem).BuildNumber
if ($build -lt 22000) {
    Write-Host "AMD's PyTorch for Windows is made for Windows 11; this is build $build. Continuing anyway." -ForegroundColor Yellow
}
$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$cpu = ((Get-CimInstance Win32_Processor | Select-Object -First 1).Name).Trim()
$gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
Write-Host "$cpu, $ramGB GB RAM"
Write-Host ("Graphics: " + ($gpus -join ", "))
if ($ramGB -lt 16) { Fail "TDPlay AI Video Render needs at least 16 GB of RAM (32 GB recommended)." }

$target = Get-GpuTarget $gpus
if (-not $target) {
    Fail "No supported graphics card found. TDPlay needs an AMD Radeon RX 7000 / RX 9000 or an NVIDIA card."
}
$kind = $target.Kind; $device = $target.Device; $cardName = $target.Name
if ($device -eq "all") {
    Write-Host "$cardName is not one of the tested cards; installing AMD's packages for all GPUs." -ForegroundColor Yellow
}
if ($kind -eq "amd") { Write-Host "Using $cardName with AMD ROCm (device-$device)." }
else { Write-Host "Using $cardName with NVIDIA CUDA." }

$drive = (Get-Item $Root).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB)
if ($freeGB -lt 80) {
    Write-Host "Only $freeGB GB free on drive $($drive.Name):. The model needs about 55 GB plus room for videos." -ForegroundColor Yellow
}

# --- Tools: uv (installs Python) and ffmpeg ---------------------------------------------------
Update-SessionPath
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Installing uv (it installs and manages Python)..."
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    Check "Installing uv"
    Update-SessionPath
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Fail "winget is missing. Install 'App Installer' from the Microsoft Store, then run install.bat again."
    }
    Say "Installing ffmpeg..."
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Check "Installing ffmpeg"
    Update-SessionPath
}
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Fail "ffmpeg was installed but Windows can't find it yet. Sign out and back in, then run install.bat again."
}

# --- Lightricks LTX-2 ------------------------------------------------------------------------
$vendor = Join-Path $Root "vendor"
$ltxDir = Join-Path $vendor "LTX-2"
$marker = Join-Path $ltxDir ".tdplay-version"
if (-not ((Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $LtxTag))) {
    Say "Getting Lightricks LTX-2 $LtxTag..."
    New-Item -ItemType Directory -Force -Path $vendor | Out-Null
    if (Test-Path $ltxDir) { Remove-Item $ltxDir -Recurse -Force }
    $zip = Join-Path $env:TEMP "tdplay-ltx-2.zip"
    $unzip = Join-Path $env:TEMP "tdplay-ltx-2"
    Invoke-WebRequest -Uri "https://github.com/Lightricks/LTX-2/archive/refs/tags/$LtxTag.zip" -OutFile $zip -UseBasicParsing
    if (Test-Path $unzip) { Remove-Item $unzip -Recurse -Force }
    Expand-Archive -Path $zip -DestinationPath $unzip
    Move-Item (Get-ChildItem $unzip | Select-Object -First 1).FullName $ltxDir
    Set-Content -Path $marker -Value $LtxTag
    Remove-Item $zip, $unzip -Recurse -Force
}

# --- Python environment ------------------------------------------------------------------------
Say "Setting up Python $PyVersion..."
uv venv --allow-existing --seed --python $PyVersion .venv
Check "Creating the Python environment"
$py = Join-Path $Root ".venv\Scripts\python.exe"

Say "Installing PyTorch for your graphics card (a few GB)..."
if ($kind -eq "amd") {
    & $py -m pip install --index-url $RocmIndex "torch[device-$device]==$TorchRocm" "torchvision[device-$device]==$VisionRocm" "torchaudio==$AudioRocm"
    Check "Installing PyTorch (ROCm)"
} else {
    & $py -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
    Check "Installing PyTorch (CUDA)"
}

Say "Installing Lightricks LTX-2 and the app..."
& $py -m pip install --no-deps (Join-Path $ltxDir "packages\ltx-core") (Join-Path $ltxDir "packages\ltx-pipelines")
Check "Installing LTX-2"
& $py -m pip install -r (Join-Path $Root "windows\requirements-windows.txt")
Check "Installing the app's packages"

Say "Checking the graphics card..."
& $py -c "import torch; ok = torch.cuda.is_available(); print('PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); raise SystemExit(0 if ok else 1)"
Check "The GPU check (update the AMD Adrenalin driver and restart if PyTorch can't see the card)"
& $py -W ignore -c "import ltx_pipelines.distilled, tdplay.app; print('LTX-2 and the app load fine.')"
Check "Loading LTX-2 and the app"

# --- Shortcuts -----------------------------------------------------------------------------------
Say "Adding TDPlay AI Video Render to the Start menu and the desktop..."
$pyw = Join-Path $Root ".venv\Scripts\pythonw.exe"
$icon = Join-Path $Root "tdplay\assets\AppIcon.ico"
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Programs"), [Environment]::GetFolderPath("Desktop"))) {
    $link = $shell.CreateShortcut((Join-Path $dir "TDPlay AI Video Render.lnk"))
    $link.TargetPath = $pyw
    $link.Arguments = "-m tdplay.desktop_app"
    $link.WorkingDirectory = $Root
    $link.IconLocation = "$icon,0"
    $link.Description = "One continuous AI video shot, made on your PC"
    $link.Save()
}

# --- Model ---------------------------------------------------------------------------------------
Say "Model download"
Write-Host "TDPlay needs the LTX-2.3 model (about 55 GB). You can also do this later in the app's Models tab."
$answer = Read-Host "Download it now? [Y/n]"
if ($answer -notmatch "^[Nn]") {
    & $py -m tdplay download
    if ($LASTEXITCODE -ne 0) { Write-Host "The download stopped. Run install.bat again, or use the Models tab: it continues where it stopped." -ForegroundColor Yellow }
}

Say "All set! Open TDPlay AI Video Render from the Start menu or your desktop."
Write-Host "(start.bat also works: it opens the app in your browser, with a window showing the log.)"
