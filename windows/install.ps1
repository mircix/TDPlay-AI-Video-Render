# TDPlay AI Video Render setup for Windows: AMD Radeon RX 7000 / RX 9000 (ROCm), or NVIDIA (CUDA).
# Run it through install.bat, or let the installer (TDPlay-AI-Video-Render-Setup.exe) run it.
# It is plain ASCII on purpose (Windows PowerShell 5.1 reads scripts as ANSI).
#   -FromInstaller  run by Setup.exe: it makes the shortcuts itself
#   -SkipModel      don't offer the model download     -NoPause  don't wait for Enter after an error
#   -TestCpu        PyTorch for the CPU, no GPU needed (only for testing the setup on machines without one)
#   -DefineOnly     load the functions without installing anything (used by the tests)
param([switch]$DefineOnly, [switch]$FromInstaller, [switch]$SkipModel, [switch]$NoPause, [switch]$TestCpu)

$ErrorActionPreference = "Stop"
# Started from PowerShell 7 (a terminal, or the installer launched from one), Windows PowerShell would
# inherit PowerShell 7's module folders and fail to load its own cmdlets: use its own folders.
if ($PSVersionTable.PSEdition -eq "Desktop") {
    $env:PSModulePath = @((Join-Path ([Environment]::GetFolderPath("MyDocuments")) "WindowsPowerShell\Modules"),
                          (Join-Path $env:ProgramFiles "WindowsPowerShell\Modules"),
                          (Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\Modules")) -join ";"
}
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with its progress bar
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$LtxTag = "v1.4.2"                                  # Lightricks LTX-2 release this version is built against
$RocmIndex = "https://stable.repo.amd.com/rocm/whl-next/"
$TorchRocm = "2.13.0+rocm10.0.0"; $VisionRocm = "0.28.0+rocm10.0.0"; $AudioRocm = "2.11.0.2+rocm10.0.0"
# Python itself: the official python.org build as published on NuGet (a portable zip), kept next to the
# app. Not uv's own Python: uv links its Pythons with a directory junction, which Windows refuses to
# follow in some setups ("untrusted mount point", error 448).
$PyVersion = "3.13.16"
$PyUrl = "https://api.nuget.org/v3-flatcontainer/python/3.13.16/python.3.13.16.nupkg"
$PySha256 = "95fad176338f1d6a799e45946488063f4aa61764e6cff2f132ddeffbf7a04c34"
$UvVersion = "0.12.23"
$UvUrl = "https://files.pythonhosted.org/packages/af/fd/816f4e8a76197312ec8037369e5b5c8a5ebde7aa118b6250259e031d3444/uv-0.12.23-py3-none-win_amd64.whl"
$UvSha256 = "fb8a4117a5224d73abe2204a1e744ae34b544f1b9a0b761576851deaa7215c12"
$FfmpegVersion = "8.1.2"
$FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip"
$FfmpegSha256 = "db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec"

function Say([string]$Text) { Write-Host ""; Write-Host $Text -ForegroundColor Cyan }
function Fail([string]$Text) {
    Write-Host ""; Write-Host $Text -ForegroundColor Red
    if (-not $NoPause) { Read-Host "Press Enter to close" | Out-Null }
    exit 1
}
function Check([string]$What) {
    # Call right after an external program: stop with a message if it failed.
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit code $LASTEXITCODE). See the messages above." }
}
function Get-Download([string]$Url, [string]$OutFile, [string]$Sha256) {
    # Download, then check the checksum (the versions are pinned, so the checksums are known).
    Invoke-WebRequest -Uri $Url -OutFile $OutFile -UseBasicParsing
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($OutFile)
    try { $got = -join ($sha.ComputeHash($stream) | ForEach-Object { $_.ToString("x2") }) } finally { $stream.Dispose() }
    if ($got -ne $Sha256) { Fail "The download of $Url is damaged (checksum mismatch). Run the setup again." }
}

function Expand-Zip([string]$Zip, [string]$Destination) {
    # .NET instead of Expand-Archive: no PowerShell module needed, and much faster on PowerShell 5.1.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Destination)
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

# Any error that stops the script (a failed download, a full disk...) ends here, with the message.
trap { Fail ("The setup stopped: " + $_.Exception.Message) }
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# A log next to the app, for when the window closed too fast to read (PowerShell 5.1 leaves out
# the output of programs like pip here; the steps and any error are in it).
Start-Transcript -Path (Join-Path $Root "setup.log") -Force | Out-Null
Say "TDPlay AI Video Render setup"
try { $Host.UI.RawUI.WindowTitle = "TDPlay AI Video Render setup - keep this window open" } catch { }
Write-Host "Keep this window open: it closes by itself when the setup is done. The downloads take a while." -ForegroundColor Yellow
Write-Host ("Options: " + (($PSBoundParameters.Keys | ForEach-Object { "-$_" }) -join " "))

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
if ($ramGB -lt 16 -and -not $TestCpu) { Fail "TDPlay AI Video Render needs at least 16 GB of RAM (32 GB recommended)." }

$target = Get-GpuTarget $gpus
if ($TestCpu) {
    $target = @{ Kind = "cpu"; Device = $null; Name = "the CPU (test setup)" }
} elseif (-not $target) {
    Fail "No supported graphics card found. TDPlay needs an AMD Radeon RX 7000 / RX 9000 or an NVIDIA card."
}
$kind = $target.Kind; $device = $target.Device; $cardName = $target.Name
if ($device -eq "all") {
    Write-Host "$cardName is not one of the tested cards; installing AMD's packages for all GPUs." -ForegroundColor Yellow
}
if ($kind -eq "amd") { Write-Host "Using $cardName with AMD ROCm (device-$device)." }
elseif ($kind -eq "nvidia") { Write-Host "Using $cardName with NVIDIA CUDA." }
else { Write-Host "Using $cardName." }

$drive = (Get-Item $Root).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB)
if ($freeGB -lt 80) {
    Write-Host "Only $freeGB GB free on drive $($drive.Name):. The model needs about 55 GB plus room for videos." -ForegroundColor Yellow
}

# --- Tools: uv (installs Python) and ffmpeg, kept next to the app -------------------------------
$vendor = Join-Path $Root "vendor"
New-Item -ItemType Directory -Force -Path $vendor | Out-Null
$uv = Join-Path $vendor "uv\uv.exe"
if (-not ((Test-Path $uv) -and ((& $uv --version) -match [regex]::Escape($UvVersion)))) {
    Say "Getting uv $UvVersion (it installs Python and the packages)..."
    $zip = Join-Path $env:TEMP "tdplay-uv.zip"
    $unzip = Join-Path $env:TEMP "tdplay-uv"
    Get-Download $UvUrl $zip $UvSha256
    if (Test-Path $unzip) { Remove-Item $unzip -Recurse -Force }
    Expand-Zip $zip $unzip
    New-Item -ItemType Directory -Force -Path (Split-Path $uv) | Out-Null
    Copy-Item (Join-Path $unzip "uv-$UvVersion.data\scripts\uv.exe") $uv -Force
    Remove-Item $zip, $unzip -Recurse -Force
}
$ffmpegBin = Join-Path $vendor "ffmpeg\bin"
$ffmpegMarker = Join-Path $vendor "ffmpeg\.tdplay-version"
if (-not ((Test-Path $ffmpegMarker) -and ((Get-Content $ffmpegMarker -Raw).Trim() -eq $FfmpegVersion))) {
    Say "Getting ffmpeg $FfmpegVersion..."
    $zip = Join-Path $env:TEMP "tdplay-ffmpeg.zip"
    $unzip = Join-Path $env:TEMP "tdplay-ffmpeg"
    Get-Download $FfmpegUrl $zip $FfmpegSha256
    if (Test-Path $unzip) { Remove-Item $unzip -Recurse -Force }
    Expand-Zip $zip $unzip
    if (Test-Path (Join-Path $vendor "ffmpeg")) { Remove-Item (Join-Path $vendor "ffmpeg") -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $ffmpegBin | Out-Null
    $built = Get-ChildItem $unzip | Select-Object -First 1
    foreach ($exe in @("ffmpeg.exe", "ffprobe.exe")) { Copy-Item (Join-Path $built.FullName "bin\$exe") $ffmpegBin }
    Set-Content -Path $ffmpegMarker -Value $FfmpegVersion
    Remove-Item $zip, $unzip -Recurse -Force
}
$probe = & (Join-Path $ffmpegBin "ffprobe.exe") -hide_banner -version
Check "Running ffprobe"
Write-Host $probe[0]

# --- Lightricks LTX-2 ------------------------------------------------------------------------
$ltxDir = Join-Path $vendor "LTX-2"
$marker = Join-Path $ltxDir ".tdplay-version"
if (-not ((Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $LtxTag))) {
    Say "Getting Lightricks LTX-2 $LtxTag..."
    if (Test-Path $ltxDir) { Remove-Item $ltxDir -Recurse -Force }
    $zip = Join-Path $env:TEMP "tdplay-ltx-2.zip"
    $unzip = Join-Path $env:TEMP "tdplay-ltx-2"
    Invoke-WebRequest -Uri "https://github.com/Lightricks/LTX-2/archive/refs/tags/$LtxTag.zip" -OutFile $zip -UseBasicParsing
    if (Test-Path $unzip) { Remove-Item $unzip -Recurse -Force }
    Expand-Zip $zip $unzip
    Move-Item (Get-ChildItem $unzip | Select-Object -First 1).FullName $ltxDir
    Set-Content -Path $marker -Value $LtxTag
    Remove-Item $zip, $unzip -Recurse -Force
}

# --- Python environment ------------------------------------------------------------------------
$pyHome = Join-Path $vendor "python"
$pyMarker = Join-Path $pyHome ".tdplay-version"
if (-not ((Test-Path $pyMarker) -and ((Get-Content $pyMarker -Raw).Trim() -eq $PyVersion))) {
    Say "Getting Python $PyVersion..."
    $zip = Join-Path $env:TEMP "tdplay-python.zip"
    $unzip = Join-Path $env:TEMP "tdplay-python"
    Get-Download $PyUrl $zip $PySha256
    if (Test-Path $unzip) { Remove-Item $unzip -Recurse -Force }
    Expand-Zip $zip $unzip
    if (Test-Path $pyHome) { Remove-Item $pyHome -Recurse -Force }
    Move-Item (Join-Path $unzip "tools") $pyHome
    Set-Content -Path $pyMarker -Value $PyVersion
    Remove-Item $zip, $unzip -Recurse -Force
}
# An environment made from another Python (an older version of this setup) is made again.
$cfg = Join-Path $Root ".venv\pyvenv.cfg"
if ((Test-Path $cfg) -and -not (Select-String -Path $cfg -SimpleMatch -Quiet -Pattern $pyHome)) {
    Remove-Item (Join-Path $Root ".venv") -Recurse -Force
}
Say "Setting up the Python environment..."
$env:UV_PYTHON_DOWNLOADS = "never"
& $uv venv --allow-existing --seed --python (Join-Path $pyHome "python.exe") .venv
Check "Creating the Python environment"
$py = Join-Path $Root ".venv\Scripts\python.exe"

Say "Installing PyTorch for your graphics card (a few GB)..."
if ($kind -eq "amd") {
    & $py -m pip install --index-url $RocmIndex "torch[device-$device]==$TorchRocm" "torchvision[device-$device]==$VisionRocm" "torchaudio==$AudioRocm"
    Check "Installing PyTorch (ROCm)"
} elseif ($kind -eq "cpu") {
    & $py -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
    Check "Installing PyTorch (CPU)"
} else {
    & $py -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
    Check "Installing PyTorch (CUDA)"
}

Say "Installing Lightricks LTX-2 and the app..."
& $py -m pip install --no-deps (Join-Path $ltxDir "packages\ltx-core") (Join-Path $ltxDir "packages\ltx-pipelines")
Check "Installing LTX-2"
& $py -m pip install -r (Join-Path $Root "windows\requirements-windows.txt")
Check "Installing the app's packages"

if ($kind -ne "cpu") {
    Say "Checking the graphics card..."
    & $py -c "import torch; ok = torch.cuda.is_available(); print('PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if ok else 'NOT FOUND'); raise SystemExit(0 if ok else 1)"
    Check "The GPU check (update the AMD Adrenalin driver and restart if PyTorch can't see the card)"
}
& $py -W ignore -c "import ltx_pipelines.distilled, tdplay.app; print('LTX-2 and the app load fine.')"
Check "Loading LTX-2 and the app"

# --- Shortcuts (Setup.exe makes its own) ----------------------------------------------------------
if (-not $FromInstaller) {
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
}

# --- Model ---------------------------------------------------------------------------------------
# From Setup.exe the window may be behind the installer, so don't wait for an answer there: the app's
# Models tab downloads the model, with a progress bar.
if ($FromInstaller -and -not $SkipModel) {
    Say "Next: open the app and download the video model (about 55 GB) in its Models tab."
} elseif (-not $SkipModel) {
    Say "Model download"
    Write-Host "TDPlay needs the LTX-2.3 model (about 55 GB). You can also do this later in the app's Models tab."
    $answer = Read-Host "Download it now? [Y/n]"
    if ($answer -notmatch "^[Nn]") {
        & $py -m tdplay download
        if ($LASTEXITCODE -ne 0) { Write-Host "The download stopped. Use the Models tab in the app: it continues where it stopped." -ForegroundColor Yellow }
    }
}

Say "All set! Open TDPlay AI Video Render from the Start menu or your desktop."
if (-not $FromInstaller) {
    Write-Host "(start.bat also works: it opens the app in your browser, with a window showing the log.)"
}
exit 0
