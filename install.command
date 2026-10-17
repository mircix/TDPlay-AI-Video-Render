#!/bin/bash
# TDPlay AI Video Render installer for Apple Silicon Macs. Double-click it, or run:  bash install.command
set -euo pipefail
cd "$(dirname "$0")"

LTX_MLX_TAG="v0.16.3"   # the ltx-2-mlx release TDPlay AI Video Render is tested against
PY_VERSION="3.12"

bold() { printf "\n\033[1m%s\033[0m\n" "$1"; }
fail() { printf "\n\033[31m%s\033[0m\n" "$1"; read -r -p "Press Return to close." _; exit 1; }

bold "TDPlay AI Video Render installer"

[[ "$(uname -s)" == "Darwin" ]] || fail "TDPlay AI Video Render runs on macOS only."
[[ "$(uname -m)" == "arm64" ]] || fail "TDPlay AI Video Render needs an Apple Silicon Mac (M1 or newer)."
macos_major=$(sw_vers -productVersion | cut -d. -f1)
(( macos_major >= 14 )) || fail "TDPlay AI Video Render needs macOS 14 Sonoma or newer (you have $(sw_vers -productVersion))."
ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
echo "$(sysctl -n machdep.cpu.brand_string), ${ram_gb} GB memory, macOS $(sw_vers -productVersion)"
(( ram_gb >= 16 )) || fail "TDPlay AI Video Render needs at least 16 GB of memory."

# --- Homebrew, ffmpeg, uv -------------------------------------------------------
export PATH="/opt/homebrew/bin:${PATH}"
if ! command -v brew >/dev/null; then
  bold "Homebrew (the standard Mac package manager) is needed to install ffmpeg."
  read -r -p "Install Homebrew now? It will ask for your Mac password. [y/N] " answer
  [[ "${answer}" =~ ^[Yy] ]] || fail "Install Homebrew from https://brew.sh, then run this installer again."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
for tool in ffmpeg uv; do
  if ! command -v "${tool}" >/dev/null; then
    bold "Installing ${tool}..."
    brew install "${tool}"
  fi
done

# --- ltx-2-mlx (LTX-2 on MLX) ------------------------------------------------------
bold "Getting ltx-2-mlx ${LTX_MLX_TAG}..."
if [[ -d vendor/ltx-2-mlx/.git ]]; then
  git -C vendor/ltx-2-mlx fetch --depth 1 origin tag "${LTX_MLX_TAG}"
  git -C vendor/ltx-2-mlx -c advice.detachedHead=false checkout -q "${LTX_MLX_TAG}"
else
  mkdir -p vendor
  git -c advice.detachedHead=false clone -q --depth 1 --branch "${LTX_MLX_TAG}" https://github.com/dgrauet/ltx-2-mlx.git vendor/ltx-2-mlx
fi

# --- Python environment -------------------------------------------------------------
bold "Setting up Python ${PY_VERSION} and the app's packages..."
uv venv --quiet --allow-existing --python "${PY_VERSION}" .venv
uv pip install --quiet --python .venv/bin/python \
  ./vendor/ltx-2-mlx/packages/ltx-core-mlx \
  ./vendor/ltx-2-mlx/packages/ltx-pipelines-mlx \
  -r requirements.txt

bold "Checking the GPU..."
.venv/bin/python -c "import mlx.core as mx; print('MLX', mx.__version__, 'on', mx.default_device())"
.venv/bin/python -m ltx_pipelines_mlx --help >/dev/null
.venv/bin/python -c "import tdplay.app" >/dev/null

bold "Adding TDPlay AI Video Render to your Applications..."
.venv/bin/python -m tdplay.macos_bundle

# --- Model --------------------------------------------------------------------------
bold "Model download"
echo "TDPlay AI Video Render needs one LTX-2 model pack (about 30-45 GB). You can also do this later in the app's Models tab."
read -r -p "Download the recommended model for this Mac now? [Y/n] " answer
if [[ ! "${answer}" =~ ^[Nn] ]]; then
  .venv/bin/python -m tdplay download || echo "The download stopped. Run this installer again, or use the Models tab - it continues where it stopped."
fi

bold "All set! Open TDPlay AI Video Render from Applications, Launchpad or Spotlight."
echo "(start.command still works too: it opens the app in your browser instead of its own window.)"
read -r -p "Press Return to close." _
