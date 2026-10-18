#!/bin/bash
# First-launch setup of the TDPlay AI Video Render app from the disk image (.dmg). The app runs this
# by itself; it installs everything the app needs into ~/TDPlay/runtime: no Homebrew, no Terminal,
# no password. It runs again after the app is updated (it is quick then: downloads are cached).
#
# Usage: setup.sh APP_SOURCE RUNTIME_DIR
# Plain ASCII with every variable braced: macOS runs this with bash 3.2.
set -euo pipefail

SRC="${1:?usage: setup.sh APP_SOURCE RUNTIME_DIR}"
RUNTIME="${2:?usage: setup.sh APP_SOURCE RUNTIME_DIR}"

PY_VERSION="3.12"
LTX_MLX_TAG="v0.16.3"   # the ltx-2-mlx release TDPlay AI Video Render is tested against
UV_VERSION="0.12.23"
UV_URL="https://files.pythonhosted.org/packages/23/e9/b75db33954e985a763bd3277a477cabd551aaba7d08872b22400838dace2/uv-0.12.23-py3-none-macosx_11_0_arm64.whl"
UV_SHA256="99197c022afbd1edd0a295e9fdac5078c61eddf2e77657a77294cfe8ea0d94b9"
# ffmpeg and ffprobe for Apple Silicon: signed and notarized builds by Martin Riedl.
FFMPEG_URL="https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2"
FFMPEG_SHA256="c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924"
FFPROBE_SHA256="fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6"
FFMPEG_LATEST="https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release"

step() { echo; echo "== $(date '+%H:%M:%S') ${1}"; }
# The app shows the last "ERROR:" line to the user.
fail() { echo "ERROR: ${1}"; exit 1; }

# fetch URL FILE [SHA256]: download, and check the checksum when one is given.
fetch() {
  curl -fsSL --retry 3 --retry-delay 3 -o "${2}" "${1}" || return 1
  if [[ -n "${3:-}" ]]; then
    local got
    got="$(shasum -a 256 "${2}" | cut -d' ' -f1)"
    if [[ "${got}" != "${3}" ]]; then
      echo "Checksum mismatch for ${1}"
      return 1
    fi
  fi
}

[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] \
  || fail "TDPlay AI Video Render needs a Mac with Apple Silicon (M1 or newer)."
macos_major="$(sw_vers -productVersion | cut -d. -f1)"
(( macos_major >= 14 )) || fail "TDPlay AI Video Render needs macOS 14 Sonoma or newer."
echo "$(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"

BIN="${RUNTIME}/bin"
TMP="${RUNTIME}/tmp"
mkdir -p "${BIN}" "${TMP}"

# --- uv (installs Python and the packages) ------------------------------------------------
if [[ ! -x "${BIN}/uv" ]] || [[ "$("${BIN}/uv" --version 2>/dev/null | cut -d' ' -f2)" != "${UV_VERSION}" ]]; then
  step "Getting uv ${UV_VERSION}"
  fetch "${UV_URL}" "${TMP}/uv.whl" "${UV_SHA256}" || fail "Could not download uv. Check the internet connection."
  rm -rf "${TMP}/uv"
  unzip -q -o "${TMP}/uv.whl" "uv-${UV_VERSION}.data/scripts/uv" -d "${TMP}/uv"
  mv -f "${TMP}/uv/uv-${UV_VERSION}.data/scripts/uv" "${BIN}/uv"
  chmod 755 "${BIN}/uv"
fi
UV="${BIN}/uv"

# --- ffmpeg ------------------------------------------------------------------------------------
if [[ -x /opt/homebrew/bin/ffmpeg && -x /opt/homebrew/bin/ffprobe ]]; then
  echo "Using Homebrew's ffmpeg."
elif [[ ! -x "${BIN}/ffmpeg" || ! -x "${BIN}/ffprobe" ]]; then
  step "Getting ffmpeg"
  for tool in ffmpeg ffprobe; do
    if [[ "${tool}" == "ffmpeg" ]]; then sha="${FFMPEG_SHA256}"; else sha="${FFPROBE_SHA256}"; fi
    if ! fetch "${FFMPEG_URL}/${tool}.zip" "${TMP}/${tool}.zip" "${sha}"; then
      # That build was replaced by a newer one: take the latest, checked against its published checksum.
      url="$(curl -fsS -o /dev/null -w '%{redirect_url}' "${FFMPEG_LATEST}/${tool}.zip")" \
        || fail "Could not download ffmpeg. Check the internet connection."
      sha="$(curl -fsSL "${url}.sha256" | cut -d' ' -f1)" || fail "Could not download ffmpeg's checksum."
      fetch "${url}" "${TMP}/${tool}.zip" "${sha}" || fail "Could not download ffmpeg. Check the internet connection."
    fi
    unzip -q -o "${TMP}/${tool}.zip" "${tool}" -d "${BIN}"
    chmod 755 "${BIN}/${tool}"
  done
fi

# --- ltx-2-mlx (LTX-2 on MLX) ------------------------------------------------------------------
LTX_DIR="${RUNTIME}/vendor/ltx-2-mlx-${LTX_MLX_TAG}"
if [[ ! -d "${LTX_DIR}/packages" ]]; then
  step "Getting ltx-2-mlx ${LTX_MLX_TAG}"
  fetch "https://github.com/dgrauet/ltx-2-mlx/archive/refs/tags/${LTX_MLX_TAG}.tar.gz" "${TMP}/ltx.tar.gz" \
    || fail "Could not download ltx-2-mlx. Check the internet connection."
  rm -rf "${TMP}/ltx"
  mkdir -p "${TMP}/ltx" "${RUNTIME}/vendor"
  tar -xzf "${TMP}/ltx.tar.gz" -C "${TMP}/ltx"
  rm -rf "${LTX_DIR}"
  mv "${TMP}/ltx/"* "${LTX_DIR}"
fi

# --- Python and the packages --------------------------------------------------------------------
step "Setting up Python ${PY_VERSION} and the app's packages (about 1 GB the first time)"
export UV_PYTHON_INSTALL_DIR="${RUNTIME}/python"
"${UV}" venv --quiet --allow-existing --managed-python --python "${PY_VERSION}" "${RUNTIME}/venv"
PY="${RUNTIME}/venv/bin/python"
"${UV}" pip install --python "${PY}" \
  "${LTX_DIR}/packages/ltx-core-mlx" \
  "${LTX_DIR}/packages/ltx-pipelines-mlx" \
  -r "${SRC}/requirements.txt" || fail "Could not install the app's packages. Check the internet connection."

# --- The app's code: a copy, so it keeps working wherever the .app is (even run from the disk image) ---
step "Installing the app"
rm -rf "${RUNTIME}/app.new" "${RUNTIME}/app.old"
ditto --noqtn "${SRC}" "${RUNTIME}/app.new"
# The app looks for ffmpeg in its own vendor/ffmpeg/bin first (tdplay/media.py), whatever PATH says.
mkdir -p "${RUNTIME}/app.new/vendor/ffmpeg/bin"
for tool in ffmpeg ffprobe; do
  if [[ -x "${BIN}/${tool}" ]]; then src="${BIN}/${tool}"; else src="/opt/homebrew/bin/${tool}"; fi
  ln -sf "${src}" "${RUNTIME}/app.new/vendor/ffmpeg/bin/${tool}"
done
if [[ -d "${RUNTIME}/app" ]]; then mv "${RUNTIME}/app" "${RUNTIME}/app.old"; fi
mv "${RUNTIME}/app.new" "${RUNTIME}/app"
rm -rf "${RUNTIME}/app.old"

step "Checking"
"${PY}" -c "import mlx.core as mx; print('MLX', mx.__version__, 'on', mx.default_device())" \
  || fail "MLX could not use this Mac's GPU."
"${PY}" -m ltx_pipelines_mlx --help >/dev/null || fail "ltx-2-mlx did not install correctly."
(cd "${RUNTIME}/app" && "${PY}" -c "import tdplay.app") || fail "The app did not install correctly."
"${RUNTIME}/app/vendor/ffmpeg/bin/ffprobe" -version | head -1

cp "${SRC}/VERSION" "${RUNTIME}/VERSION"
rm -rf "${TMP}"
step "Ready."
