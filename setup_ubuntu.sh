#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODE="${1:-cpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.11}"
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
TORCH_VERSION_CUDA="${TORCH_VERSION_CUDA:-2.11.0}"
TORCHVISION_VERSION_CUDA="${TORCHVISION_VERSION_CUDA:-0.26.0}"
TORCH_VERSION_CPU="${TORCH_VERSION_CPU:-2.11.0}"
TORCHVISION_VERSION_CPU="${TORCHVISION_VERSION_CPU:-0.26.0}"

usage() {
  cat <<EOF
Usage:
  ./setup_ubuntu.sh [cpu|cuda]

What it does:
  - Installs Ubuntu system packages
  - Installs pyenv if needed
  - Installs Python ${PYTHON_VERSION} if needed
  - Creates .venv with that Python
  - Installs repo Python dependencies
  - Installs missing extras used by the repo:
      - segmentation-models-pytorch
      - Meta segment-anything

Modes:
  cpu   Install CPU PyTorch wheels
  cuda  Install CUDA 12.8 PyTorch wheels

Notes:
  - 'cuda' mode assumes NVIDIA drivers are already installed and working.
  - CUDA mode uses the current official CUDA 12.8 PyTorch wheels to support newer GPUs such as RTX 50-series.
  - This repo's U-Net training scripts are placeholders.
  - SAM also needs a downloaded checkpoint file at runtime.
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${MODE}" != "cpu" && "${MODE}" != "cuda" ]]; then
  echo "Invalid mode: ${MODE}"
  usage
  exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "This script expects sudo to be available."
  exit 1
fi

install_pyenv() {
  if [[ -x "${PYENV_ROOT}/bin/pyenv" ]]; then
    return
  fi

  echo "Installing pyenv into ${PYENV_ROOT}"
  curl -fsSL https://pyenv.run | bash
}

init_pyenv() {
  export PYENV_ROOT
  export PATH="${PYENV_ROOT}/bin:${PATH}"

  if [[ ! -x "${PYENV_ROOT}/bin/pyenv" ]]; then
    echo "pyenv installation not found at ${PYENV_ROOT}"
    exit 1
  fi

  eval "$("${PYENV_ROOT}/bin/pyenv" init - bash)"
}

echo "[1/7] Installing Ubuntu system packages"
sudo apt update
sudo apt install -y \
  git \
  curl \
  ca-certificates \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  build-essential \
  pkg-config \
  gfortran \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libxml2-dev \
  libxmlsec1-dev \
  libffi-dev \
  liblzma-dev

echo "[2/7] Installing pyenv if needed"
install_pyenv
init_pyenv

echo "[3/7] Installing Python ${PYTHON_VERSION} if needed"
if ! pyenv versions --bare | grep -Fxq "${PYTHON_VERSION}"; then
  pyenv install "${PYTHON_VERSION}"
fi
PYTHON_BIN="${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python"

echo "[4/7] Creating virtual environment at ${VENV_DIR}"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[5/7] Upgrading packaging tools"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[6/7] Installing repo Python dependencies"
REQS_NO_TORCH="$(mktemp)"
grep -vE '^(torch|torchvision)==.*$' "${ROOT_DIR}/requirements.txt" > "${REQS_NO_TORCH}"
pip install -r "${REQS_NO_TORCH}"
rm -f "${REQS_NO_TORCH}"

if [[ "${MODE}" == "cuda" ]]; then
  pip install \
    "torch==${TORCH_VERSION_CUDA}" \
    "torchvision==${TORCHVISION_VERSION_CUDA}" \
    --index-url https://download.pytorch.org/whl/cu128
else
  pip install \
    "torch==${TORCH_VERSION_CPU}" \
    "torchvision==${TORCHVISION_VERSION_CPU}" \
    --index-url https://download.pytorch.org/whl/cpu
fi

pip install segmentation-models-pytorch
pip install git+https://github.com/facebookresearch/segment-anything.git

echo "[7/7] Running quick environment checks"
python - <<'PY'
import importlib
import sys

modules = [
    "cv2",
    "numpy",
    "h5py",
    "skimage",
    "torch",
    "torchvision",
    "ultralytics",
    "segmentation_models_pytorch",
]

missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)))

if missing:
    print("Missing modules:")
    for name, exc in missing:
        print(f"  - {name}: {exc}")
    sys.exit(1)

import torch

print("Python package check: OK")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY

cat <<EOF

Setup completed.

Activate the environment with:
  source .venv/bin/activate

Python used:
  ${PYTHON_BIN}

Important repo notes:
  - U-Net training scripts in code/train_unet.py and code/train_guided_unet.py are placeholders.
  - SAM inference requires a SAM checkpoint path in config.
  - Additional datasets may still need to be downloaded into the paths expected by code/build_manifests.py.
EOF
