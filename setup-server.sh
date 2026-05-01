#!/usr/bin/env bash
# Server-side env bootstrap for mesh2robot training (H200 + similar).
#
# Strategy: tiny conda env (Python + pip only), then install everything
# via pip. This sidesteps the conda-forge / pytorch / nvidia channel
# conflict hell on libabseil/numpy/protobuf that environment.yml hits
# on miniconda 26+. PyTorch's official pip wheels are the reliable
# CUDA distribution path now.
#
# Run on a headless training box:
#   bash setup-server.sh
#
# Then activate with:
#   conda activate mesh2robot
#
# Re-run idempotently — `conda env remove -n mesh2robot -y` first if you
# want a fully clean reinstall.

set -euo pipefail

ENV_NAME=mesh2robot
PY_VER=3.12
TORCH_VER=2.6.0
CUDA_TAG=cu124        # bundled toolkit version; runs fine on host nvcc 12.8

echo "=== [1/4] Creating minimal conda env: $ENV_NAME (python=$PY_VER) ==="
# Skip if env already exists with the right Python; re-create otherwise.
if conda env list | grep -qE "^$ENV_NAME\s"; then
    echo "  env exists; skipping create. Remove with 'conda env remove -n $ENV_NAME -y' to reset."
else
    conda create -n "$ENV_NAME" "python=$PY_VER" pip -y
fi

# Source conda activate inside this script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo
echo "=== [2/4] PyTorch $TORCH_VER + CUDA $CUDA_TAG (from pytorch.org wheels) ==="
pip install --upgrade pip
pip install "torch==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

echo
echo "=== [3/4] PT-V3 CUDA-bound deps (spconv + torch-scatter) ==="
pip install "spconv-$CUDA_TAG"
pip install torch-scatter -f "https://data.pyg.org/whl/torch-$TORCH_VER+$CUDA_TAG.html"

echo
echo "=== [4/4] Remaining deps ==="
# Server-only: no pyvista/vtk (interactive GUI not used here).
pip install \
    "numpy>=2.0" \
    "scipy>=1.13" \
    "trimesh>=4.10" \
    "yourdfpy>=0.0.55" \
    "mujoco>=3.5" \
    "robot_descriptions>=1.20" \
    "xacrodoc>=0.1" \
    "imageio>=2.35" \
    "jinja2>=3.1" \
    "tqdm>=4.65" \
    "matplotlib>=3.8" \
    "opencv-contrib-python>=4.10" \
    "pybullet>=3.2" \
    "timm>=1.0" \
    "addict>=2.4"

echo
echo "=== Verify ==="
python -c "
import torch, mujoco, trimesh, yourdfpy, pybullet, spconv, torch_scatter, timm
print('torch     :', torch.__version__, '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| cuda?', torch.cuda.is_available())
print('mujoco    :', mujoco.__version__)
print('trimesh   :', trimesh.__version__)
print('yourdfpy  :', yourdfpy.__version__)
print('spconv    : ok')
print('torch-scatter: ok')
"

echo
echo "Done. Activate with: conda activate $ENV_NAME"
