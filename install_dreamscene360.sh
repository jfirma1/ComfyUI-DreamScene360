#!/bin/bash
set -e

# Resolve paths relative to this script — works on any machine, not just RunPod
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$SCRIPT_DIR/dreamscene360_engine"
CPP_DIR="$ENGINE_DIR/submodules/360monodepth/code/cpp"
echo "============================================"
echo "  DreamScene360: Restoring Dependencies"
echo "============================================"

# 0. SYSTEM LIBRARIES (wiped on pod restart)
echo "[0/6] Installing system libraries..."
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libopencv-dev libgoogle-glog-dev libgflags-dev libsuitesparse-dev libboost-all-dev libeigen3-dev 2>&1 | tail -3

# 1. COMFYUI CORE DEPS (sqlalchemy etc)
echo "[1/6] Installing ComfyUI requirements..."
COMFYUI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$COMFYUI_DIR/requirements.txt" ]; then
    pip install -r "$COMFYUI_DIR/requirements.txt" --break-system-packages -q 2>&1 | tail -3
else
    echo "  requirements.txt not found at $COMFYUI_DIR, skipping."
fi

# 2. PIP PACKAGES
echo "[2/6] Installing DreamScene360 pip packages..."
pip install --break-system-packages --no-cache-dir -q \
    "opencv-python==4.9.0.80" \
    "scipy==1.13.1" \
    matplotlib timm colorama omegaconf plyfile pybind11 addict kiui \
    torchmetrics scikit-image plotly \
    2>&1 | tail -3

# 3. INSTAOMNI DEPTH
echo "[3/6] Restoring instaOmniDepth..."
cd "$CPP_DIR/python"
pip install . --break-system-packages --no-build-isolation --no-cache-dir -q 2>&1 | tail -3
BUILT_SO="$CPP_DIR/python/build/lib.linux-x86_64-cpython-311/instaOmniDepth/depthmapAlign.cpython-311-x86_64-linux-gnu.so"
if [ -f "$BUILT_SO" ]; then
    cp "$BUILT_SO" "$CPP_DIR/python/instaOmniDepth/"
    echo "  Copied depthmapAlign.so to local dir"
fi

# 4. CUDA SUBMODULES
echo "[4/6] Restoring CUDA submodules..."
cd "$ENGINE_DIR"
pip install submodules/diff-gaussian-rasterization-depth --break-system-packages --no-build-isolation -q 2>&1 | tail -3
pip install submodules/simple-knn --break-system-packages --no-build-isolation -q 2>&1 | tail -3

# 5. COMFYUI NODE DEPS
echo "[5/6] Restoring ComfyUI node dependencies..."
pip install --break-system-packages -q addict rich GitPython soundfile replicate requests toml packaging 2>&1 | tail -3

# *** NUMPY LOCKDOWN — absolute last pip command, force 1.26.4 ***
echo "  Pinning numpy==1.26.4..."
pip install --break-system-packages --no-cache-dir --force-reinstall -q "numpy==1.26.4" 2>&1 | tail -3

# 6. VERIFY
echo "[6/6] Verifying..."
cd "$ENGINE_DIR"
python3 -c "
import sys
sys.path.insert(0, 'submodules/360monodepth/code/python/src')
sys.path.insert(0, 'submodules/360monodepth/code/python/src/utility')
import numpy; assert numpy.__version__ == '1.26.4', f'WRONG NUMPY: {numpy.__version__}'
print(f'  numpy: {numpy.__version__} (CORRECT)')
import scipy; print(f'  scipy: {scipy.__version__}')
import cv2; print(f'  opencv: {cv2.__version__}')
from instaOmniDepth import depthmapAlign; print('  1. instaOmniDepth: OK')
from utility.projection_icosahedron import erp2ico_image; print('  2. erp2ico_image: OK')
from utility import depthmap_utils; print('  3. depthmap_utils: OK')
from utility import blending; print('  4. blending: OK')
from utility import pointcloud_utils; print('  5. pointcloud_utils: OK')
from gs_renderer import Renderer; print('  6. gs_renderer: OK')
from utils import cam_utils; print('  7. cam_utils: OK')
from utils.loss_utils import l1_loss, ssim; print('  8. loss_utils: OK')
from utils import depthmap_align; print('  9. depthmap_align: OK')
from utils.Equirec2Perspec import Equirectangular; print('  10. Equirec2Perspec: OK')
import timm; print('  11. timm: OK')
print('  ALL IMPORTS OK')
" || echo "  VERIFICATION FAILED — check errors above"
echo "============================================"
echo "  DreamScene360: Setup Complete"
echo "============================================"
