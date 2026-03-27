"""
Install script for ComfyUI-DreamScene360
Automatically called by ComfyUI-Manager or can be run manually.
Designed for RunPod / Linux GPU environments.
"""
import subprocess
import sys
import os

DREAMSCENE_DIR = os.path.join(os.path.dirname(__file__), "dreamscene360_engine")
CHECKPOINT_DIR = os.path.join(DREAMSCENE_DIR, "pre_checkpoints")

def run(cmd, cwd=None):
    print(f"[DreamScene360 Install] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout[-500:])  # last 500 chars
    if result.returncode != 0:
        print(f"[DreamScene360 Install] WARNING: {result.stderr[-500:]}")
    return result.returncode == 0

def install():
    print("=" * 60)
    print("[DreamScene360 Install] Starting installation...")
    print("=" * 60)

    # 1. Clone DreamScene360 if not present
    if not os.path.exists(os.path.join(DREAMSCENE_DIR, "train.py")):
        print("[DreamScene360 Install] Cloning DreamScene360 repository...")
        if os.path.exists(DREAMSCENE_DIR):
            run(f"rm -rf {DREAMSCENE_DIR}")
        run(f"git clone --recursive https://github.com/ShijieZhou-UCLA/DreamScene360.git {DREAMSCENE_DIR}")
    else:
        print("[DreamScene360 Install] DreamScene360 already cloned, skipping.")

    # 2. Install system dependencies
    print("[DreamScene360 Install] Installing system dependencies...")
    run("apt-get update -qq && apt-get install -y -qq libglm-dev libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev 2>/dev/null || true")

    # 3. Install Python dependencies (into the active ComfyUI environment)
    print("[DreamScene360 Install] Installing Python packages...")
    pip = sys.executable.replace("python", "pip") if "python" in sys.executable else "pip"
    python = sys.executable

    # Core packages DreamScene360 needs beyond what ComfyUI provides
    packages = [
        "plyfile",
        "open3d",
        "trimesh",
        "scipy",
        "tqdm",
        "Pillow",
        "opencv-python",
        "einops",
        "timm",
    ]
    for pkg in packages:
        run(f"{python} -m pip install {pkg} --quiet --break-system-packages 2>/dev/null || {python} -m pip install {pkg} --quiet")

    # 4. Build the custom CUDA rasterizer submodules
    diff_rast_dir = os.path.join(DREAMSCENE_DIR, "submodules", "diff-gaussian-rasterization-depth")
    simple_knn_dir = os.path.join(DREAMSCENE_DIR, "submodules", "simple-knn")

    if os.path.exists(diff_rast_dir):
        print("[DreamScene360 Install] Building diff-gaussian-rasterization-depth...")
        run(f"{python} -m pip install {diff_rast_dir} --break-system-packages 2>/dev/null || {python} -m pip install {diff_rast_dir}")

    if os.path.exists(simple_knn_dir):
        print("[DreamScene360 Install] Building simple-knn...")
        run(f"{python} -m pip install {simple_knn_dir} --break-system-packages 2>/dev/null || {python} -m pip install {simple_knn_dir}")

    # 5. Install tiny-cuda-nn (optional but recommended)
    print("[DreamScene360 Install] Installing tiny-cuda-nn (may take a few minutes)...")
    run(f"{python} -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --break-system-packages 2>/dev/null || echo 'tiny-cuda-nn install failed - will work without it'")

    # 6. Download the omnidata depth checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "omnidata_dpt_depth_v2.ckpt")
    if not os.path.exists(ckpt_path):
        print("[DreamScene360 Install] Downloading omnidata_dpt_depth_v2.ckpt...")
        # Try the dropbox link from the repo README
        run(f"wget -q 'https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt' -O {ckpt_path} 2>/dev/null || echo 'Auto-download failed. Please manually download omnidata_dpt_depth_v2.ckpt into {CHECKPOINT_DIR}'")
        if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) < 1000:
            print(f"[DreamScene360 Install] ⚠️  Could not auto-download checkpoint.")
            print(f"[DreamScene360 Install] Please manually download omnidata_dpt_depth_v2.ckpt")
            print(f"[DreamScene360 Install] from the DreamScene360 README dropbox link")
            print(f"[DreamScene360 Install] and place it in: {CHECKPOINT_DIR}")
    else:
        print("[DreamScene360 Install] Checkpoint already present, skipping download.")

    print("=" * 60)
    print("[DreamScene360 Install] Installation complete!")
    print("=" * 60)

if __name__ == "__main__":
    install()
