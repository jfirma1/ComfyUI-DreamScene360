# ComfyUI-DreamScene360

A ComfyUI custom node wrapping the **Panorama-to-3D** path of [DreamScene360](https://github.com/ShijieZhou-UCLA/DreamScene360) (ECCV 2024). Feed it a 360° equirectangular panorama and it outputs a gap-filled **3D Gaussian Splat point cloud** (`.ply`) you can connect to downstream 3D nodes.

> **What this wraps:** DreamScene360 is a text-to-3D scene generation system using Panoramic Gaussian Splatting. It has two modes: *text-to-3D* (generates a panorama from a prompt via StitchDiffusion + GPT-4V refinement, then lifts to 3D) and *panorama-to-3D* (lifts an existing panorama directly). **This node uses the panorama-to-3D path only** — no OpenAI key or diffusion models required.

---

## What It Does

```
360° Panorama → [DreamScene360 Gaussian Splatting] → POINTCLOUD (.ply)
```

The core problem: a panorama is 2.5D — one viewpoint with no geometry behind objects. DreamScene360 reconstructs true 3D by:

1. **Tangent image projection** — reproject the equirectangular panorama into overlapping perspective crops
2. **Monocular depth estimation** — run `omnidata_dpt_depth_v2` on each crop
3. **Global depth alignment** — align and Poisson-blend all depth maps into a coherent ERP depth field
4. **MLP refinement** — a small neural net smooths cross-seam depth discontinuities globally
5. **Point cloud reprojection** — unproject the aligned depth into 3D world space
6. **Gaussian Splatting training** — initialize 3DGS from the point cloud and optimize against the original panorama as a spherical reference image

The output is a standard 3DGS `.ply` file (SH DC coefficients for color) plus a passthrough of your original inputs.

---

## Nodes

| Node | Description |
|------|-------------|
| **DreamScene360 Pano to Pointcloud** | Main node. Runs the full pipeline on a panorama + depth map. |
| **Load Gaussian PLY** | Utility. Loads any pre-existing 3DGS `.ply` directly as a POINTCLOUD. |

---

## Installation

### Option A: Automatic

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-DreamScene360.git
cd ComfyUI-DreamScene360
python install.py
```

`install.py` handles: cloning the DreamScene360 engine, system deps, Python packages, CUDA submodule builds, and checkpoint download.

### Option B: Manual

```bash
# 1. Clone this node
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-DreamScene360.git
cd ComfyUI-DreamScene360

# 2. Clone the upstream engine
git clone --recursive https://github.com/ShijieZhou-UCLA/DreamScene360.git dreamscene360_engine

# 3. System dependencies (Debian/Ubuntu)
sudo apt-get install -y libglm-dev libglew-dev libassimp-dev libboost-all-dev \
    libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev \
    libeigen3-dev libxxf86vm-dev

# 4. Python dependencies
pip install plyfile open3d trimesh scipy einops timm opencv-python scikit-image omegaconf

# 5. Build the CUDA submodules (requires nvcc)
pip install dreamscene360_engine/submodules/diff-gaussian-rasterization-depth
pip install dreamscene360_engine/submodules/simple-knn

# 6. (Optional but recommended) tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 7. Download the depth checkpoint (see below)

# 8. Restart ComfyUI
```

### Checkpoint

`omnidata_dpt_depth_v2.ckpt` is required. Download it from the [dropbox link in the DreamScene360 README](https://github.com/ShijieZhou-UCLA/DreamScene360#checkpoints) and place it at:

```
ComfyUI-DreamScene360/dreamscene360_engine/pre_checkpoints/omnidata_dpt_depth_v2.ckpt
```

---

## Usage

### Workflow

```
[Load Image: 360° Panorama] ──┐
                              ├──► [DreamScene360 Pano to Pointcloud] ──► pointcloud ──► (your 3D node)
[Load Image: Depth Map]     ──┘
```

A matching equirectangular depth map is required. Use any upstream depth estimation node, or generate one from the panorama directly. The depth is used as an initial geometry prior — better depth = cleaner point cloud.

### Node Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scene_name` | `my_scene` | Cache key. Same name reuses the saved `.ply` and skips re-training. |
| `iterations` | `3000` | Gaussian Splatting training steps. ~3000 for a draft, ~9000 for quality output. |
| `opacity_threshold` | `0.05` | Filter out Gaussians below this opacity before returning the point cloud. |
| `max_points` | `500000` | Cap on returned points. Excess is downsampled weighted by opacity. |
| `upscale` | `false` | 2× upscale panorama before processing for finer geometry. Uses more VRAM. |
| `skip_if_exists` | `true` | Load cached `.ply` if already trained for this `scene_name`. |

### Outputs

| Output | Type | Description |
|--------|------|-------------|
| `pointcloud` | POINTCLOUD | Dict: `points` (N×3 float32), `colors` (N×3 float32), `ply_path` (str) |
| `panorama_passthrough` | IMAGE | Input panorama passed through unchanged |
| `depth_passthrough` | IMAGE | Input depth map passed through unchanged |

### Caching

Trained Gaussians are saved to:
```
ComfyUI-DreamScene360/output/{scene_name}/{scene_name}_gs.ply
```
With `skip_if_exists = true`, re-running with the same `scene_name` loads instantly — useful for iterating on downstream nodes without re-running the full training pipeline.

---

## Typical Training Time

| GPU | 3,000 iters | 9,000 iters |
|-----|------------|------------|
| A100 | ~3 min | ~8 min |
| A40 | ~5 min | ~14 min |
| RTX 4090 | ~4 min | ~12 min |
| RTX 3090 | ~6 min | ~18 min |

---

## Tips

- **Panorama resolution matters.** 4096×2048 minimum recommended. Higher resolution = better coverage and fewer holes.
- **Start with 3000 iterations** to verify everything works, then increase to 9000 for final output.
- **Depth map quality is a multiplier.** A sharp, accurate depth map (especially at object boundaries) produces noticeably cleaner Gaussians.
- **If VRAM is tight:** lower `max_points`, keep `upscale = false`, reduce `iterations`.
- **The `Load Gaussian PLY` node** lets you skip training entirely if you already have a `.ply` on disk from a prior run or another tool.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `DreamScene360 engine not found` | Run `python install.py`. Confirm `dreamscene360_engine/train_headless.py` exists. |
| CUDA build errors | Verify `nvcc --version` works. Install CUDA toolkit if missing. |
| `omnidata_dpt_depth_v2.ckpt not found` | Download manually from the DreamScene360 dropbox link → `dreamscene360_engine/pre_checkpoints/`. |
| Out of memory | Set `upscale = false`, lower `max_points`, reduce `iterations`. |
| Imports fail after pod restart | Re-run `python install.py`. |
| CUDA rasterizer fails to compile | `sudo apt-get install libglm-dev` then retry. |

---

## Requirements

- NVIDIA GPU with CUDA support
- 8 GB+ VRAM (16 GB+ recommended)
- ComfyUI
- Python 3.10 or 3.11
- CUDA toolkit with `nvcc` (for building submodules)

---

## Credits

- [DreamScene360](https://github.com/ShijieZhou-UCLA/DreamScene360) — Zhou, Fan, Xu et al., ECCV 2024
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — INRIA
- [PERF](https://github.com/perf-project/PeRF) — omnidata depth checkpoint hosting
- [360monodepth](https://github.com/manurare/360monodepth) — ERP depth alignment pipeline

## License

Follows the [DreamScene360 license](https://github.com/ShijieZhou-UCLA/DreamScene360/blob/main/LICENSE.md).
