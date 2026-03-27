"""
ComfyUI-DreamScene360 Nodes

Takes a 360° panorama and produces a gap-filled POINTCLOUD via
DreamScene360's Gaussian Splatting pipeline.

Output PLY locations:
  {outdir}/{scene_name}/pointcloud.ply      (initial aligned point cloud)
  {outdir}/{scene_name}/{scene_name}_gs.ply (final trained Gaussians)
"""

import os
import sys
import uuid
import subprocess
import time
import glob
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(NODE_DIR, "dreamscene360_engine")
OUTPUT_DIR = os.path.join(NODE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def tensor_to_pil(tensor):
    """Convert ComfyUI IMAGE tensor [B, H, W, C] → PIL (first frame)."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def find_gs_ply(log_dir, scene_name):
    """
    Find the trained Gaussian PLY in a DreamScene360 output dir.
    
    Checks for (in priority order):
      1. {log_dir}/{scene_name}_gs.ply  (final trained Gaussians)
      2. {log_dir}/pointcloud.ply       (initial aligned point cloud)
      3. Any .ply file in log_dir
    """
    # Final trained Gaussians
    gs_ply = os.path.join(log_dir, f"{scene_name}_gs.ply")
    if os.path.exists(gs_ply):
        return gs_ply

    # Initial point cloud (still useful, has gap-filled geometry)
    init_ply = os.path.join(log_dir, "pointcloud.ply")
    if os.path.exists(init_ply):
        return init_ply

    # Fallback: any PLY
    plys = glob.glob(os.path.join(log_dir, "*.ply"))
    if plys:
        # Prefer _gs.ply files
        gs_files = [p for p in plys if p.endswith("_gs.ply")]
        return gs_files[0] if gs_files else plys[0]

    return None


def load_gaussian_ply(ply_path, opacity_threshold=0.05):
    """
    Load a 3DGS PLY and extract positions, colors, opacities.

    Returns:
        points:    np.ndarray (N, 3)  XYZ
        colors:    np.ndarray (N, 3)  RGB [0,1]
        opacities: np.ndarray (N,)    [0,1]
    """
    from plyfile import PlyData

    print(f"[DreamScene360] Loading PLY: {ply_path}")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    xyz = np.stack([
        np.asarray(vertex['x']),
        np.asarray(vertex['y']),
        np.asarray(vertex['z']),
    ], axis=-1)

    # Colors: try SH DC band first, then raw RGB
    SH_C0 = 0.28209479177387814
    try:
        colors = np.stack([
            np.asarray(vertex['f_dc_0']),
            np.asarray(vertex['f_dc_1']),
            np.asarray(vertex['f_dc_2']),
        ], axis=-1)
        colors = colors * SH_C0 + 0.5
        colors = np.clip(colors, 0.0, 1.0)
    except (ValueError, KeyError):
        try:
            colors = np.stack([
                np.asarray(vertex['red']),
                np.asarray(vertex['green']),
                np.asarray(vertex['blue']),
            ], axis=-1).astype(np.float64) / 255.0
        except (ValueError, KeyError):
            colors = np.ones_like(xyz) * 0.5

    # Opacity filter
    try:
        opacity_logit = np.asarray(vertex['opacity'])
        opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
        mask = opacity > opacity_threshold
        xyz = xyz[mask]
        colors = colors[mask]
        opacity = opacity[mask]
        print(f"[DreamScene360] {xyz.shape[0]} Gaussians after opacity > {opacity_threshold}")
    except (ValueError, KeyError):
        opacity = np.ones(len(xyz))
        print(f"[DreamScene360] {xyz.shape[0]} points (no opacity field)")

    return xyz, colors, opacity


# ---------------------------------------------------------------------------
# Node: DreamScene360 Panorama → Pointcloud
# ---------------------------------------------------------------------------

class DreamScene360PanoToPointcloud:
    """
    Runs DreamScene360 on a 360° panorama to produce a gap-filled
    POINTCLOUD with geometry inferred behind occluded regions.

    Outputs:
      - pointcloud            → 3D point cloud dict (points, colors, ply_path)
      - panorama_passthrough  → original panorama IMAGE passed through
      - depth_passthrough     → original depth_map IMAGE passed through
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "scene_name": ("STRING", {
                    "default": "my_scene",
                    "tooltip": "Name for this scene. Used for caching — "
                               "same name skips training if PLY exists."
                }),
                "iterations": ("INT", {
                    "default": 3000,
                    "min": 500, "max": 30000, "step": 500,
                    "tooltip": "GS training iterations. 3000≈5min on A5000."
                }),
            },
            "optional": {
                "opacity_threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0, "max": 0.5, "step": 0.01,
                }),
                "max_points": ("INT", {
                    "default": 500000,
                    "min": 50000, "max": 5000000, "step": 50000,
                }),
                "upscale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Upscale panorama before processing. "
                               "Better quality but needs more VRAM."
                }),
                "skip_if_exists": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse cached PLY. Train once, adjust camera freely."
                }),
            }
        }

    RETURN_TYPES = ("POINTCLOUD", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("pointcloud", "panorama_passthrough", "depth_passthrough",)
    FUNCTION = "process"
    CATEGORY = "3D/DreamScene360"

    def process(self, panorama, depth_map, scene_name, iterations,
                opacity_threshold=0.05, max_points=500000,
                upscale=False, skip_if_exists=True):

        # 1. Validate engine
        train_script = os.path.join(ENGINE_DIR, "train_headless.py")
        if not os.path.exists(train_script):
            raise FileNotFoundError(
                f"DreamScene360 engine not found at {ENGINE_DIR}.\n"
                f"Clone it: git clone --recursive "
                f"https://github.com/TingtingLiao/dreamscene360.git {ENGINE_DIR}\n"
                f"Then copy train_headless.py into that directory."
            )

        # 2. Clean scene name
        if not scene_name:
            scene_name = f"scene_{uuid.uuid4().hex[:8]}"
        scene_name = "".join(c for c in scene_name if c.isalnum() or c in "_- ")
        scene_name = scene_name.strip()

        # DreamScene360 uses {outdir}/{save_path}/ as log dir
        # save_path defaults to the prompt, but we override it
        log_dir = os.path.join(OUTPUT_DIR, scene_name)

        # 3. Check cache
        existing_ply = find_gs_ply(log_dir, scene_name) if os.path.isdir(log_dir) else None

        if skip_if_exists and existing_ply:
            print(f"[DreamScene360] Cached: {existing_ply}")
        else:
            # 4. Save panorama for DreamScene360
            os.makedirs(log_dir, exist_ok=True)
            pil_img = tensor_to_pil(panorama)
            pano_path = os.path.join(log_dir, "panorama_image.png")
            pil_img.save(pano_path, "PNG")
            print(f"[DreamScene360] Saved panorama: {pano_path} "
                  f"({pil_img.size[0]}x{pil_img.size[1]})")

            # 5. Run DreamScene360 headless training
            cmd = [
                sys.executable, train_script,
                "--panorama", pano_path,
                "--outdir", OUTPUT_DIR,
                "--scene_name", scene_name,
                "--iters", str(iterations),
            ]

            print(f"[DreamScene360] Running: {' '.join(cmd)}")
            print(f"[DreamScene360] Training {iterations} iterations...")
            start = time.time()

            env = os.environ.copy()
            env["PYTHONPATH"] = ENGINE_DIR + ":" + env.get("PYTHONPATH", "")

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, cwd=ENGINE_DIR, env=env,
            )

            for line in proc.stdout:
                line = line.strip()
                if line:
                    # Print key progress lines
                    if any(k in line.lower() for k in
                           ["step", "iter", "loss", "point", "saving",
                            "error", "exception", "traceback", "depth",
                            "panorama", "gaussian", "training"]):
                        print(f"[DreamScene360] {line}")

            proc.wait()
            elapsed = time.time() - start
            print(f"[DreamScene360] Done in {elapsed:.1f}s (exit: {proc.returncode})")

            if proc.returncode != 0:
                raise RuntimeError(
                    f"DreamScene360 failed (exit {proc.returncode}). "
                    f"Check console output above for errors."
                )

            existing_ply = find_gs_ply(log_dir, scene_name)

        # 6. Load PLY
        if not existing_ply:
            # List what's actually in the log dir for debugging
            if os.path.isdir(log_dir):
                contents = os.listdir(log_dir)
                print(f"[DreamScene360] Log dir contents: {contents}")
            raise FileNotFoundError(
                f"No PLY found in {log_dir}. "
                f"Training may not have completed successfully."
            )

        points, colors, opacities = load_gaussian_ply(existing_ply, opacity_threshold)

        # 7. Subsample if needed
        if len(points) > max_points:
            probs = opacities / opacities.sum()
            idx = np.random.choice(len(points), max_points, replace=False, p=probs)
            pc_pts, pc_col = points[idx], colors[idx]
        else:
            pc_pts, pc_col = points, colors

        pointcloud = {
            'ply_path': existing_ply,
            'points': torch.from_numpy(pc_pts.copy()).float(),
            'colors': torch.from_numpy(pc_col.copy()).float(),
        }

        print(f"[DreamScene360] POINTCLOUD: {len(pc_pts)} points from {existing_ply}")

        return (pointcloud, panorama, depth_map,)


# ---------------------------------------------------------------------------
# Node: Load Gaussian PLY
# ---------------------------------------------------------------------------

class LoadGaussianPLY:
    """Load a pre-trained 3D Gaussian Splatting PLY file as a POINTCLOUD."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ply_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "opacity_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                }),
                "max_points": ("INT", {
                    "default": 500000, "min": 50000, "max": 5000000, "step": 50000,
                }),
            }
        }

    RETURN_TYPES = ("POINTCLOUD",)
    RETURN_NAMES = ("pointcloud",)
    FUNCTION = "load"
    CATEGORY = "3D/DreamScene360"

    def load(self, ply_path, opacity_threshold=0.05, max_points=500000):
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY not found: {ply_path}")

        points, colors, opacities = load_gaussian_ply(ply_path, opacity_threshold)

        if len(points) > max_points:
            probs = opacities / opacities.sum()
            idx = np.random.choice(len(points), max_points, replace=False, p=probs)
            pc_pts, pc_col = points[idx], colors[idx]
        else:
            pc_pts, pc_col = points, colors

        return ({
            'points': torch.from_numpy(pc_pts.copy()).float(),
            'colors': torch.from_numpy(pc_col.copy()).float(),
        },)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "DreamScene360_PanoToPointcloud": DreamScene360PanoToPointcloud,
    "DreamScene360_LoadGaussianPLY": LoadGaussianPLY,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamScene360_PanoToPointcloud": "DreamScene360 Pano to Pointcloud",
    "DreamScene360_LoadGaussianPLY": "Load Gaussian PLY",
}
