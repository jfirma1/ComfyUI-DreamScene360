import os
import sys
import time
import argparse
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import gc # Added for memory cleanup
from PIL import Image

# Add engine source to path
ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, os.path.join(ENGINE_DIR, "submodules", "360monodepth", "code", "python", "src"))
sys.path.insert(0, os.path.join(ENGINE_DIR, "submodules", "360monodepth", "code", "python", "src", "utility"))

from omegaconf import OmegaConf

from gs_renderer import Renderer, MiniCam
from utils import cam_utils
from utils.loss_utils import l1_loss, ssim
from utils import depthmap_align
from utils.Equirec2Perspec import Equirectangular
from utility import blending, image_io, depthmap_utils, serialization, pointcloud_utils
from utility.projection_icosahedron import erp2ico_image

# --- IMPORTS ---
from skimage.transform import resize
from scipy.interpolate import griddata

# -------------------------------------------------------------------------
# MLP DEFINITION
# -------------------------------------------------------------------------
class DepthMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super(DepthMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_config(overrides=None):
    config_path = os.path.join(ENGINE_DIR, "configs", "default.yaml")
    opt = OmegaConf.load(config_path)
    if overrides:
        opt = OmegaConf.merge(opt, OmegaConf.create(overrides))
    return opt

def cleanup_cuda():
    """Aggressively clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def train_from_panorama(panorama_path, outdir, scene_name, iters=3000, opt=None):
    if opt is None:
        opt = load_config()

    opt.gui = False
    opt.outdir = outdir
    opt.save_path = scene_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup directories ---
    log_dir = os.path.join(outdir, scene_name)
    os.makedirs(log_dir, exist_ok=True)
    subimages_dir = os.path.join(log_dir, "subimages")
    depth_estimate_dir = os.path.join(log_dir, "depth_estimate")
    depth_align_dir = os.path.join(log_dir, "depth_align")
    os.makedirs(subimages_dir, exist_ok=True)
    os.makedirs(depth_estimate_dir, exist_ok=True)
    os.makedirs(depth_align_dir, exist_ok=True)
    
    # --- Copy panorama ---
    pano_dst = os.path.join(log_dir, "panorama_image.png")
    if os.path.abspath(panorama_path) != os.path.abspath(pano_dst):
        img = Image.open(panorama_path)
        img.save(pano_dst)
    
    # --- 1. Project ---
    print("[train_headless] Step 1/7: Projecting to tangent images...")
    erp_image = cv2.cvtColor(cv2.imread(pano_dst), cv2.COLOR_BGR2RGB)
    erp_image = cv2.resize(erp_image, (opt.pano_width, opt.pano_height))

    subimg_rgb_list, _, points_gnomocoord = erp2ico_image(
        erp_image, opt.tangent_img_width,
        padding_size=opt.subimage_padding_size,
        full_face_image=True
    )
    subimg_gnomo_xy = points_gnomocoord[1]

    # --- 2. Depth ---
    print("[train_headless] Step 2/7: Estimating depth...")
    dispmap_persp_list = depthmap_utils.run_persp_monodepth(subimg_rgb_list, opt.persp_monodepth)

    depthmap_persp_list = []
    depthmap_erp_list = []
    dispmap_erp_list = []

    for dispmap_persp in dispmap_persp_list:
        depthmap_persp = depthmap_utils.disparity2depth(dispmap_persp)
        depthmap_erp = depthmap_utils.subdepthmap_tang2erp(depthmap_persp, subimg_gnomo_xy)
        dispmap_erp = depthmap_utils.depth2disparity(depthmap_erp).astype(np.float32)
        depthmap_persp_list.append(depthmap_persp)
        depthmap_erp_list.append(depthmap_erp)
        dispmap_erp_list.append(dispmap_erp)

    # --- 3. Align ---
    print("[train_headless] Step 3/7: Aligning depth maps...")
    depthmap_aligner_obj = depthmap_align.DepthmapAlign(opt, depth_align_dir, subimg_rgb_list, debug=False)

    subimage_available_list = list(range(len(dispmap_erp_list)))
    dispmap_aligned_list, coeffs_scale, coeffs_offset, subimg_cam_list = \
        depthmap_aligner_obj.align_multi_res(erp_image, dispmap_erp_list, opt.subimage_padding_size, subimage_available_list)

    # --- 4. Blend ---
    print("[train_headless] Step 4/7: Blending depth → pointcloud...")
    blend_it = blending.BlendIt(opt.subimage_padding_size, len(subimage_available_list), opt.blending_method)
    blend_it.fidelity_weight = 0.5  # Increased from 0.1: stick closer to input, fewer Poisson oscillations

    erp_image_height = erp_image.shape[0]
    blend_it.tangent_images_coordinates(erp_image_height, dispmap_aligned_list[0].shape)
    blend_it.erp_blendweights(subimg_cam_list, erp_image_height, dispmap_aligned_list[0].shape)
    blend_it.compute_linear_system_matrices(erp_image_height, erp_image_height * 2, blend_it.frustum_blendweights)

    erp_dispmap_blend = blend_it.blend(dispmap_aligned_list, erp_image_height)
    blending_method = 'poisson' if opt.blending_method == 'all' else opt.blending_method

    # CLEANUP: Free memory from the blending process
    del dispmap_aligned_list, subimg_cam_list, blend_it, depthmap_aligner_obj
    cleanup_cuda()

    # -------------------------------------------------------------------------
    # SURGICAL PIVOT V13: MEMORY OPTIMIZED
    # -------------------------------------------------------------------------
    
    # 1. Get Initial Disparity
    disp0 = erp_dispmap_blend[blending_method].astype(np.float32)
    disp0 = np.nan_to_num(disp0, nan=0.0, posinf=0.0, neginf=0.0)

    # 1b. Post-blend edge-preserving smoothing to reduce Poisson seam ripples.
    #     Uses a guided filter (O(1) edge-preserving) if OpenCV has it,
    #     otherwise falls back to bilateral filter.
    print("[train_headless] Applying post-blend seam smoothing...")
    try:
        # Guided filter: uses the disparity itself as the guide image.
        # radius=8, eps=0.01 * (range^2) preserves edges while damping ripples.
        d_range = float(np.percentile(disp0, 98) - np.percentile(disp0, 2))
        eps = 0.01 * (d_range ** 2)
        disp_smooth = cv2.ximgproc.guidedFilter(
            guide=disp0, src=disp0, radius=8, eps=eps
        )
        # Blend: keep 60% original structure, 40% smoothed
        disp0 = 0.6 * disp0 + 0.4 * disp_smooth
        print(f"[train_headless]   → Guided filter applied (eps={eps:.4f})")
    except (AttributeError, cv2.error):
        # cv2.ximgproc not available — fall back to bilateral
        try:
            d99 = max(1e-6, float(np.percentile(np.abs(disp0), 99)))
            dnorm = (disp0 / d99).astype(np.float32)
            dnorm = np.clip(dnorm, -1.0, 1.0)
            disp_bf = cv2.bilateralFilter(dnorm, d=9, sigmaColor=0.15, sigmaSpace=12)
            disp0 = 0.6 * disp0 + 0.4 * (disp_bf * d99)
            print("[train_headless]   → Bilateral filter fallback applied")
        except Exception as e:
            print(f"[train_headless]   → Smoothing skipped: {e}")
    
    # 2. Inpaint Holes
    mask_nan = (disp0 <= 0.001)
    if np.any(mask_nan):
        valid_coords = np.argwhere(~mask_nan)
        invalid_coords = np.argwhere(mask_nan)
        if valid_coords.shape[0] > 100000:
            idx = np.random.choice(valid_coords.shape[0], 100000, replace=False)
            valid_coords = valid_coords[idx]
        try:
            disp0[mask_nan] = griddata(valid_coords, disp0[~mask_nan][idx], invalid_coords, method='nearest')
        except:
            pass 

    # 3. Convert to DEPTH
    #    The Poisson blender outputs a signed relative field (-8.5 to +6.9).
    #    Key question: does higher value mean closer or farther?
    #    
    #    MiDaS outputs DISPARITY (higher = closer). But after Poisson blending
    #    with gradient constraints, the absolute relationship can flip.
    #    
    #    Empirically: the negative values correspond to far regions (walls behind),
    #    positive values to near regions. So higher = closer = should be SMALLER depth.
    #    We flip the sign: depth_relative = max_val - raw_val, then rescale.
    #    This avoids the 1/x nonlinearity which crushes dynamic range.
    
    v_all = disp0.flatten()
    p2 = float(np.percentile(v_all, 2))
    p98 = float(np.percentile(v_all, 98))
    
    # Flip: highest raw disparity (closest) → smallest depth
    # Rescale [p2, p98] → [near_m, far_m]  (near=close surfaces, far=distant walls)
    near_m, far_m = 1.0, 8.0
    drange = max(1e-6, p98 - p2)
    
    # Linear mapping: p98 (close) → near_m, p2 (far) → far_m
    depth = near_m + (far_m - near_m) * (p98 - disp0) / drange
    depth = np.clip(depth, 0.3, 15.0).astype(np.float32)
    
    print(f"[train_headless] Depth conversion: p2={p2:.2f}→{far_m}m, p98={p98:.2f}→{near_m}m")
    print(f"[train_headless] Depth stats: min={depth.min():.2f} max={depth.max():.2f} "
          f"mean={depth.mean():.2f} median={np.median(depth):.2f}")

    # 3b. Depth-space edge-preserving smooth to reduce seam ridges.
    #     Uses a larger kernel than the disparity-space pass since we're targeting
    #     low-frequency Poisson ripples, not high-frequency noise.
    try:
        d_lo = float(np.percentile(depth, 1))
        d_hi = float(np.percentile(depth, 99))
        d_norm = ((depth - d_lo) / max(1e-6, d_hi - d_lo)).astype(np.float32)
        d_norm = np.clip(d_norm, 0.0, 1.0)
        # Large kernel bilateral: d=15 smooths ~15px seam ripples,
        # sigmaColor=0.08 preserves depth edges (furniture vs wall)
        d_smooth = cv2.bilateralFilter(d_norm, d=15, sigmaColor=0.08, sigmaSpace=20)
        d_smooth = d_smooth * (d_hi - d_lo) + d_lo
        depth = (0.5 * depth + 0.5 * d_smooth).astype(np.float32)
        print("[train_headless]   → Depth-space bilateral smooth applied (d=15)")
    except Exception as e:
        print(f"[train_headless]   → Depth smooth skipped: {e}")

    # 4. Pole handling: mask extreme pole regions to avoid singularity spikes.
    #    At ERP poles (top/bottom rows), cos(phi)→0 collapses all points onto
    #    a single ray regardless of depth, creating spike artifacts.
    #    Strategy: replace pole regions with mid-band median depth (creates a
    #    smooth dome/floor rather than spikes). The Gaussian training fills gaps.
    H, W = depth.shape
    pole_frac = 0.08  # top/bottom 8%
    top_end = int(H * pole_frac)
    bot_start = H - top_end

    mid_band = depth[top_end:bot_start, :]
    mid_valid = mid_band[np.isfinite(mid_band) & (mid_band > 0)]
    mid_median = float(np.median(mid_valid)) if mid_valid.size > 0 else 2.0

    # Smooth blend toward median at poles
    for y in range(top_end):
        alpha = float(top_end - y) / float(top_end)  # 1.0 at row 0, 0.0 at boundary
        depth[y, :] = alpha * mid_median + (1.0 - alpha) * depth[top_end, :]

    for y in range(bot_start, H):
        alpha = float(y - bot_start + 1) / float(H - bot_start)  # 0→1
        depth[y, :] = alpha * mid_median + (1.0 - alpha) * depth[bot_start - 1, :]

    print(f"[train_headless] Pole regions blended toward median={mid_median:.2f}m")

    # -------------------------------------------------------------------------
    # NEW STEP 6.5: MLP GLOBAL ALIGNMENT (Neural Smoothing)
    # -------------------------------------------------------------------------
    print("[train_headless] Optimizing Full MLP Hybrid for global alignment...")
    
    # Generate View Vectors
    x_grid, y_grid = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
    theta = 2 * np.pi * (x_grid / W - 0.5) 
    phi = np.pi * (0.5 - y_grid / H)       
    
    v_x = np.cos(phi) * np.sin(theta)
    v_y = np.sin(phi)
    v_z = np.cos(phi) * np.cos(theta)
    v_vectors = np.stack([v_x, v_y, v_z], axis=-1).reshape(-1, 3) 
    
    v_tensor = torch.from_numpy(v_vectors).float().to(device)
    D_mono = torch.from_numpy(depth.flatten()).float().to(device)
    valid_mask = torch.isfinite(D_mono) & (D_mono < 20.0)

    # Setup Optimization
    mlp = DepthMLP().to(device)
    alpha = torch.tensor(1.0, requires_grad=True, device=device) 
    beta = torch.tensor(0.0, requires_grad=True, device=device) 
    
    optimizer = optim.Adam(list(mlp.parameters()) + [alpha, beta], lr=0.005)
    
    num_mlp_iters = 1000
    lambda_reg = 0.01

    # Training Loop
    for i in tqdm.trange(num_mlp_iters, desc="MLP Optimization"):
        optimizer.zero_grad()
        pred_depth = mlp(v_tensor).squeeze()
        target = alpha * D_mono + beta
        loss = torch.mean((pred_depth[valid_mask] - target[valid_mask])**2)
        loss += lambda_reg * (alpha - 1.0)**2
        loss.backward()
        optimizer.step()

    # Inference
    print("[train_headless] Generating refined depth from MLP...")
    with torch.no_grad():
        refined_flat = mlp(v_tensor).squeeze()
        refined_flat = (refined_flat - beta) / (alpha + 1e-6)
        refined_depth = refined_flat.reshape(H, W).cpu().numpy()
    
    depth = np.clip(refined_depth, 0.1, 20.0).astype(np.float32)

    # CLEANUP: Kill the MLP and tensors immediately
    del mlp, v_tensor, D_mono, optimizer, loss, valid_mask, pred_depth, target
    cleanup_cuda()

    # 7. FINAL UPSAMPLE (DIET MODE: 2x)
    print("[train_headless] Step 7/7: Upsampling pointcloud (2x Diet Mode)...")
    
    # V13 CHANGE: 2x Upsample instead of 4x (Saves ~12GB VRAM)
    high_res_h, high_res_w = H * 2, W * 2 
    
    depth_high = resize(depth, (high_res_h, high_res_w), order=1, anti_aliasing=True)
    depth_high = np.clip(depth_high, 0.1, 50.0).astype(np.float32)
    
# We add 'preserve_range=True' so values stay 0-255 instead of becoming 0-1
    erp_image_high = resize(erp_image.astype(np.float32), (high_res_h, high_res_w), 
                            order=1, anti_aliasing=True, preserve_range=True)
                            
    erp_image_high = np.clip(erp_image_high, 0, 255).astype(np.uint8)

    # Save
    pointcloud_path = os.path.join(log_dir, "pointcloud.ply")
    pointcloud_utils.depthmap2pointcloud_erp(depth_high, erp_image_high, pointcloud_path)
    
    # CLEANUP: Kill the high-res arrays before GS initialization
    del depth_high, erp_image_high, depth, erp_image
    cleanup_cuda()
    
    # --- 5. Initialize Gaussians ---
    print("[train_headless] Step 5/6: Initializing Gaussians...")
    renderer = Renderer(sh_degree=opt.sh_degree, white_background=opt.white_background)
    renderer.initialize(pointcloud_path)
    renderer.gaussians.training_setup(opt)

    Equirect = Equirectangular(pano_dst)
    cam = cam_utils.OrbitCamera(opt.W, opt.H, r=0.01, fovy=opt.fovy)

    # --- 6. Train ---
    print(f"[train_headless] Step 6/6: Training {iters} iterations...")
    for step in tqdm.trange(iters, desc="Training Gaussians"):
        renderer.gaussians.update_learning_rate(step)
        
        if np.random.random() < 0.2:
            ver = np.random.choice([-opt.max_ver + 5, opt.max_ver - 5])
        else:
            ver = np.random.randint(-opt.max_ver, opt.max_ver)
            
        hor = np.random.randint(-180, 180)
        # radius=0.01: camera at origin (center of room), looking outward.
        # For interior scenes, the panorama was captured from the center.
        pose = cam_utils.orbit_camera(ver, hor, radius=0.01)

        cur_cam = MiniCam(pose, opt.ref_size, opt.ref_size, cam.fovy, cam.fovx, cam.near, cam.far)
        bg_color = torch.tensor([1, 1, 1] if np.random.random() > opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device=device)

        out = renderer.render(cur_cam, bg_color=bg_color)
        image = out["image"]
        ref_image = Equirect.GetPerspective(math.degrees(cam.fovy), hor, -ver, opt.ref_size, opt.ref_size)
        ref_image = torch.from_numpy(ref_image.astype(np.float32) / 255.0).permute(2, 0, 1).to(device)

        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, ref_image) + opt.lambda_dssim * (1.0 - ssim(image, ref_image))
        loss.backward()
        renderer.gaussians.optimizer.step()
        renderer.gaussians.optimizer.zero_grad()

        if step < opt.density_end_iter and step > opt.density_start_iter and step % opt.densification_interval == 0:
            renderer.gaussians.add_densification_stats(out["viewspace_points"], out["visibility_filter"])
            renderer.gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_min_opacity, opt.densify_extent, opt.densify_max_screen_size)

    # --- 7. Save ---
    ply_path = os.path.join(log_dir, f"{scene_name}_gs.ply")
    renderer.gaussians.save_ply(ply_path)
    print(f"[train_headless] Saved GS PLY: {ply_path}")

    # Also save an Open3D-compatible PLY with standard RGB vertex colors.
    # The GS PLY uses SH coefficients (f_dc_0/1/2) which Open3D can't read as colors.
    try:
        import struct
        xyz = renderer.gaussians.get_xyz.detach().cpu().numpy()           # [N, 3]
        # SH DC band (band 0) to RGB: color = 0.5 + C0 * sh_dc
        # where C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        sh_dc = renderer.gaussians._features_dc.detach().cpu().numpy()    # [N, 1, 3]
        sh_dc = sh_dc.squeeze(1)                                          # [N, 3]
        rgb = np.clip(0.5 + C0 * sh_dc, 0.0, 1.0)                       # [N, 3] in [0,1]
        rgb_u8 = (rgb * 255).astype(np.uint8)

        o3d_ply_path = os.path.join(log_dir, f"{scene_name}_open3d.ply")
        N = xyz.shape[0]
        with open(o3d_ply_path, 'wb') as f:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {N}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )
            f.write(header.encode('utf-8'))
            for i in range(N):
                f.write(struct.pack('fff', xyz[i, 0], xyz[i, 1], xyz[i, 2]))
                f.write(struct.pack('BBB', rgb_u8[i, 0], rgb_u8[i, 1], rgb_u8[i, 2]))

        print(f"[train_headless] Saved Open3D PLY: {o3d_ply_path}")
    except Exception as e:
        print(f"[train_headless] (note) Open3D PLY export failed: {e}")

    return ply_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamScene360 headless training")
    parser.add_argument("--panorama", required=True, help="Path to 360 panorama image")
    parser.add_argument("--outdir", default="output")
    parser.add_argument("--scene_name", default="scene")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--pano_width", type=int, default=2048)
    parser.add_argument("--pano_height", type=int, default=1024)
    parser.add_argument("--sh_degree", type=int, default=3)

    args = parser.parse_args()
    opt = load_config({'pano_width': args.pano_width, 'pano_height': args.pano_height, 'sh_degree': args.sh_degree})

    train_from_panorama(args.panorama, args.outdir, args.scene_name, iters=args.iters, opt=opt)