"""
ComfyUI-DreamScene360
Converts a 360° panorama into a gap-filled 3D Gaussian Splat point cloud.
"""

import traceback

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    print("\n" + "=" * 60)
    print("[DreamScene360] ERROR: Failed to load nodes.")
    print(f"[DreamScene360] {type(e).__name__}: {e}")
    print("[DreamScene360]")
    print("[DreamScene360] This usually means a required package is missing")
    print("[DreamScene360] or a CUDA submodule failed to build.")
    print("[DreamScene360]")
    print("[DreamScene360] To fix, run install.py from the node directory:")
    print("[DreamScene360]   python install.py")
    print("[DreamScene360]")
    print("[DreamScene360] On RunPod after a pod restart, run:")
    print("[DreamScene360]   bash install_dreamscene360.sh")
    print("[DreamScene360]")
    print("[DreamScene360] Full traceback:")
    traceback.print_exc()
    print("=" * 60 + "\n")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
