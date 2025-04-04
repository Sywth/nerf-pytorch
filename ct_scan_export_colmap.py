# %%
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from ct_nerf_interp import scale_astra_ct_outputs
from ct_scan import *
import load_blender
# %%

def matrix_to_quaternion(matrix):
    """Convert a 3x3 rotation matrix to a quaternion (Hamilton convention)."""
    import scipy.spatial.transform
    return scipy.spatial.transform.Rotation.from_matrix(matrix).as_quat()
# %%

def export_npz_colmap(npz, phantom_idx=None):
    """
    Exports CT scan dataset into a COLMAP-compatible format.
    """
    # Define the export directory
    size = npz["images"].shape[1]
    num_scans = npz["images"].shape[0]
    
    export_path = Path(f"export/colmap_ct_data_{phantom_idx}_{size}_{num_scans}")
    export_path.mkdir(parents=True, exist_ok=True)

    images_path = export_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    sparse_path = export_path / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    # Store COLMAP camera metadata
    cameras_file = sparse_path / "cameras.txt"
    images_file = sparse_path / "images.txt"
    points3D_file = sparse_path / "points3D.txt"  # Optional (for sparse recon)

    # 1. Save Images
    for idx, img in enumerate(npz["images"]):
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img, mode="RGBA")
        img_path = images_path / f"{idx}.png"
        img_pil.save(img_path)

    # 2. Write `cameras.txt`
    width, height = npz["images"].shape[1:3]
    focal_length = 1e6  # Large value to approximate orthographic projection

    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 SIMPLE_PINHOLE {width} {height} {focal_length} {width//2} {height//2}\n")

    # 3. Write `images.txt`
    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for idx, pose in enumerate(npz["poses"]):
            # Extract rotation matrix and translation vector
            R = pose[:3, :3]
            t = pose[:3, 3]

            # Convert rotation matrix to quaternion
            qx, qy, qz, qw = matrix_to_quaternion(R)

            # Convert translation vector to COLMAP format
            tx, ty, tz = -R.T @ t  # World-to-camera transformation

            f.write(f"{idx+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {idx}.png\n")
            f.write("\n")  # Empty second line (no 3D points yet)

    # 4. Write `points3D.txt` (if needed)
    with open(points3D_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        # Leave empty since no sparse 3D points are generated

    print(f"COLMAP dataset successfully exported to {export_path}")
    return export_path
# %%

def create_nerf_ds_colmap(phantom_idx, phantom, scaled_ct_imgs, ct_poses, ct_angles):
    fov_deg = float("inf")

    scaled_ct_imgs = utils.mono_to_rgb(scaled_ct_imgs)
    scaled_ct_imgs = rgb_to_rgba(scaled_ct_imgs)
    scaled_ct_imgs = remove_bg(scaled_ct_imgs)

    npz_dict = get_npz_dict(scaled_ct_imgs, ct_poses, ct_angles, phantom, fov_deg)
    export_npz_colmap(npz_dict, phantom_idx)

if __name__ == "__main__":
    # Phatom params 
    phantom_idx = 13
    ph_size = 256
    num_scans = 32
    assert num_scans % 2 == 0, "Number of scans must be even"

    # scan parameters
    radius = 2.0
    img_res = 256
    hwf = (img_res, img_res, None)

    phantom = load_phantom(phantom_idx, ph_size)
    spherical_angles = np.array([
        [theta, 0.0] for theta in np.linspace(0, np.pi, num_scans, endpoint=False)
    ])
    poses = torch.stack([
        load_blender.pose_spherical_deg(np.rad2deg(theta), np.rad2deg(phi), radius=radius)
        for theta,phi in spherical_angles
    ])

    scan_2n = AstraScanVec3D(
        phantom.shape, spherical_angles, img_res=img_res
    )
    scan_n = AstraScanVec3D(
        phantom.shape, spherical_angles[::2], img_res=img_res
    )
    ct_imgs = scan_2n.generate_ct_imgs(phantom)
    ct_imgs = scale_astra_ct_outputs(ct_imgs)



    # </3
    create_nerf_ds_colmap(phantom_idx, phantom, ct_imgs, poses, spherical_angles)