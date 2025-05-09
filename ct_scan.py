# %%
import json
import os
import random
import numpy as np
import astra
import matplotlib.pyplot as plt
import torch
import tomophantom
import trimesh
import pyrender
import cv2

from tomophantom import TomoP3D
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from typing import Any
from load_blender import pose_spherical_deg

import utils
import ct_scan_debug

# %%
tomo_root_path = os.path.dirname(tomophantom.__file__)
tomo_path = os.path.join(tomo_root_path, "phantomlib", "Phantom3DLibrary.dat")


# %% [markdown]
# ### For Generating CT Data
# %%
class AstraScanVec3D:
    def __init__(self, phantom_shape, angles, img_res=64.0):
        self.phantom_shape = phantom_shape
        self.angles = np.array(
            angles
        )  # shape: (num_projections, 2) where each is (theta, phi,)
        self.inv_res = phantom_shape[0] / img_res

        # These will be created in re_init()
        self.vol_geom = None
        self.proj_geom = None
        self.detector_resolution = None

        self.re_init()

    def re_init(self):
        self.vol_geom = astra.create_vol_geom(*self.phantom_shape)
        # Detector resolution derived from the phantom shape (using first two dimensions)
        self.detector_resolution = (
            np.array(self.phantom_shape[:2]) / self.inv_res
        ).astype(int)

        # Compute the 12-element projection vectors from angles.
        vectors = self._compute_projection_vectors()
        self.proj_geom = astra.create_proj_geom(
            "parallel3d_vec",
            self.detector_resolution[0],  # number of detector rows
            self.detector_resolution[1],  # number of detector columns
            vectors,
        )

    def get_ct_camera_poses(self, radius=2.0):
        poses = []
        for theta, phi in self.angles:
            # Compute the camera-to-world transformation using spherical coordinates.
            c2w = pose_spherical_deg(np.rad2deg(theta), np.rad2deg(phi), radius)
            if isinstance(c2w, torch.Tensor):
                c2w = c2w.detach().cpu().numpy()

            # Convert to a numpy array if necessary.
            poses.append(c2w.numpy() if hasattr(c2w, "numpy") else np.array(c2w))
        return np.stack(poses)

    def _compute_projection_vectors(self):
        # Get the camera poses (each a 4x4 matrix).
        poses = self.get_ct_camera_poses(radius=2.0)
        num_projections = poses.shape[0]
        vectors = np.zeros((num_projections, 12), dtype=np.float32)
        spacing = self.inv_res  # detector spacing for both X and Y

        for i, pose in enumerate(poses):
            # Extract the rotation matrix and translation vector from the pose.
            R = pose[:3, :3]  # 3x3 rotation
            t = pose[:3, 3]  # camera center in world coordinates

            # Compute the ray: a unit vector pointing from the camera toward the origin.
            norm_t = np.linalg.norm(t)
            if norm_t == 0:
                raise ValueError("Camera center cannot be at the origin.")
            ray = -t / norm_t  # unit ray direction

            # Using the NeRF convention (columns: [right, up, -view_dir]), extract:
            right = R[:, 0]  # detector's horizontal direction (u)
            up = R[:, 1]  # detector's vertical direction (v)

            # Assemble the 12-element vector: [ray, d, u, v].
            # Here, we choose d (detector center) to be [0, 0, 0].
            vectors[i, 0:3] = ray
            vectors[i, 3:6] = 0.0
            vectors[i, 6:9] = right * spacing
            vectors[i, 9:12] = up * spacing

        return vectors

    def generate_ct_imgs(self, phantom):
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError(
                "Projection geometry and volume geometry must be initialized."
            )

        volume_id = astra.data3d.create("-vol", self.vol_geom, phantom)
        proj_id = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom)
        sinogram_id, sinogram = astra.create_sino3d_gpu(
            volume_id, self.proj_geom, self.vol_geom, returnData=True
        )

        # Free resources
        astra.data3d.delete(volume_id)
        astra.projector.delete(proj_id)
        astra.data3d.delete(sinogram_id)

        # Rearranging axes if necessary (this step can be adjusted based on downstream use)
        return np.moveaxis(sinogram, 0, 1)

    def reconstruct_3d_volume_sirt(self, ct_imgs, num_iterations=64):
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError(
                "Projection geometry and volume geometry must be initialized."
            )

        sinogram = ct_imgs.swapaxes(0, 1)
        sinogram_id = astra.data3d.create("-proj3d", self.proj_geom, sinogram)
        reconstruction_id = astra.data3d.create("-vol", self.vol_geom)

        # Configure the reconstruction algorithm (here SIRT3D_CUDA is used)
        alg_cfg = astra.astra_dict("SIRT3D_CUDA")
        alg_cfg["ProjectionDataId"] = sinogram_id
        alg_cfg["ReconstructionDataId"] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)

        astra.algorithm.run(algorithm_id, num_iterations)
        reconstruction = astra.data3d.get(reconstruction_id)

        # Clean up
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(sinogram_id)
        astra.data3d.delete(reconstruction_id)

        return reconstruction

    def reconstruct_3d_volume_cgls(self, ct_imgs, num_iterations=64):
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError(
                "Projection geometry and volume geometry must be initialized."
            )

        sinogram = ct_imgs.swapaxes(0, 1)

        # Create ASTRA data objects for the sinogram and reconstruction volume
        sinogram_id = astra.data3d.create("-proj3d", self.proj_geom, sinogram)
        reconstruction_id = astra.data3d.create("-vol", self.vol_geom)

        # Configure the CGLS3D_CUDA reconstruction algorithm
        alg_cfg = astra.astra_dict("CGLS3D_CUDA")
        alg_cfg["ProjectionDataId"] = sinogram_id
        alg_cfg["ReconstructionDataId"] = reconstruction_id

        # Create and run the reconstruction algorithm
        algorithm_id = astra.algorithm.create(alg_cfg)
        astra.algorithm.run(algorithm_id, num_iterations)

        # Retrieve the reconstructed volume
        reconstruction = astra.data3d.get(reconstruction_id)

        # Clean up resources
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(sinogram_id)
        astra.data3d.delete(reconstruction_id)

        return reconstruction


# %% [markdown]
# ### For Generating Visual Data
# %%
def generate_visuale_scene(mesh_path, image_size, cam_pose, fov_deg, use_ortho=False):
    assert use_ortho == False, "Orthographic camera not supported yet"
    mesh_or_scene = trimesh.load(mesh_path)

    if isinstance(mesh_or_scene, trimesh.Scene):
        print("[SCENE GEN] Interpreting as 'Scene'")
        meshes = [
            pyrender.Mesh.from_trimesh(geometry)
            for geometry in mesh_or_scene.geometry.values()
        ]
    else:
        print("[SCENE GEN] Interpreting as 'Mesh'")
        meshes = [pyrender.Mesh.from_trimesh(mesh_or_scene)]

    # yfov = xfov because aspectRatio = 1.0
    fov = np.deg2rad(fov_deg)
    camera = pyrender.PerspectiveCamera(
        yfov=fov,
        aspectRatio=1.0,
    )
    light1 = pyrender.DirectionalLight(
        color=np.array([0.95, 0.35, 0.25]),
        intensity=30.0,
    )
    light2 = pyrender.DirectionalLight(
        color=np.array([0.08, 0.15, 0.90]),
        intensity=30.0,
    )
    bg_color = np.array([1.0, 0.0, 1.0, 0.0])  # Transparent background
    scene = pyrender.Scene(bg_color=bg_color)

    for mesh in meshes:
        scene.add(mesh)

    camera_node = scene.add(camera, pose=cam_pose)
    scene.add(light1, pose=utils.compute_camera_matrix(np.pi / 2, 1, "y"))
    scene.add(light2, pose=utils.compute_camera_matrix(3 * (np.pi / 2), 1, "y"))

    renderer = pyrender.OffscreenRenderer(*image_size)
    return scene, camera_node, renderer


DEBUG = False


def generate_visual_projections(
    mesh_path,
    num_views=16,
    image_size=(64, 64),
    fov_deg: float = 90,
):
    """Generates 2D projections from different viewpoints."""

    # Generate camera poses
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    poses = utils.generate_camera_poses(angles, radius=4.0, axis="y")

    # Generate scene
    scene, camera_node, renderer = generate_visuale_scene(
        mesh_path, image_size, poses[0], fov_deg
    )
    if DEBUG:
        ct_scan_debug.add_cardinal_axes(scene, axis_length=1.0)

    images = []
    flags = pyrender.RenderFlags.RGBA | 0

    camera_translate = np.array([0, 0.5, 0.0])

    for i, pose in enumerate(poses):
        pose[:3, 3] += camera_translate
        scene.set_pose(camera_node, pose=pose)

        color, _ = renderer.render(scene, flags=flags)
        images.append(color)

    renderer.delete()
    scene.clear()

    images = np.array(images) / 255.0

    return np.array(images), poses, angles


# %% [markdown]
# ### Auxillary Functions
# %%
def load_phantom(phantom_idx=4, size=256):
    """
    Some common phantom indices:

    Thin Cylinder                       : 1
    Big Cylinder                        : 2
    Verticle Cylinders                  : 3
    Cube Spheres                        : 4
    Cross of cylinders and ellipsoids   : 5
    Light Splotch                       : 6
    Lots of dots                        : 7
    Spheres in a disk                   : 8
    Spread out cube spheres             : 9
    SheppLogan                          : 13
    Head of Screw                       : 16
    Thick Shepp Logan                   : 17
    """
    phantom = TomoP3D.Model(phantom_idx, size, tomo_path)
    return phantom


def get_npz_dict(ct_imgs, ct_poses, angles, phantom, fov_deg):
    return {
        "images": ct_imgs,
        "poses": ct_poses,
        "phantom": phantom,
        "angles": angles,
        "camera_angle_x": np.deg2rad(fov_deg),
    }


def rgb_to_rgba(imgs: np.ndarray) -> np.ndarray:
    # (N, H, W, 3) -> (N, H, W, 4)
    assert imgs.ndim == 4, "Must be N-array of (H,W,3) images"
    assert imgs.shape[-1] == 3, "Must be rgb images"

    alpha_channel = np.ones((*imgs.shape[:-1], 1), dtype=imgs.dtype)
    rgba_imgs = np.concatenate([imgs, alpha_channel], axis=-1)
    return rgba_imgs


def remove_bg(imgs, white_threshold=0.0, black_threshold=0.05):
    # (N, H, W, 4) -> (N, H, W, 4)
    # Remove if pixel is
    assert imgs.ndim == 4, "Input must be 4D array"
    assert imgs.shape[-1] == 4, "Input must be RGBA images"

    rgb = imgs[..., :3]
    alpha = imgs[..., 3:]

    # White-bg mask : All rgb channels are above (1 - white_threshold)
    white_mask = np.all(rgb >= (1.0 - white_threshold), axis=-1, keepdims=True)

    # Black-bg mask : When all channels are below black_threshold
    black_mask = np.all(rgb <= black_threshold, axis=-1, keepdims=True)

    # Combine masks for both black and white background removal
    bg_mask = white_mask | black_mask

    # Set alpha to 0 for background pixels
    alpha[bg_mask] = 0.0
    imgs[..., 3:] = alpha

    return imgs


def export_npz(npz, phantom_idx=None):
    # Define the export directory
    size = npz["images"].shape[1]
    num_scans = npz["images"].shape[0]

    export_path = Path(
        f"export/ct_data_{phantom_idx}_{size}_{num_scans}_{random.randint(100,999)}"
    )
    export_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for train, test, val
    (export_path / "train").mkdir(parents=True, exist_ok=True)
    (export_path / "test").mkdir(parents=True, exist_ok=True)
    (export_path / "val").mkdir(parents=True, exist_ok=True)

    num_samples = len(npz["images"])
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices
    val_indices = []
    test_indices = []

    splits = {"train": train_indices, "val": val_indices, "test": test_indices}

    # NOTE : ToDo figure wtf this should be
    camera_angle_x = npz.get("camera_angle_x", 0.6911)

    # Helper function to create transform JSON files
    def create_transform_file(split_name, indices):
        frames = []
        for idx in indices:
            frames.append(
                {
                    "file_path": f"./{split_name}/{idx}",
                    "transform_matrix": npz["poses"][idx].tolist(),
                }
            )
        transform_dict = {"camera_angle_x": camera_angle_x, "frames": frames}
        with open(export_path / f"transforms_{split_name}.json", "w") as f:
            json.dump(transform_dict, f, indent=4)

    # Create and save transform files
    for split_name, indices in splits.items():
        create_transform_file(split_name, indices)

    for split_name, indices in splits.items():
        split_dir = export_path / split_name
        for idx in indices:
            img = npz["images"][idx]  # Get the image

            # Scale to [0, 255] for saving and cast to uint8
            img = (img * 255).clip(0, 255).astype(np.uint8)

            # Convert to PIL Image in RGBA mode
            img_pil = Image.fromarray(img, mode="RGBA")

            # Save the image
            img_path = split_dir / f"{idx}.png"
            img_pil.save(img_path)

    print(f"Dataset successfully exported to {export_path}")

    return export_path


# %%
def lerp_ct_imgs(ct_imgs_even):
    """
    (N, H, W) -> (N * 2, H, W)
    """
    n = ct_imgs_even.shape[0]
    ct_imgs_lerp = np.zeros((n * 2, *ct_imgs_even.shape[1:]), dtype=ct_imgs_even.dtype)
    ct_imgs_lerp[::2] = ct_imgs_even
    ct_imgs_odd = (ct_imgs_even[:-1] + ct_imgs_even[1:]) / 2
    ct_imgs_odd = np.concatenate(
        [ct_imgs_odd, [(ct_imgs_even[-1] + ct_imgs_even[0]) / 2]]
    )
    ct_imgs_lerp[1::2] = ct_imgs_odd
    return ct_imgs_lerp


# %%


def lanczos_ct_imgs(ct_imgs_even):
    """
    Interpolate sinogram images using Lanczos interpolation along the projection axis.

    For each pixel location (i,j), the 1D array over the N views is resized
    from length N to length 2N using cv2.resize with Lanczos interpolation.
    The function ensures that the original sinogram values appear at even indices.

    Parameters:
        ct_imgs_even (np.ndarray): Input sinogram of shape (N, H, W)

    Returns:
        np.ndarray: Interpolated sinogram of shape (2N, H, W)
    """
    N, H, W = ct_imgs_even.shape
    ct_imgs_interp = np.zeros((2 * N, H, W), dtype=ct_imgs_even.dtype)

    # Loop over spatial dimensions to interpolate along the sinogram (angle) axis
    for i in range(H):
        for j in range(W):
            # Extract 1D signal for pixel (i, j)
            col = ct_imgs_even[:, i, j].astype(np.float32)  # shape: (N,)
            col = col.reshape(N, 1)  # treat as a column image of shape (N, 1)
            # Resize from (N, 1) to (2N, 1) using Lanczos interpolation
            col_interp = cv2.resize(col, (1, 2 * N), interpolation=cv2.INTER_LANCZOS4)
            ct_imgs_interp[:, i, j] = col_interp[:, 0]

    # Ensure that the original views remain exactly at even indices
    ct_imgs_interp[::2] = ct_imgs_even
    return ct_imgs_interp


# %%
from skimage.metrics import structural_similarity as ssim_metric


def compute_ssim(gt_img: np.ndarray, pred_img: np.ndarray) -> float:
    ssim_value = ssim_metric(gt_img, pred_img, data_range=gt_img.max() - gt_img.min())
    return ssim_value


def plot_reconstructions(
    scan, ct_imgs, phantom, ph_size, title="Reconstructions", num_dp=4
) -> np.ndarray:
    print(f"\nReconstructing from {ct_imgs.shape[0]} views...")

    recon_sirt = scan.reconstruct_3d_volume_sirt(ct_imgs)

    test_idx = ph_size // 2
    recon_sirt_slice = recon_sirt[test_idx]
    phantom_slice = phantom[test_idx]

    psnr_sirt = utils.psnr(phantom, recon_sirt)
    ssim_sirt = compute_ssim(phantom_slice, recon_sirt_slice)

    psnr_sirt = round(psnr_sirt, num_dp)
    ssim_sirt = round(ssim_sirt, num_dp)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    axs[0].imshow(recon_sirt_slice, cmap="gray")
    axs[0].set_title(f"SIRT Recon. (PSNR: {psnr_sirt}) (SSIM: {ssim_sirt})", fontsize=8)
    axs[1].imshow(phantom_slice, cmap="gray")
    axs[1].set_title("Phantom GT")
    plt.show()
    return recon_sirt


# %% [markdown]
# ### Main
# %%
if __name__ == "__main__":
    phantom_idx = 13
    ph_size = 256
    num_scans = 64
    assert num_scans % 2 == 0, "Number of scans must be even"

    phantom = load_phantom(phantom_idx, ph_size)
    spherical_angles = np.array(
        [[theta, 0.0] for theta in np.linspace(0, np.pi, num_scans, endpoint=False)]
    )

    scan_2n = AstraScanVec3D(phantom.shape, spherical_angles, img_res=256)
    scan_n = AstraScanVec3D(phantom.shape, spherical_angles[::2], img_res=256)
    ct_imgs = scan_2n.generate_ct_imgs(phantom)

    # Every 2nd image
    ct_imgs_even = ct_imgs[::2]
    ct_imgs_lanczos = lanczos_ct_imgs(ct_imgs_even)
    ct_imgs_lerp = lerp_ct_imgs(ct_imgs_even)
    # plot_reconstructions(ct_imgs_even, phantom, size, title=f"Reconstruction from {num_scans} views")

    plot_reconstructions(
        scan_2n,
        ct_imgs,
        phantom,
        ph_size,
        title=f"[Full Orignial] Reconstruction from {num_scans} views",
    )
    plot_reconstructions(
        scan_n,
        ct_imgs_even,
        phantom,
        ph_size,
        title=f"[Half Orignial] Reconstruction from {num_scans // 2} views",
    )

    plot_reconstructions(
        scan_2n,
        ct_imgs_lerp,
        phantom,
        ph_size,
        title=f"[Half Orignial, Half Lerp] Reconstruction from {num_scans} views",
    )
    plot_reconstructions(
        scan_2n,
        ct_imgs_lanczos,
        phantom,
        ph_size,
        title=f"[Half Orignial, Half lanczos] Reconstruction from {num_scans} views",
    )

    # Create GIFs of one interpolation method at a tiem
    ct_imgs_interp = ct_imgs_lanczos
    method = "lanczos"

    ct_bp_interp = scan_2n.reconstruct_3d_volume_cgls(ct_imgs_interp)
    ct_bp = scan_2n.reconstruct_3d_volume_cgls(ct_imgs)

    render_gifs = True
    if render_gifs:
        print("Creating GIFs...")
        prefix = f"[{phantom_idx}_{ph_size}] "

        # Scan-wise GIFs
        utils.create_gif(
            ct_imgs,
            np.round(spherical_angles, 3),
            "Scan at angle {}",
            f"./figs/temp/{prefix}ct_scan.gif",
            fps=8,
        )
        utils.create_gif(
            ct_imgs_lerp,
            np.round(ct_imgs_lerp, 3),
            "Scan at angle {}",
            f"./figs/temp/{prefix}ct_scan_{method}.gif",
            fps=16,
        )

        # Slice-wise GIFs
        every_n = 4
        utils.create_gif(
            ct_bp[::every_n],
            np.arange(0, ct_bp.shape[0], every_n),
            "Reconstruction slice {}",
            f"./figs/temp/{prefix}ct_bp.gif",
            fps=12,
        )
        utils.create_gif(
            ct_bp_interp[::every_n],
            [
                str(utils.psnr(ct_bp_interp[i], ct_bp[i]))
                for i in range(0, ct_bp.shape[0], every_n)
            ],
            "Reconstruction PSNR {}",
            f"./figs/temp/{prefix}ct_bp_{method}.gif",
            fps=12,
        )
        print("GIFs DONE")
