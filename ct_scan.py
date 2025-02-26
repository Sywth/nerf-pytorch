# %%
import os
import json
import numpy as np
import astra
import matplotlib.pyplot as plt
import torch
import tomophantom
import trimesh
import pyrender

from tomophantom import TomoP3D
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from scipy.ndimage import rotate
from typing import Any
from scipy.spatial.transform import Rotation

import utils
import ct_scan_debug
import override_plotting
# %% [markdown]
# Setup initial consts / config
#

# %%
tomo_root_path = os.path.dirname(tomophantom.__file__)
tomo_path = os.path.join(tomo_root_path, "phantomlib", "Phantom3DLibrary.dat")


# %% [markdown]
# ## CT Definitions
#

def save_phantom_gif(phantom, size):
    utils.create_gif(
        phantom, np.arange(0, size, 1), "Slice {}", "figs/tomophatom_test.gif"
    )


# %% [markdown]
# #### Temp Dataset Loading Phase
#
# to be replaced with CT scan
#
# %%
@dataclass
class AstraScanParameters:
    phantom_shape: tuple
    num_scans: int = 64
    inv_res: float = 1.0
    min_theta: float = 0
    max_theta: float = 2 * np.pi

    proj_geom: Any | None = None
    vol_geom: Any | None = None

    def re_init(self):
        """
        Reset using the current parameters
        """
        self.vol_geom = astra.create_vol_geom(*self.phantom_shape)
        self.detector_resolution = (
            np.array(self.phantom_shape[:2]) / self.inv_res
        ).astype(int)

        angles = self.get_angles_rad()
        self.proj_geom = astra.create_proj_geom(
            "parallel3d",
            self.inv_res,  # DetectorSpacingX
            self.inv_res,  # DetectorSpacingY
            self.detector_resolution[0],  # DetectorRowCount
            self.detector_resolution[1],  # DetectorColCount
            angles,  # ProjectionAngles
        )

    def __post_init__(self):
        self.re_init()

    def get_angles_rad(self):
        return np.linspace(
            self.min_theta, self.max_theta, self.num_scans, endpoint=False
        )

    def reconstruct_3d_volume_alg(self, sinogram, num_iterations=4):
        """
        Note this will only work for sinograms generated using this instance
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("proj_geom and vol_geom must be set before reconstruction")

        sinogram_id = astra.data3d.create("-proj3d", self.proj_geom, sinogram)
        reconstruction_id = astra.data3d.create("-vol", self.vol_geom)

        # Initialize algorithm parameters
        alg_cfg = astra.astra_dict("SIRT3D_CUDA")
        alg_cfg["ProjectionDataId"] = sinogram_id
        alg_cfg["ReconstructionDataId"] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)

        astra.algorithm.run(algorithm_id, num_iterations)
        reconstruction = astra.data3d.get(reconstruction_id)

        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(sinogram_id)
        astra.data3d.delete(reconstruction_id)

        return reconstruction

    def generate_ct_imgs(self, phantom):
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError(
                "proj_geom and vol_geom must be set before sinogram generation"
            )

        volume_id = astra.data3d.create("-vol", self.vol_geom, phantom)
        proj_id = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom)
        sinogram_id, sinogram = astra.create_sino3d_gpu(
            volume_id, self.proj_geom, self.vol_geom, returnData=True
        )

        # Free GPU memory
        astra.data3d.delete(volume_id)
        astra.projector.delete(proj_id)
        astra.data3d.delete(sinogram_id)

        return np.moveaxis(sinogram, 0, 1)


class AstraScanVec3D:
    def __init__(self, phantom_shape, euler_angles, img_res = 64.0):
        self.phantom_shape = phantom_shape
        self.euler_angles = np.array(euler_angles)  # shape: (num_projections, 3)
        self.inv_res = phantom_shape[0] / img_res
        
        # These will be created in re_init()
        self.vol_geom = None
        self.proj_geom = None
        self.detector_resolution = None
        
        self.re_init()
    
    def re_init(self):
        self.vol_geom = astra.create_vol_geom(*self.phantom_shape)
        # Detector resolution derived from the phantom shape (using first two dimensions)
        self.detector_resolution = (np.array(self.phantom_shape[:2]) / self.inv_res).astype(int)
        
        # Compute the 12-element projection vectors from Euler angles.
        vectors = self._compute_projection_vectors()
        self.proj_geom = astra.create_proj_geom(
            'parallel3d_vec',
            self.detector_resolution[0],  # number of detector rows
            self.detector_resolution[1],  # number of detector columns
            vectors
        )
    
    def _compute_projection_vectors(self):
        num_projections = len(self.euler_angles)
        vectors = np.zeros((num_projections, 12), dtype=float)
        for i, (theta, phi, gamma) in enumerate(self.euler_angles):
            R = self._euler_to_rotation_matrix(theta, phi, gamma)
            # Define base directions:
            base_ray = np.array([0, 0, 1])
            base_u = np.array([1, 0, 0])  # direction for detector column increase
            base_v = np.array([0, 1, 0])  # direction for detector row increase
            
            ray = R @ base_ray
            # Center of detector is set to origin (could be offset if needed)
            d = np.zeros(3)
            # Scale the detector pixel vectors by inv_res (serves as DetectorSpacing)
            u = R @ base_u * self.inv_res
            v = R @ base_v * self.inv_res
            
            # Pack the 12 elements: ray, detector center, u, and v
            vectors[i, 0:3] = ray
            vectors[i, 3:6] = d
            vectors[i, 6:9] = u
            vectors[i, 9:12] = v
        return vectors

    def _euler_to_rotation_matrix(self, theta, phi, gamma):
        return Rotation.from_euler("ZYX", [theta, phi, gamma], degrees=False).as_matrix()

    def get_ct_camera_poses(self, radius=2.0):
        """
        # TODO
        This is not workign inline with astra but im going assume its fine for most 360 spin cases 
        """
        num_projections = len(self.euler_angles)
        poses = np.zeros((num_projections, 4, 4), dtype=float)

        # Obtain the 12-element vectors for each projection from the astra geometry.
        vectors = astra.geom_2vec(self.proj_geom)['Vectors']

        # With reference to astra https://astra-toolbox.com/docs/geom3d.html#projection-geometries
            # ray : the ray direction
            # d : the center of the detector
            # u : the vector from detector pixel (0,0) to (0,1)
            # v : the vector from detector pixel (0,0) to (1,0)

        for i, (rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ) in enumerate(vectors):
            # Extract and normalize the ray direction.
            ray = np.array([rayX, rayY, rayZ])
            ray /= np.linalg.norm(ray)

            # Use the 'u' vector to define the right direction (normalize it).
            right = np.array([uX, uY, uZ])
            right /= np.linalg.norm(right)

            # Compute the camera's up vector to form a right-handed coordinate system.
            # (Cross product of ray and right; note: order matters.)
            up = np.cross(ray, right)
            up /= np.linalg.norm(up)

            # In our convention the camera looks along -z, so we want:
            # R @ [0, 0, -1] = ray  =>  R's third column = -ray.
            R = np.column_stack((right, up, -ray))

            # Position the camera at -radius along the ray direction so that
            # the vector from the camera to the origin is aligned with ray.
            t = -radius * ray

            # Build the 4x4 pose matrix (camera-to-world).
            pose = np.eye(4, dtype=float)
            pose[:3, :3] = R
            pose[:3, 3] = t

            poses[i] = pose

        return poses

    def generate_ct_imgs(self, phantom):
        """
        Generate CT projection data (sinograms) from the given phantom.
        
        Parameters:
            phantom (np.ndarray): The 3D phantom (volume) data.
            
        Returns:
            np.ndarray: The sinogram data with axes rearranged appropriately.
            
        Raises:
            ValueError: If the projection or volume geometry has not been initialized.
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("Projection geometry and volume geometry must be initialized.")
        
        volume_id = astra.data3d.create("-vol", self.vol_geom, phantom)
        proj_id = astra.create_projector("cuda3d", self.proj_geom, self.vol_geom)
        sinogram_id, sinogram = astra.create_sino3d_gpu(volume_id, self.proj_geom, self.vol_geom, returnData=True)
        
        # Free resources
        astra.data3d.delete(volume_id)
        astra.projector.delete(proj_id)
        astra.data3d.delete(sinogram_id)
        
        # Rearranging axes if necessary (this step can be adjusted based on downstream use)
        return np.moveaxis(sinogram, 0, 1)
    
    def reconstruct_3d_volume_sirt(self, ct_imgs, num_iterations=64):
        """
        Reconstruct a 3D volume from the given sinogram data using the SIRT3D_CUDA algorithm.
        
        Parameters:
            sinogram (np.ndarray): The projection data.
            num_iterations (int): Number of iterations for the reconstruction algorithm.
            
        Returns:
            np.ndarray: The reconstructed 3D volume.
            
        Raises:
            ValueError: If the projection or volume geometry has not been initialized.
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("Projection geometry and volume geometry must be initialized.")
        
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
        """
        Reconstruct a 3D volume from the given sinogram data using the CGLS3D_CUDA algorithm.

        Parameters:
            sinogram (np.ndarray): The projection data (sinogram).
            num_iterations (int): Number of iterations for the CGLS algorithm.

        Returns:
            np.ndarray: The reconstructed 3D volume.

        Raises:
            ValueError: If the projection or volume geometry has not been initialized.
        """
        if self.proj_geom is None or self.vol_geom is None:
            raise ValueError("Projection geometry and volume geometry must be initialized.")

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




# %%
def load_phantom(phantom_idx=4, size=256):
    """
    Some common phantom indices:
    "Snake"         : 1
    "Defrise"       : 2
    "Cube Spheres"  : 4
    "SheppLogan"    : 13
    """
    phantom = TomoP3D.Model(phantom_idx, size, tomo_path)
    return phantom


# %%
# NOTE : START : ADDITION


def generate_scene(mesh_path, image_size, cam_pose, fov_deg, use_ortho=False):
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
    bg_color = np.array([1.0, 0.0, 1.0, 0.0]) # Transparent background
    scene = pyrender.Scene(bg_color=bg_color)

    for mesh in meshes:
        scene.add(mesh)

    camera_node = scene.add(camera, pose=cam_pose)
    scene.add(light1, pose=utils.compute_camera_matrix(np.pi / 2, 1, "y"))
    scene.add(light2, pose=utils.compute_camera_matrix(3 * (np.pi / 2), 1, "y"))

    renderer = pyrender.OffscreenRenderer(*image_size)
    return scene, camera_node, renderer

DEBUG = False
def generate_projections(
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
    scene, camera_node, renderer = generate_scene(
        mesh_path, image_size, poses[0], fov_deg
    )
    if DEBUG:
        ct_scan_debug.add_cardinal_axes(scene, axis_length=1.0)

    images = []
    flags =pyrender.RenderFlags.RGBA | 0

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


# NOTE : END : ADDITION
# %%

def rgb_to_rgba(imgs : np.ndarray) -> np.ndarray:
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

def acquire_ct_data(
    phantom: np.ndarray,
    num_scans=16,
    use_rgb=True,
    normalize_instead_of_standardize=True,
):
    scan_params = AstraScanParameters(
        phantom_shape=phantom.shape,
        num_scans=num_scans,
    )
    ct_imgs = scan_params.generate_ct_imgs(phantom)
    ct_poses = utils.generate_camera_poses(
        angles=scan_params.get_angles_rad(), 
        radius=4.0,
        axis="x"
    )

    if use_rgb:
        ct_imgs = utils.mono_to_rgb(ct_imgs)

    if normalize_instead_of_standardize:
        ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())
    else:
        # we standardize the images
        ct_imgs = (ct_imgs - ct_imgs.mean()) / ct_imgs.std()

    # Remove background and cast to RGBA
    ct_imgs = rgb_to_rgba(ct_imgs)
    ct_imgs = remove_bg(ct_imgs)

    return ct_imgs, ct_poses, scan_params.get_angles_rad()


# %%


def test_plot(ct_imgs, ct_poses):
    test_idx = np.random.randint(0, ct_imgs.shape[0])
    plt.title(f"Sinogram slice {test_idx}")
    plt.imshow(ct_imgs[test_idx])
    plt.colorbar()
    plt.show()

    # Rotate the phantom by the corresponding camera pose
    rotated_phantom = utils.rotate_phantom_by_pose(phantom, ct_poses[test_idx])

    # Visualize a central slice of the rotated phantom
    plt.figure()
    plt.title(f"Rotated Phantom Slice for Camera Pose {test_idx}")
    plt.imshow(rotated_phantom.sum(axis=1))
    plt.colorbar()
    plt.show()


def get_npz_dict(ct_imgs, ct_poses, angles, phantom, fov_deg):
    return {
        "images": ct_imgs,
        "poses": ct_poses,
        "phantom": phantom,
        "angles": angles,
        "camera_angle_x": np.deg2rad(fov_deg),
    }


def export_npz(npz, phantom_idx=None):
    # Define the export directory
    size = npz["images"].shape[1]
    num_scans = npz["images"].shape[0]

    export_path = Path(
        f"export/ct_data_{phantom_idx}_{size}_{num_scans}_{np.random.randint(0, 1000)}/"
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
def plot_reconstructions(scan, ct_imgs, phantom, size, title="Reconstructions"):
    recon_cgls = scan.reconstruct_3d_volume_cgls(ct_imgs)
    recon_sirt = scan.reconstruct_3d_volume_sirt(ct_imgs)
    test_idx = (size // 2) 

    recon_cgls_slice = recon_cgls[test_idx]
    recon_sirt_slice = recon_sirt[test_idx]
    phantom_slice = phantom[test_idx]
    psnr_cgls = utils.psnr(phantom, recon_cgls)
    psnr_sirt = utils.psnr(phantom, recon_sirt)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title) 
    axs[0].imshow(recon_cgls_slice, cmap="gray")
    axs[0].set_title(f"CGLS Reconstruction (PSNR: {psnr_cgls:.2f})")
    axs[1].imshow(recon_sirt_slice, cmap="gray")
    axs[1].set_title(f"SIRT Reconstruction (PSNR: {psnr_sirt:.2f})")
    axs[2].imshow(phantom_slice, cmap="gray")
    axs[2].set_title("Phantom Ground Truth")
    plt.show()

# %% [markdown]
# ### Export dataset

def lerp_ct_imgs(ct_imgs_even):
    """
    (N, H, W) -> (N * 2, H, W)
    """
    n = ct_imgs_even.shape[0]
    ct_imgs_lerp = np.zeros((n * 2, *ct_imgs_even.shape[1:]), dtype=ct_imgs_even.dtype)
    ct_imgs_lerp[::2] = ct_imgs_even
    ct_imgs_odd = (ct_imgs_even[:-1] + ct_imgs_even[1:]) / 2
    ct_imgs_odd = np.concatenate([ct_imgs_odd, [((ct_imgs_even[-1] + ct_imgs_even[0]) / 2)]])
    ct_imgs_lerp[1::2] = ct_imgs_odd
    return ct_imgs_lerp
# %%
import cv2

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
if __name__ == "__main__":
    phantom_idx = 13
    size = 256
    num_scans = 8
    assert num_scans % 2 == 0, "Number of scans must be even"
    use_rgb = True

    phantom = load_phantom(phantom_idx, size)
    euler_poses = np.array([
        [0.0, theta, 0.0] for theta in np.linspace(0, np.pi, num_scans, endpoint=False)
    ])

    scan_2n = AstraScanVec3D(
        phantom.shape, euler_poses, img_res=256
    )
    scan_n = AstraScanVec3D(
        phantom.shape, euler_poses[::2], img_res=256
    )
    ct_imgs = scan_2n.generate_ct_imgs(phantom)

    # Every 2nd image
    ct_imgs_even = ct_imgs[::2]
    ct_imgs_lanczos = lanczos_ct_imgs(ct_imgs_even)
    ct_imgs_lerp = lerp_ct_imgs(ct_imgs_even)
    # plot_reconstructions(ct_imgs_even, phantom, size, title=f"Reconstruction from {num_scans} views")

    plot_reconstructions(scan_2n, ct_imgs, phantom, size, title=f"[Full Orignial] Reconstruction from {num_scans} views")
    plot_reconstructions(scan_n, ct_imgs_even, phantom, size, title=f"[Half Orignial] Reconstruction from {num_scans // 2} views")

    plot_reconstructions(scan_2n, ct_imgs_lerp, phantom, size, title=f"[Half Orignial, Half Lerp] Reconstruction from {num_scans} views")
    plot_reconstructions(scan_2n, ct_imgs_lanczos, phantom, size, title=f"[Half Orignial, Half lanczos] Reconstruction from {num_scans} views")

# %%
def old():
    ct_mode = True
    visible_mode = False

    phantom_idx = 13
    size = 320
    num_scans = 64
    use_rgb = True

    phantom = load_phantom(phantom_idx, size)
    if ct_mode:
        fov_deg = float("inf")
        ct_imgs, ct_poses, ct_angles = acquire_ct_data(phantom, num_scans, use_rgb)
        npz_dict = get_npz_dict(ct_imgs, ct_poses, ct_angles, phantom, fov_deg)
        export_npz(npz_dict)

    if visible_mode:
        fov_deg = 26.0
        visible_imgs, visible_poses, visible_angles = generate_projections(
            "./data/objs/test_ct/test_ct.obj",
            num_views=num_scans,
            image_size=(size, size),
            fov_deg=fov_deg,
        )
        npz_dict = get_npz_dict(
            visible_imgs,
            visible_poses,
            visible_angles,
            phantom,
            fov_deg=fov_deg,
        )
        export_npz(npz_dict)
