# %%
from typing import Any
import numpy as np
import astra
import matplotlib.pyplot as plt
import tomophantom
import trimesh
import pyrender

import os
import json
import utils

from tomophantom import TomoP3D
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
from scipy.ndimage import rotate

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

    def __post_init__(self):
        self.vol_geom = astra.create_vol_geom(*self.phantom_shape)
        angles = self.get_angles_rad()
        self.detector_resolution = (
            np.array(self.phantom_shape[:2]) / self.inv_res
        ).astype(int)

        self.reset_angles(angles)

    def get_angles_rad(self):
        return np.linspace(
            self.min_theta, self.max_theta, self.num_scans, endpoint=False
        )
    
    def reset_angles(self, angles):
        self.num_scans = len(angles)
        self.proj_geom = astra.create_proj_geom(
            "parallel3d",
            self.inv_res,  # DetectorSpacingX
            self.inv_res,  # DetectorSpacingY
            self.detector_resolution[0],  # DetectorRowCount
            self.detector_resolution[1],  # DetectorColCount
            angles,  # ProjectionAngles
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


# TESTED : Works with ASTRA (Use rotation_axis='x')
def generate_camera_poses(angles: np.ndarray, radius: float = 4.0, axis="x"):
    """"""
    poses = [utils.compute_camera_matrix(theta, radius, axis) for theta in angles]
    return poses


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
    mesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(mesh)

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
    scene = pyrender.Scene()

    num_meshes = 4
    for i in range(num_meshes):
        node = scene.add(mesh)
        y_pos = (i / 10) - 0.15
        node.translation = np.array([0, y_pos, 0])

    camera_node = scene.add(camera, pose=cam_pose)
    scene.add(light1, pose=utils.compute_camera_matrix(np.pi / 2, 1, "y"))
    scene.add(light2, pose=utils.compute_camera_matrix(3 * (np.pi / 2), 1, "y"))

    renderer = pyrender.OffscreenRenderer(*image_size)
    return scene, camera_node, renderer


def generate_projections(
    num_views=16,
    image_size=(64, 64),
    mesh_path="./data/objs/fuze/fuze.obj",
    fov_deg: float = 90,
):
    """Generates 2D projections from different viewpoints."""

    # Generate camera poses
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    radius = 1.0
    poses = generate_camera_poses(angles, radius=radius)

    # Generate scene
    scene, camera_node, renderer = generate_scene(
        mesh_path, image_size, poses[0], fov_deg
    )

    images = []
    for i, pose in enumerate(poses):
        scene.set_pose(camera_node, pose=pose)
        color, _ = renderer.render(scene)
        images.append(color)

    renderer.delete()
    scene.clear()

    images = np.array(images) / 255.0

    return np.array(images), poses, angles


# NOTE : END : ADDITION
# %%


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
    ct_poses = generate_camera_poses(scan_params.get_angles_rad())

    if use_rgb:
        ct_imgs = utils.mono_to_rgb(ct_imgs)

    if normalize_instead_of_standardize:
        ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())
    else:
        # we standardize the images
        ct_imgs = (ct_imgs - ct_imgs.mean()) / ct_imgs.std()

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


remove_bg_threshold_white = 0.95
remove_bg_threshold_black = 0.05


def export_npz(npz, remove_bg=True):
    # Define the export directory
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
            # add extra 1 to the image to make it RGBA
            img = np.concatenate([img, np.ones(img.shape[:2] + (1,))], axis=-1)

            if remove_bg:
                # Extract the RGB channels (ignoring the alpha channel).
                rgb = img[..., :3]

                # Create boolean masks: a pixel is considered background if all RGB values are
                # above the white threshold or below the black threshold.
                white_mask = np.all(rgb >= remove_bg_threshold_white, axis=-1)
                black_mask = np.all(rgb <= remove_bg_threshold_black, axis=-1)
                bg_mask = np.logical_or(white_mask, black_mask)

                # Set the alpha channel (4th channel) to 0 where the background mask is True.
                img[..., 3][bg_mask] = 0

            # Scale to [0, 255] for saving and cast to uint8
            img = (img * 255).clip(0, 255).astype(np.uint8)

            # Convert to PIL Image in RGBA mode
            img_pil = Image.fromarray(img, mode="RGBA")

            # Save the image
            img_path = split_dir / f"{idx}.png"
            img_pil.save(img_path)

    print(f"Dataset successfully exported to {export_path}")


# %% [markdown]
# ### Export dataset
if __name__ == "__main__":

    ct_mode = False
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
            num_views=num_scans, image_size=(size, size), fov_deg=fov_deg
        )
        npz_dict = get_npz_dict(
            visible_imgs, visible_poses, visible_angles, phantom, fov_deg=fov_deg
        )
        export_npz(npz_dict)
