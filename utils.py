import os
import numpy as np
import cv2
import torch

import datetime
import random
import string

from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import affine_transform

def ceildiv(a, b):
    return -(-a // b)


def extract_frames(
    video_path,
    output_folder,
    frame_ratio,
    width=None,
    height=None,
) -> int:
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    save_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decide whether to save this frame
        if frame_count % int(1 / frame_ratio) == 0:
            # If width and height are specified, crop the frame
            if width is not None and height is not None:
                h, w, _ = frame.shape
                startx = w // 2 - width // 2
                starty = h // 2 - height // 2
                frame = frame[starty : starty + height, startx : startx + width]

            # Save frame
            cv2.imwrite(
                os.path.join(output_folder, f"frame.{save_count:04d}.png"), frame
            )
            save_count += 1
        frame_count += 1

    cap.release()
    print(f"Finished: Extracted {save_count} frames to {output_folder}")
    return save_count


def create_gif(
    images,
    labels=None,
    template_str=None,
    output_filename=None,
    fps=10,
):
    if labels is None:
        labels = np.arange(len(images)) + 1
    if template_str is None:
        template_str = "Image {}"
    if output_filename is None:
        output_filename = f"figs/Animation {np.random.randint(0, 1000)}.gif"

    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(images[0], cmap="viridis")
    plt.colorbar(img, label="Intensity")

    def update(frame):
        """Update function for animation"""
        # Clear the current plot
        ax.clear()

        # Create new image
        img = ax.imshow(images[frame], cmap="viridis")
        ax.set_title(template_str.format(labels[frame]))

        return [img]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(images),
        interval=1000 / fps,  # interval in milliseconds
        blit=True,
    )
    anim.save(output_filename, writer="pillow", fps=fps)
    plt.close()


def vec3_to_rot_mat(euler_angles: np.array):
    """
    Cast <theta,phi,gamma> to a 4x4 rotation matrix
    """
    rotation = Rotation.from_euler("xyz", euler_angles)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    return rotation_matrix


def vec3_to_tran_mat(translation: np.array):
    """
    Cast <x,y,z> to a 4x4 translation matrix
    """
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation
    return translation_matrix


def get_rotation_vector(theta: float, rotation_axis: str):
    rotation_mapping = {
        "x": lambda x: np.array([x, 0.0, 0.0]),
        "y": lambda x: np.array([0.0, x, 0.0]),
        "z": lambda x: np.array([0.0, 0.0, x]),
        "xy": lambda x: np.array([x, x, 0.0]),
        "xz": lambda x: np.array([x, 0.0, x]),
        "yz": lambda x: np.array([0.0, x, x]),
        "xyz": lambda x: np.array([x, x, x]),
    }
    return rotation_mapping[rotation_axis](theta)


def compute_camera_matrix(
    theta: float,
    radius: float = 4.0,
    rotation_axis: str = "y",
):
    rot_euler = get_rotation_vector(theta, rotation_axis)
    rot_mat = vec3_to_rot_mat(rot_euler)
    tran_mat = vec3_to_tran_mat(np.array([0.0, 0.0, radius]))
    return rot_mat @ tran_mat


def get_ct_cam_mats(
    steps: int = 16,
    radius: float = 4.0,
    min_theta: float = 0.0,
    max_theta: float = 2 * np.pi,
    rotation_axis: str = "y",
):
    rotations = np.linspace(min_theta, max_theta, steps)
    mats = [compute_camera_matrix(theta, radius, rotation_axis) for theta in rotations]
    return np.stack(mats)


def rotate_phantom_by_pose(phantom: np.ndarray, cam_pose: np.ndarray):
    """
    Rotate the phantom by the camera pose.
    """

    assert type(phantom) == np.ndarray, "Phantom must be a numpy array"
    assert type(cam_pose) == np.ndarray, "Camera pose must be a numpy array"

    # Extract rotation and translation components from the pose matrix
    rotation_matrix = cam_pose[:3, :3]
    translation_vector = cam_pose[:3, 3]

    # Correct rotation direction
    rotation_matrix = np.linalg.inv(rotation_matrix)

    # Compute the center of the phantom for rotation
    center = np.array(phantom.shape) / 2.0
    offset = center - np.dot(rotation_matrix, center) - translation_vector

    # Apply affine transformation
    rotated_phantom = affine_transform(
        phantom, rotation_matrix, offset=offset, order=1, mode="constant", cval=0
    )
    return rotated_phantom


def mono_to_rgb(image):
    """
    Convert image.shape == (H, W) to image.shape == (H, W, 3)
    """
    return np.stack([image] * 3, axis=-1)


def rgb_to_mono(image):
    """
    Convert image.shape == (H, W, 3) to image.shape == (H, W)
    """
    return np.mean(image, axis=-1)


def lerp(v1, v2, t):
    return v1 + t * (v2 - v1)

def psnr(img1 : np.ndarray, img2 : np.ndarray):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# %% [markdown]
# Functions for NeRF


# %% [markdown]
# Mimic Tensorflow functions
#
# %%
def mesh_grid_xy(t1, t2):
    ii, jj = torch.meshgrid(t1, t2, indexing="ij")
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


# %%
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    dim = -1
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.0
    return cumprod


# %%
# %%
def get_ortho_rays(
    width: int,
    height: int,
    pixel_size: float,
    cam2world: torch.Tensor,
):
    # Create a grid of pixel coordinates (in image plane space).
    ii, jj = mesh_grid_xy(
        torch.arange(width).to(cam2world), torch.arange(height).to(cam2world)
    )

    # Calculate positions on the image plane in camera space.
    x = (ii - width * 0.5) * pixel_size
    y = -(jj - height * 0.5) * pixel_size
    z = torch.zeros_like(
        x
    )  # Orthogonal rays are parallel to the Z-axis in camera space.

    # Compute directions in camera space (all rays point directly along -Z).
    directions = torch.stack([x, y, z], dim=-1)

    # Transform ray origins to world coordinates.
    ray_o = (
        torch.sum(directions[..., None, :] * cam2world[:3, :3], dim=-1)
        + cam2world[:3, -1]
    )

    # Ray directions are constant for orthogonal rays and equal to the -Z axis of the world space.
    ray_d = -cam2world[:3, 2].expand_as(ray_o)

    return ray_o, ray_d


# %%
def get_presp_rays(
    width: int,
    height: int,
    focal_length: float,
    cam2world: torch.Tensor,
):
    ii, jj = mesh_grid_xy(
        torch.arange(width).to(cam2world), torch.arange(height).to(cam2world)
    )
    directions = torch.stack(
        [
            (ii - width * 0.5) / focal_length,
            -(jj - height * 0.5) / focal_length,
            -torch.ones_like(ii),
        ],
        dim=-1,
    )
    ray_d = torch.sum(directions[..., None, :] * cam2world[:3, :3], dim=-1)
    ray_o = cam2world[:3, -1].expand(ray_d.shape)
    return ray_o, ray_d


# %%
def batchify(inputs: torch.Tensor, chunk_size: int = 1024 * 8):
    """
    Casts the input tensor to a list of chunk_size-ed tensor
    Note : Last chunk (tensor) may have length < chunk_size
    """
    return [inputs[i : i + chunk_size] for i in range(0, inputs.shape[0], chunk_size)]


# %%
def load_npz(fp, nerf_dtype=torch.float32):
    data = np.load(fp)
    imgs = torch.tensor(data["images"], dtype=nerf_dtype)
    poses = torch.tensor(data["poses"], dtype=nerf_dtype)
    focal = torch.tensor(data["focal"], dtype=nerf_dtype)
    return imgs, poses, focal


# %% [markdown]
# Plotting rays
# %%
def plot_rays(ray_o: torch.Tensor, ray_d: torch.Tensor, scan_lim: float = 1.0):
    ax = plt.figure().add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.quiver(
        ray_o[..., 0],
        ray_o[..., 1],
        ray_o[..., 2],
        ray_d[..., 0],
        ray_d[..., 1],
        ray_d[..., 2],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-scan_lim, scan_lim)
    ax.set_ylim(-scan_lim, scan_lim)
    ax.set_zlim(-scan_lim, scan_lim)

    plt.show()


# %%


def get_plot_file_name(tag="", prefix="./figs/dumps/", suffix=".png"):
    # Get the current time in ISO 8601 format
    iso_time = datetime.datetime.now().isoformat()

    # Generate a random alphanumeric hash of 4 to 6 characters
    hash_length = random.randint(4, 6)
    random_hash = "".join(
        random.choices(string.ascii_letters + string.digits, k=hash_length)
    )

    # Combine the ISO time and hash with a space in between
    return f"{prefix}{iso_time}_{random_hash}#{tag}{suffix}"


# %% [markdown]
# Plot the rays for perspective and orthographic cameras

# %%
if __name__ == "__main__":
    focal_length = 12
    rot_angle = np.array([0.0, -np.pi / 2, 0.0])
    rot_mat = vec3_to_rot_mat(rot_angle)
    dims = 8

    ray_o, ray_d = get_presp_rays(dims, dims, focal_length, torch.tensor(rot_mat))
    plot_rays(ray_o, ray_d, 1.0)

    ray_o, ray_d = get_ortho_rays(dims, dims, 1 / focal_length, torch.tensor(rot_mat))
    plot_rays(ray_o, ray_d, 1.0)


# %%
# Code from the NeRF load_blender.py file

trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()
