import torch
import matplotlib.pyplot as plt
import numpy as np
import run_nerf_helpers


def mesh_grid_xy(t1, t2):
    ii, jj = torch.meshgrid(t1, t2, indexing="ij")
    return ii.transpose(-1, -2), jj.transpose(-1, -2)


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


def plot_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, axes_lim: float = 1.0):
    rays_o = rays_o.cpu().numpy()
    rays_d = rays_d.cpu().numpy()

    # normalize ray_ds
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # first move each origin along its direction by a small epsilon
    epsilon = 0.0
    rays_o = rays_o + epsilon * rays_d

    ax = plt.figure().add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.quiver(
        rays_o[..., 0],
        rays_o[..., 1],
        rays_o[..., 2],
        rays_d[..., 0],
        rays_d[..., 1],
        rays_d[..., 2],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-axes_lim, axes_lim)
    ax.set_ylim(-axes_lim, axes_lim)
    ax.set_zlim(-axes_lim, axes_lim)

    plt.show()


def make_plot_of_rays(K: np.ndarray, pose: np.ndarray, use_ortho=False):
    K = K.copy()
    pose = pose.copy()

    fovx = np.deg2rad(45)
    fovy = fovx
    H, W = 5, 5
    fx = W / (2 * np.tan(fovx / 2))
    fy = H / (2 * np.tan(fovy / 2))

    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = W // 2
    K[1, 2] = H // 2

    num_rays = 5

    rays_o, rays_d = (
        run_nerf_helpers.get_rays_ortho(num_rays, num_rays, torch.Tensor(pose))
        if use_ortho
        else run_nerf_helpers.get_rays(num_rays, num_rays, K, torch.Tensor(pose))
    )

    plot_rays(rays_o, rays_d, axes_lim=5.0)
