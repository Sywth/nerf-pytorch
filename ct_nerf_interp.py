import ast
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

import load_blender
from run_nerf import create_nerf, render, get_rays_ortho, device
from run_nerf_helpers import to8b
import ct_scan
import utils


# %%
class ArgsNamespace:
    """A minimal replacement for argparse.Namespace."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return None

    def __repr__(self):
        return f"ArgsNamespace({self.__dict__})"


# %%
def parse_args(args_path: Path) -> ArgsNamespace:
    """Parse an args.txt file into an ArgsNamespace object."""
    args_dict = {}
    with open(args_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = map(str.strip, line.split("=", 1))
            if value.lower() == "none":
                args_dict[key] = None
            elif value.lower() in {"true", "false"}:
                args_dict[key] = value.lower() == "true"
            else:
                try:
                    args_dict[key] = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    args_dict[key] = value
    return ArgsNamespace(**args_dict)


# %%


def get_image_at_pose(
    camera_pose: torch.Tensor, H: int, W: int, render_kwargs: dict
) -> np.ndarray:
    """Render an image given a camera pose using orthographic rays."""
    if not render_kwargs.get("use_ortho", False):
        raise ValueError("Only orthographic cameras are supported.")
    # Generate orthographic rays.
    rays_o, rays_d = get_rays_ortho(H, W, camera_pose)
    rays = torch.stack([rays_o, rays_d], 0).reshape(2, -1, 3).to(device)

    with torch.no_grad():
        rgb_map, _, _, _ = render(
            H, W, K=None, chunk=1024 * 32, rays=rays, **render_kwargs
        )
    rgb_image = rgb_map.cpu().numpy().reshape(H, W, 3)
    return rgb_image


# %%


def get_image_at_theta(
    theta_rad: float, hwf: tuple, render_kwargs: dict, radius: float = 2.0
) -> np.ndarray:
    """Render an image at a given theta (in radians)."""
    camera_pose = load_blender.pose_spherical(
        theta=np.rad2deg(theta_rad), phi=0.0, radius=radius
    )
    rgb = get_image_at_pose(camera_pose, hwf[0], hwf[1], render_kwargs)
    return rgb


# %%


def plot_image_at_theta(theta: float, hwf: tuple, render_kwargs: dict):
    """Plot a rendered image at a given theta (degrees)."""
    camera_pose = load_blender.pose_spherical(theta=theta, phi=0.0, radius=2.0)
    rgb = get_image_at_pose(camera_pose, hwf[0], hwf[1], render_kwargs)
    plt.imshow(rgb)
    plt.title(f"Image at {theta:.2f}°")
    plt.show()


# %%


def interpolate_with_nerf(
    given_ct_imgs: np.ndarray,
    min_rad: float,
    max_rad: float,
    hwf: tuple[int, int, None],
    render_kwargs: dict,
):
    N = len(given_ct_imgs)
    new_N = 2 * N

    out_angles = np.linspace(
        min_rad, max_rad, new_N, endpoint=False
    )

    out_imgs = np.empty((new_N, *given_ct_imgs.shape[1:]), dtype=given_ct_imgs.dtype)
    out_imgs[::2] = given_ct_imgs

    print(f"Interpolating {N} images with NeRF...")
    for i in trange(1, new_N, 2):
        # for i in trange(new_N - 1, 0, -2):
        img = get_image_at_theta(out_angles[i], hwf, render_kwargs)
        img = utils.rgb_to_mono(img)
        out_imgs[i] = img

    return out_imgs, out_angles


def interpolate_with_nerf(
    given_ct_imgs: np.ndarray,
    min_rad: float,
    max_rad: float,
    hwf: tuple[int, int, None],
    render_kwargs: dict,
):
    N = len(given_ct_imgs)
    new_N = N 

    out_angles = np.linspace(
        min_rad, max_rad, new_N, endpoint=False
    )

    out_imgs = np.empty((new_N, *given_ct_imgs.shape[1:]), dtype=given_ct_imgs.dtype)

    print(f"Interpolating {N} images with NeRF...")
    for i in trange(new_N):
        img = get_image_at_theta(out_angles[i], hwf, render_kwargs)
        img = utils.rgb_to_mono(img)
        out_imgs[i] = img

    return out_imgs, out_angles


# %%
def get_model(path: Path | str):
    base_path = Path(path)
    args = parse_args(base_path / "args.txt")
    args.use_ortho = True
    args.chunk = 1024 * 8
    args.white_bkgd = False

    # Define rendering bounds.
    bounds = {"near": 2.0, "far": 6.0}
    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)
    render_kwargs_train.update(bounds)
    render_kwargs_test.update(bounds)

    # Load and set model to evaluation.
    model = render_kwargs_train["network_fn"].to(device)
    model.eval()
    return model, render_kwargs_test


# %%
if __name__ == "__main__":
    # Device setup.
    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    model, render_kwargs_test = get_model("./logs/ct_data_13_320_64_741")

    # CT phantom and scan parameters.
    phantom_idx = 13
    size = 128
    num_scans = 16  # Control: use 16 scans.
    recon_iters = 48
    hwf = (size, size, None)

    min_theta_rad = 0.0
    max_theta_rad = 2 * np.pi

    phantom = ct_scan.load_phantom(phantom_idx, size)
    scan_params = ct_scan.AstraScanParameters(
        phantom_shape=phantom.shape,
        num_scans=num_scans,
        min_theta=min_theta_rad,
        max_theta=max_theta_rad,
    )

    # Generate original CT sinogram and normalized images.
    ct_imgs = scan_params.generate_ct_imgs(phantom)
    ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())
    angles_rad = scan_params.get_angles_rad()

    # Baseline CT reconstruction from original sinogram.
    sinogram = ct_imgs.swapaxes(0, 1)
    ct_recon = scan_params.reconstruct_3d_volume_alg(sinogram, recon_iters)

    # NeRF interpolation: generate intermediate images.
    ct_imgs_interp, angles_interp = interpolate_with_nerf(
        ct_imgs, min_theta_rad, max_theta_rad, hwf, render_kwargs_test
    )
    sinogram_interp = ct_imgs_interp.swapaxes(0, 1)
    # Prepare scan for interpolated sinogram.
    scan_params.num_scans = len(angles_interp)
    scan_params.re_init()
    ct_recon_interp = scan_params.reconstruct_3d_volume_alg(
        sinogram_interp, recon_iters
    )

    # Visualization: compare a representative CT slice and NeRF novel view.
    idx_slice = (size - 1) // 2
    mid_idx = (len(angles_rad) - 1) // 2
    novel_angle = utils.lerp(angles_rad[mid_idx], angles_rad[mid_idx + 1], 0.5)
    novel_view = utils.rgb_to_mono(
        get_image_at_theta(novel_angle, hwf, render_kwargs_test)
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    axes[0].imshow(ct_imgs[mid_idx])
    axes[0].set_title(f"CT View at {angles_rad[mid_idx]:.2f} Rad")
    axes[1].imshow(novel_view)
    axes[1].set_title(f"NeRF View at {novel_angle:.2f} Rad")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    axes[0].imshow(phantom[idx_slice])
    axes[0].set_title("GT Phantom Slice")
    axes[1].imshow(ct_recon[idx_slice])
    axes[1].set_title("SIRT Reconstruction (Original)")
    axes[2].imshow(ct_recon_interp[idx_slice])
    axes[2].set_title("SIRT Reconstruction (NeRF Interpolated)")
    plt.tight_layout()
    plt.show()

    # Compute PSNR for quantitative comparison.
    psnr_value = utils.psnr(ct_recon_interp, phantom)
    print(f"PSNR (Reconstructed vs. Phantom): {psnr_value:.2f} dB")

    # Optionally, create GIFs of the scans and reconstructions.
    render_gifs = False
    if render_gifs:
        print("Creating GIFs...")
        suffix = f"{phantom_idx}_{size}"
        prefix = "only_"
        utils.create_gif(
            ct_imgs_interp,
            np.round(angles_interp, 3),
            "Scan at angle {}",
            f"./figs/temp/{prefix}ct_nerf_interp_{suffix}.gif",
            fps=16,
        )
        utils.create_gif(
            ct_imgs,
            np.round(angles_rad, 3),
            "Scan at angle {}",
            f"./figs/temp/{prefix}ct_gt_{suffix}.gif",
            fps=8,
        )
        utils.create_gif(
            ct_recon_interp,
            np.arange(ct_recon_interp.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/{prefix}ct_recon_interp_{suffix}.gif",
            fps=12,
        )
        utils.create_gif(
            ct_recon,
            np.arange(ct_recon.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/{prefix}ct_recon_gt_{suffix}.gif",
            fps=12,
        )

    pass

# %%
