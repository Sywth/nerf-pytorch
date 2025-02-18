import ast
import os
import imageio
from matplotlib import pyplot as plt
import torch
import numpy as np
from tqdm import trange
import load_blender

from pathlib import Path
from run_nerf import create_nerf, render, get_rays_ortho, device
from run_nerf_helpers import to8b
import run_nerf
import ct_scan
import utils

# %%
class ArgsNamespace:
    """Custom namespace to mimic argparse.Namespace behavior."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        """Returns None for missing attributes."""
        return None

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __repr__(self):
        return f"ArgsNamespace({self.__dict__})"

# %%
def parse_args(args_path):
    """Parses an args.txt file into an ArgsNamespace object."""
    args_dict = {}

    with open(args_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or "=" not in line:  # Skip empty lines
            continue

        key, value = map(str.strip, line.split("=", 1))

        # Convert values to appropriate types
        if value.lower() == "none":
            args_dict[key] = None
        elif value.lower() in {"true", "false"}:
            args_dict[key] = value.lower() == "true"
        else:
            try:
                args_dict[key] = ast.literal_eval(value)  # Convert numbers, lists, etc.
            except (ValueError, SyntaxError):
                args_dict[key] = value  # Keep as string if conversion fails

    return ArgsNamespace(**args_dict)

# %%

def get_image_at_pose(camera_pose: torch.Tensor, H: int, W: int, render_kwargs: dict):
    assert render_kwargs.get(
        "use_ortho", False
    ), "Only orthographic cameras are supported."

    # Generate rays for the given camera pose
    if render_kwargs.get("use_ortho", False):
        rays_o, rays_d = get_rays_ortho(H, W, camera_pose)
    else:
        raise ValueError("Perspective camera not supported yet.")

    # Prepare ray batch
    rays = torch.stack([rays_o, rays_d], 0)  # Shape: (2, H, W, 3)
    rays = rays.reshape(2, -1, 3)  # Flatten rays for batch processing

    # TODO : Figure out how to handle device placement cause idek what the actual f the original nerf script does
    rays = rays.to(device)
    camera_pose = camera_pose.to(device)

    # Perform rendering
    with torch.no_grad():
        rgb_map, _, _, _ = render(
            H,
            W,
            K=None,
            chunk=1024 * 32,
            rays=rays,
            **render_kwargs,
        )

    # Reshape the output image
    rgb_image = rgb_map.cpu().numpy().reshape(H, W, 3)

    return rgb_image

# %%

def plot_iamge_at_theta(theta, hwf_rendering, render_kwargs_test):
    camera_pose = load_blender.pose_spherical(
        theta=theta,
        phi=0.0,
        radius=2.0,
    )

    rgb = get_image_at_pose(
        camera_pose,
        hwf_rendering[0],
        hwf_rendering[1],
        render_kwargs_test,
    )

    plt.imshow(rgb)
    plt.show()

# %%

def plot_nerf_path(args, hwf_rendering, render_kwargs_test, intervals=10):
    poses = [
        load_blender.pose_spherical(theta=theta, phi=0.0, radius=4.0)
        for theta in np.linspace(0, 360, intervals, endpoint=False)
    ]
    poses = torch.stack(poses, 0)

    with torch.no_grad():
        rgbs, disps = run_nerf.render_path(
            poses,
            hwf_rendering,
            None,
            args.chunk,
            args.use_ortho,
            render_kwargs_test,
        )

    moviebase = os.path.join(
        args.basedir, args.expname, f"{args.expname}_spin_{hwf_rendering}"
    )

    imageio.mimwrite(
        moviebase + "rgb.mp4",
        to8b(rgbs),
        fps=args.fps,
        quality=8,
    )

# %%

def get_image_at_theta(theta_rad, hwf_rendering, render_kwargs_test, radius=2.0):
    camera_pose = load_blender.pose_spherical(
        theta=np.rad2deg(theta_rad),
        phi=0.0,
        radius=radius,
    )

    rgb = get_image_at_pose(
        camera_pose,
        hwf_rendering[0],  # H
        hwf_rendering[1],  # W
        render_kwargs_test,
    )
    return rgb

# %%
def interpolate_with_nerf(ct_imgs, angles_rad, hwf_rendering, render_kwargs_test):
    N = len(angles_rad)
    new_N = 2 * N  
    interpolated_imgs = np.zeros((new_N, *ct_imgs.shape[1:]), dtype=ct_imgs.dtype)
    interpolated_angles = np.zeros(new_N, dtype=angles_rad.dtype)
    
    interpolated_imgs[::2] = ct_imgs
    interpolated_angles[::2] = angles_rad
    
    # lin interp angles from i to i+1 so we get 
    #   [theta_{(i + i+1) / 2} for i in angles]
    interp_angles = utils.lerp(angles_rad[:-1], angles_rad[1:], 0.5)
    # lin interp angle between last and first
    interp_angles = np.concatenate((interp_angles, [utils.lerp(angles_rad[-1], angles_rad[0], 0.5)]))

    interp_imgs = np.zeros((len(interp_angles), *ct_imgs.shape[1:]))
    print(f"Interpolating {len(interp_angles)} images...")
    for i in trange(len(interp_angles)):
        theta = interp_angles[i]
        img = get_image_at_theta(theta, hwf_rendering, render_kwargs_test)
        interp_imgs[i] = utils.rgb_to_mono(img)

    interpolated_imgs[1::2] = np.stack(interp_imgs, axis=0)
    interpolated_angles[1::2] = interp_angles
    
    return interpolated_imgs, interpolated_angles

# DEBUG : Overwrite the function for testing 
def interpolate_with_nerf(ct_imgs, angles_rad, hwf_rendering, render_kwargs_test):
    imgs = np.zeros((len(angles_rad), *ct_imgs.shape[1:]))
    for i in trange(len(angles_rad)):
        theta = angles_rad[i]
        img = get_image_at_theta(theta, hwf_rendering, render_kwargs_test)
        imgs[i] = utils.rgb_to_mono(img)

    return imgs, angles_rad

# %%
if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    base_path = Path("./logs/ct_data_13_320_64_741")

    # Load trained NeRF model
    args = parse_args(base_path / "args.txt")
    args.use_ortho = True
    args.chunk = 1024 * 8
    args.white_bkgd = False
    
    near = 2.0
    far = 6.0
    bounds = {
        "near": 2.0,
        "far": 6.0,
    }

    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)
    render_kwargs_test.update(bounds)
    render_kwargs_train.update(bounds)

    model = render_kwargs_train["network_fn"].to(device)
    model.eval()

    render_gifs = True

    # Plot and comaprins with NeRF using ASTRA
    phantom_idx = 13
    size = 128
    num_scans = 15
    recon_iters = 48
    use_rgb = True
    hwf_rendering = (size, size, None)  # h, w, f

    phantom = ct_scan.load_phantom(phantom_idx, size)
    scan_params = ct_scan.AstraScanParameters(
        phantom_shape=phantom.shape,
        num_scans=num_scans,
    )

    # TODO : This flawed as the actuall training data is stored in data/nerf_synthetic/...
    #   So in future use that instead
    ct_imgs = scan_params.generate_ct_imgs(phantom)
    ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())
    angles_rad = scan_params.get_angles_rad()

    sinogram = ct_imgs.swapaxes(0, 1)
    ct_recon = scan_params.reconstruct_3d_volume_alg(sinogram, recon_iters)

    ct_imgs_interp, angles_interp = interpolate_with_nerf(
        ct_imgs, angles_rad, hwf_rendering, render_kwargs_test
    )
    sinogram_interp = ct_imgs_interp.swapaxes(0, 1)

    scan_params.reset_angles(angles_interp)
    ct_recon_interp = scan_params.reconstruct_3d_volume_alg(
        sinogram_interp, recon_iters
    )

    # Prep for plotting
    idx_slice = (size - 1) // 2
    idx_angle = (len(angles_rad) - 1) // 2
    novel_angle = utils.lerp(angles_rad[idx_angle], angles_rad[idx_angle + 1], 0.5)
    novel_view = get_image_at_theta(novel_angle, hwf_rendering, render_kwargs_test)
    novel_view = utils.rgb_to_mono(novel_view)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(8, 8))  

    # GT CT Scan View
    axes[0].imshow(ct_imgs[idx_angle])  # Note ideally we show the same angle as NeRF
    axes[0].set_title(f"CT View at {angles_rad[idx_angle]:.2f} Rad")

    # Novel view from NeRF
    axes[1].imshow(novel_view)
    axes[1].set_title(f"NeRF View at {novel_angle:.2f} Rad")

    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))  

    # Reconstruction from interpolated sinogram
    axes[0].imshow(phantom[idx_slice])
    axes[0].set_title("GT Slice")

    # Reconstruction from GT sinogram
    axes[1].imshow(ct_recon[idx_slice])
    psnr = utils.psnr(ct_recon, phantom)
    axes[1].set_title(f"SIRT Reconstruction from CT scans")

    # Reconstruction from interpolated sinogram
    axes[2].imshow(ct_recon_interp[idx_slice])
    axes[2].set_title("SIRT Reconstruction from Interpolated Scans")


    plt.tight_layout()
    plt.show()

    if render_gifs:
        print("Creating GIFs...")
        fps = 8
        suffix = f"test-{phantom_idx}_{size}"
        utils.create_gif(
            ct_imgs_interp,
            angles_interp.round(3),
            "Scan at angle {}",
            f"./figs/temp/ct_nerf_interp_{suffix}.gif",
            fps=fps,
        )

        utils.create_gif(
            ct_imgs,
            angles_rad.round(3),
            "Scan at angle {}",
            f"./figs/temp/ct_gt_{suffix}.gif",
            fps=fps,
        )

        utils.create_gif(
            ct_recon_interp,
            np.arange(ct_recon_interp.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/ct_recon_interp_{suffix}.gif",
            fps=fps,
        )

        utils.create_gif(
            ct_recon,
            np.arange(ct_recon.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/ct_recon_gt_{suffix}.gif",
            fps=fps,
        )

# %%
