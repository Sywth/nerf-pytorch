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


def interpolate_with_nerf(ct_imgs, angles_rad, hwf_rendering, render_kwargs_test):
    interpolated_shape = (len(angles_rad) * 2, *ct_imgs.shape[1:])
    interpolated_imgs = np.zeros(interpolated_shape)
    angles = np.zeros(len(angles_rad) * 2)

    print("Interpolating views with NeRF...")
    for i in trange(len(angles_rad) - 1):
        theta1, theta2 = angles_rad[i], angles_rad[i + 1]
        theta_interp = utils.lerp(theta1, theta2, 0.5)
        angles[i * 2] = theta1
        angles[i * 2 + 1] = theta_interp

        # Get interpolated view from NeRF
        img_interp = get_image_at_theta(theta_interp, hwf_rendering, render_kwargs_test)
        img_interp = utils.rgb_to_mono(img_interp)
        interpolated_imgs[i * 2] = ct_imgs[i]  # Original
        interpolated_imgs[i * 2 + 1] = img_interp

    # Last view, interpolate between index 0 amd -1
    theta_interp = utils.lerp(angles_rad[-1], angles_rad[0], 0.5)
    angles[-1] = theta_interp

    img_interp = get_image_at_theta(theta_interp, hwf_rendering, render_kwargs_test)
    interpolated_imgs[-1] = utils.rgb_to_mono(img_interp)

    return interpolated_imgs, angles


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    base_path = Path("./logs/ct_data_13_320_64_741")

    # Load trained NeRF model
    args = parse_args(base_path / "args.txt")
    args.use_ortho = True
    args.chunk = 1024 * 8
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

    render_gifs = False

    # Plot and comaprins with NeRF using ASTRA
    phantom_idx = 13
    size = 128
    num_scans = 50
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

    # START TBD : Quick Rendering Test for NeRF
    angles = np.linspace(0, 2 * np.pi, 50)
    imgs_shape = (len(angles), 128, 128)
    imgs = np.zeros(imgs_shape)
    for i, angle in enumerate(angles):
        # TODO figure out how to get this to sync with the existing astra scan and line up nicely. 
        # TODO figure out how to get this to 
        # TODO Check if radius changes the nerf model, as it shouldn't as its orthographic
        # TODO Figure out why background is white (1.0) when it should be black (0.0)
        imgs[i] = utils.rgb_to_mono(
            get_image_at_theta(-angle, hwf_rendering, render_kwargs_test)
        )

    utils.create_gif(
        imgs,
        angles,
        "Angle {}",
        f"./figs/temp/ct_nerf_neg.gif",
    )
    # END TBD  Quick Rendering Test for NeRF

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
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  # 2 rows, 2 columns

    # GT CT Scan View
    axes[0, 0].imshow(ct_imgs[idx_angle])  # Note ideally we show the same angle as NeRF
    axes[0, 0].set_title(f"CT View at {angles_rad[idx_angle]:.2f} Rad")

    # Novel view from NeRF
    axes[0, 1].imshow(novel_view)
    axes[0, 1].set_title(f"NeRF View at {novel_angle:.2f} Rad")

    # Reconstruction from GT sinogram
    axes[1, 0].imshow(ct_recon[idx_slice])
    axes[1, 0].set_title(f"SIRT Reconstruction from GT scans")

    # Reconstruction from interpolated sinogram
    axes[1, 1].imshow(ct_recon_interp[idx_slice])
    axes[1, 1].set_title("SIRT Reconstruction from Interpolated Scans")

    plt.tight_layout()
    plt.show()

    if render_gifs:
        print("Creating GIFs...")

        suffix = f"{phantom_idx}_{size}"
        utils.create_gif(
            ct_imgs_interp,
            angles_interp,
            "Scan at angle {}",
            f"./figs/temp/ct_nerf_interp_{suffix}.gif",
        )

        utils.create_gif(
            ct_imgs,
            angles_rad,
            "Scan at angle {}",
            f"./figs/temp/ct_gt_{suffix}.gif",
        )

        utils.create_gif(
            ct_recon_interp,
            np.arange(ct_recon_interp.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/ct_recon_interp_{suffix}.gif",
        )

        utils.create_gif(
            ct_recon,
            np.arange(ct_recon.shape[0]),
            "Reconstruction at slice {}",
            f"./figs/temp/ct_recon_gt_{suffix}.gif",
        )
