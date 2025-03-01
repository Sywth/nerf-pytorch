import ast
import os
from pathlib import Path
import random
from typing import Literal

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange

import load_blender
from run_nerf import config_parser, create_nerf, render, get_rays_ortho, device, train
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

# %% [markdown]
# Scaling the CT images
# %%
MethodType = Literal["standardize", "normalize"]
def scale_astra_ct_outputs(ct_imgs : np.ndarray, method : MethodType = "normalize"):
    # NOTE : I think astra gives imgs in Hounsfield units
    if method == "standardize":
        ct_imgs  = (ct_imgs - ct_imgs.mean()) / ct_imgs.std()
    if method == "normalize":
        ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())

    return ct_imgs

def scale_nerf_ct_outputs(ct_imgs : np.ndarray, method : MethodType = "normalize"):
    if method == "standardize":
        ct_imgs  = (ct_imgs - ct_imgs.mean()) / ct_imgs.std()
    if method == "normalize":
        ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())

    return ct_imgs

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
    camera_pose = load_blender.pose_spherical_deg(
        theta=np.rad2deg(theta_rad), phi=0.0, radius=radius
    )
    rgb = get_image_at_pose(camera_pose, hwf[0], hwf[1], render_kwargs)
    return rgb


# %%


def plot_image_at_theta(theta: float, hwf: tuple, render_kwargs: dict):
    """Plot a rendered image at a given theta (degrees)."""
    camera_pose = load_blender.pose_spherical_deg(theta=theta, phi=0.0, radius=2.0)
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

    out_angles = np.linspace(min_rad, max_rad, new_N, endpoint=False)

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

    out_angles = np.linspace(min_rad, max_rad, new_N, endpoint=False)

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


def nerf_ct_imgs(ct_imgs_even, full_poses, hwf, render_kwargs, full_nerf=False):
    """
    (N, H, W) -> (N * 2, H, W)
    full_euler_poses need to be in a manner where full_euler_poses[::2] corresponds to the given ct_imgs_even
    """
    # Cast euler poses to torch tensor
    full_poses = torch.Tensor(full_poses)

    n = ct_imgs_even.shape[0]
    ct_imgs_full = np.zeros((n * 2, *ct_imgs_even.shape[1:]), dtype=ct_imgs_even.dtype)

    print(f"Interpolating {n} images with NeRF...")
    for i in trange(n):
        # we interpolate the odd indexes 
        idx = 1 + (2 * i)
        img = get_image_at_pose(full_poses[idx], hwf[0], hwf[1], render_kwargs)
        img = utils.rgb_to_mono(img)
        ct_imgs_full[idx] = img

    # NOTE : There is a somewhat fundamental flaw in that 
    #   we scale to full range based on the interpolated images alone without consdering 
    #   the full context that the astra images should be in. i.e. this will have systematic bias
    #   However, the NeRF images allways seem to cap at 0.96 when the astra ones will by defintion cap a 1.0
    ct_imgs_even = scale_nerf_ct_outputs(ct_imgs_even)
    ct_imgs_full[::2] = ct_imgs_even

    return ct_imgs_full



def nerf_ct_imgs_full(
    ct_imgs_even, full_poses, hwf, render_kwargs, full_nerf=False
):
    full_poses = torch.Tensor(full_poses)

    n = ct_imgs_even.shape[0]
    ct_imgs_full = np.zeros((n * 2, *ct_imgs_even.shape[1:]), dtype=ct_imgs_even.dtype)
    print(f"Interpolating {n} images with NeRF...")
    for i in trange(n * 2):
        img = get_image_at_pose(full_poses[i], hwf[0], hwf[1], render_kwargs)
        img = utils.rgb_to_mono(img)
        ct_imgs_full[i] = img

    return ct_imgs_full

# %%
def visualize_camera_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter_size = 25    
    # Plot the origin
    ax.scatter(0, 0, 0, color='k', s=scatter_size, label='Origin', marker='1')

    local_forawrd = np.array([0, 0, 1, 1])
    local_pos = np.array([0, 0, 0, 1])
    for pose in poses:
        cam_pos = (pose @ local_pos)[:3]
        ray_dir = -(pose @ local_forawrd)[:3]

        ax.scatter(*cam_pos, color='b', s=scatter_size, marker='x')
        ax.quiver(*cam_pos, *(ray_dir), length=1.0, color='r', normalize=True)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    size = 2.5
    ax.set_xlim([-size, size])
    ax.set_ylim([-size, size])
    ax.set_zlim([-1, 1])

    ax.set_title('Camera Poses')
    ax.legend()
    plt.show()

# %% [markdown]
# Automated training of the NeRF dataset generation and training  
# %%
# config_path="configs/ct_data_13_320_64_741.txt"
def train_nerf(config_path: str, video_ckpt : int = 250, weights_ckpt : int = 250, n_iters : int = 1000):
    parser = config_parser()
    args = parser.parse_args(args=[
        "--config", config_path,
        "--dataset_type", "blender",
        "--i_video", str(video_ckpt),
        "--i_weights", str(weights_ckpt),
        "--n_iters", str(n_iters),
        "--fps", "8",
        "--chunk", "32768",
        "--use_ortho"
    ])
    train(args)

def create_nerf_ds(phantom_idx, phantom, scaled_ct_imgs, ct_poses, ct_angles):
    fov_deg = float("inf")

    scaled_ct_imgs = utils.mono_to_rgb(scaled_ct_imgs)
    scaled_ct_imgs = ct_scan.rgb_to_rgba(scaled_ct_imgs)
    scaled_ct_imgs = ct_scan.remove_bg(scaled_ct_imgs)

    npz_dict = ct_scan.get_npz_dict(scaled_ct_imgs, ct_poses, ct_angles, phantom, fov_deg)
    return ct_scan.export_npz(npz_dict, phantom_idx)


def create_config(nerf_title, config_template_path="./configs/auto/ct_data_template.txt") -> Path:
    with open(config_template_path, 'r') as template_file:
        content = template_file.read()

    modified_content = content.replace("{}", nerf_title)

    directory = os.path.dirname(config_template_path)
    output_filename = f"{nerf_title}.txt"
    output_path = os.path.join(directory, output_filename)

    with open(output_path, 'w') as output_file:
        output_file.write(modified_content)

    return Path(output_path)


# %% [markdown]
# ## Known Issues
# TODO 
#   - Fix the fact CT images are different range (0 - 1) than the scan (0 - 120) !!!
#       - Evident by the colorbar plot in cell bellow 
#   - Fix SSIM Computation in plotting 
#   - Figure out why NeRF is worse the LERP (I think its because the range is not being normalized properly)
#       - i.e. the range of the astra scans are being normalized but not in the sameway the MLP does it.

#   - Once you have PSNR for CT-NeRF trained on 64 better than recon from 32 then move on training on 32 
#   - Add novel optimization 
#   - Do your write up 
#   - Test on opaques 
#   - Test on low to high image counts
#   - Do in-fill tests 
#   - Do novel addition thought up on train on friday 

# %%
if __name__ == "__main__":
    # RNG Seed for reproducibility
    global_rng_seed = random.randint(0, 1000)
    global_rng_seed = 42
    print(f"Using seed {global_rng_seed}")
    random.seed(global_rng_seed)
    np.random.seed(global_rng_seed)

    # GPU
    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Phatom params 
    phantom_idx = 16
    ph_size = 256
    num_scans = 32
    assert num_scans % 2 == 0, "Number of scans must be even"

    # scan parameters
    radius = 2.0
    img_res = 256
    hwf = (img_res, img_res, None)

    phantom = ct_scan.load_phantom(phantom_idx, ph_size)
    spherical_angles = np.array([
        [theta, 0.0] for theta in np.linspace(0, np.pi, num_scans, endpoint=False)
    ])
    poses = torch.stack([
        load_blender.pose_spherical_deg(np.rad2deg(theta), np.rad2deg(phi), radius=radius)
        for theta,phi in spherical_angles
    ])

    scan_2n = ct_scan.AstraScanVec3D(
        phantom.shape, spherical_angles, img_res=img_res
    )
    scan_n = ct_scan.AstraScanVec3D(
        phantom.shape, spherical_angles[::2], img_res=img_res
    )
    ct_imgs = scan_2n.generate_ct_imgs(phantom)
    ct_imgs = scale_astra_ct_outputs(ct_imgs)


    # Train the model
    # TRAIN = True
    TrainType = Literal["train new", "train existing", "load existing"]
    train_type : TrainType = "load existing"
    model_name : None | str = "ct_data_16_256_32_754"

    if train_type == "train new":
        path_ds = create_nerf_ds(phantom_idx, phantom, ct_imgs, poses, spherical_angles)
        path_cfg = create_config(path_ds.name)
        train_nerf(
            config_path=str(path_cfg),
            video_ckpt=2000,
            weights_ckpt=1000,
            n_iters=10000,
        )
        model_path = f"./logs/auto/{path_ds.name}"
    
    if train_type == "train existing":
        path_cfg = f"./configs/auto/{model_name}.txt"
        train_nerf(
            config_path=path_cfg,
            video_ckpt=2000,
            weights_ckpt=1000,
            n_iters=10000,
        )
        model_path = f"./logs/auto/{model_name}"

    if train_type == "load existing":
        model_path = f'./logs/auto/{model_name}'

    # Load the model
    model, render_kwargs_test = get_model(model_path)


    # Every 2nd image
    ct_imgs_even = ct_imgs[::2]
    ct_imgs_lanczos = ct_scan.lanczos_ct_imgs(ct_imgs_even)
    ct_imgs_lerp = ct_scan.lerp_ct_imgs(ct_imgs_even)
    ct_imgs_nerf = nerf_ct_imgs(ct_imgs_even, poses, hwf, render_kwargs_test)

    num_dp = 5
    bp_cg_1, bp_si_1 = ct_scan.plot_reconstructions(scan_2n, ct_imgs, phantom, ph_size, title=f"[Full Orignial] Reconstructed Slice ({num_scans} views)", num_dp=num_dp)
    bp_cg_2, bp_si_2 = ct_scan.plot_reconstructions(scan_n, ct_imgs_even, phantom, ph_size, title=f"[Half Orignial] Reconstructed Slice ({num_scans} views)" , num_dp=num_dp)
    
    bp_cg_3, bp_si_3 = ct_scan.plot_reconstructions(scan_2n, ct_imgs_lerp, phantom, ph_size, title=f"[Half Orignial, Half Lerp] Reconstructed Slice ({num_scans} views)", num_dp=num_dp)
    bp_cg_4, bp_si_4 = ct_scan.plot_reconstructions(scan_2n, ct_imgs_lanczos, phantom, ph_size, title=f"[Half Orignial, Half lanczos] Reconstructed Slice ({num_scans} views)", num_dp=num_dp)
    bg_cg_5, bp_si_5 = ct_scan.plot_reconstructions(scan_2n, ct_imgs_nerf, phantom, ph_size, title=f"[Half Orignial, Half Nerf] Reconstructed Slice ({num_scans} views)", num_dp=num_dp)

    # Create GIFs of one interpolation method at a tiem 
    RENDER_GIF = True
    if RENDER_GIF:
        ct_imgs_interp = ct_imgs_nerf
        method = "nerf"

        suffix = f"[{num_scans}]_cs2_"
        utils.create_gif(
            ct_imgs,
            np.round(spherical_angles, 3),
            "Scan at angle {}",
            f"./figs/temp/{suffix}ct_scan.gif",
            fps=8,
        )

        utils.create_gif(
            ct_imgs_interp,
            np.round(spherical_angles, 3),
            "Scan at angle {}",
            f"./figs/temp/{suffix}ct_scan_{method}.gif",
            fps=8,
        )

# %%
def compare(img_0, img_1, img_2 = None, img_3 = None):
    n = 2 + int(img_2 is not None) + int(img_3 is not None)
    fig, axes = plt.subplots(1, n, figsize=(16 + (n * 2),8))

    im0 = axes[0].imshow(img_0)
    axes[0].set_title("Original")

    im1 = axes[1].imshow(img_1)
    axes[1].set_title("NeRF")

    if img_2 is not None:
        im2 = axes[2].imshow(img_2)
        axes[2].set_title("LeRP")

    if img_3 is not None:
        im1 = axes[3].imshow(img_3)
        axes[3].set_title("Lanczos")

    plt.tight_layout()
    plt.show()

# %%
DEBUG = False
if DEBUG:
    for i,(intp, gt) in enumerate(zip(ct_imgs_nerf, ct_imgs)):
        compare(gt, intp)