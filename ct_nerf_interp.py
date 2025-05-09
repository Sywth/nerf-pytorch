import ast
import os
from pathlib import Path
import pickle
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


def scale_astra_ct_outputs(ct_imgs: np.ndarray, method: MethodType = "normalize"):
    # NOTE : I think astra gives imgs in Hounsfield units
    if method == "standardize":
        ct_imgs = (ct_imgs - ct_imgs.mean()) / ct_imgs.std()
    if method == "normalize":
        ct_imgs = (ct_imgs - ct_imgs.min()) / (ct_imgs.max() - ct_imgs.min())

    return ct_imgs


def scale_nerf_ct_outputs(ct_imgs: np.ndarray, method: MethodType = "normalize"):
    if method == "standardize":
        ct_imgs = (ct_imgs - ct_imgs.mean()) / ct_imgs.std()
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


def nerf_ct_imgs(ct_imgs_even, full_poses, hwf, render_kwargs):
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


def nerf_ct_imgs_limited(
    ct_imgs_partial, full_poses, hwf, render_kwargs, partial_indices
):
    """
    Given ct_imgs_partial : (M, H, W) and full_poses : (N, 4, 4) where M < N
    and partial_indices where the full_poses[partial_indices[i]] corresponds to ct_imgs_partial[i]
    """
    assert (
        hwf[:2] == ct_imgs_partial.shape[1:]
    ), "hwf doesnt match given ct_imgs_partial images shape"

    ct_imgs_full = np.zeros(
        (len(full_poses), hwf[0], hwf[1]), dtype=ct_imgs_partial.dtype
    )
    for i, img in zip(partial_indices, ct_imgs_partial):
        ct_imgs_full[i] = img

    print(f"Infilling {len(full_poses) - len(partial_indices)} images with NeRF...")
    for i in trange(len(full_poses)):
        if i in partial_indices:
            continue

        img = get_image_at_pose(full_poses[i], hwf[0], hwf[1], render_kwargs)
        img = utils.rgb_to_mono(img)
        ct_imgs_full[i] = img

    return ct_imgs_full


def nerf_ct_imgs_full(full_poses, hwf, render_kwargs, dtype=np.float32):
    N = len(full_poses)

    full_poses = torch.Tensor(full_poses)
    ct_imgs_full = np.zeros((N, hwf[0], hwf[1]), dtype=dtype)
    print(f"Interpolating {N} images with NeRF...")
    for i in trange(N):
        img = get_image_at_pose(full_poses[i], hwf[0], hwf[1], render_kwargs)
        img = utils.rgb_to_mono(img)
        ct_imgs_full[i] = img

    return ct_imgs_full


# %%
def visualize_camera_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter_size = 25
    # Plot the origin
    ax.scatter(0, 0, 0, color="k", s=scatter_size, label="Origin", marker="1")

    local_forawrd = np.array([0, 0, 1, 1])
    local_pos = np.array([0, 0, 0, 1])
    for pose in poses:
        cam_pos = (pose @ local_pos)[:3]
        ray_dir = -(pose @ local_forawrd)[:3]

        ax.scatter(*cam_pos, color="b", s=scatter_size, marker="x")
        ax.quiver(*cam_pos, *(ray_dir), length=1.0, color="r", normalize=True)

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    size = 2.5
    ax.set_xlim([-size, size])
    ax.set_ylim([-size, size])
    ax.set_zlim([-1, 1])

    ax.set_title("Camera Poses")
    ax.legend()
    plt.show()


# %% [markdown]
# Automated training of the NeRF dataset generation and training
# %%
# config_path="configs/ct_data_13_320_64_741.txt"
def train_nerf(
    config_path: str,
    video_ckpt: int = 250,
    weights_ckpt: int = 250,
    n_iters: int = 1000,
    use_mono_ct: bool = True,
    use_view_dirs: bool = True,
):
    parser = config_parser()
    args = [
        "--config",
        config_path,
        "--dataset_type",
        "blender",
        "--i_video",
        str(video_ckpt),
        "--i_weights",
        str(weights_ckpt),
        "--n_iters",
        str(n_iters),
        "--fps",
        "8",
        "--chunk",
        "32768",
        "--use_ortho",
    ]
    if use_mono_ct:
        args.append("--use_mono_ct")
    if use_view_dirs:
        args.append("--use_viewdirs")

    args = parser.parse_args(args=args)
    psnrs = []
    train(args, psnrs)
    return psnrs


def create_nerf_ds(phantom_idx, phantom, scaled_ct_imgs, ct_poses, ct_angles):
    fov_deg = float("inf")

    scaled_ct_imgs = utils.mono_to_rgb(scaled_ct_imgs)
    scaled_ct_imgs = ct_scan.rgb_to_rgba(scaled_ct_imgs)
    scaled_ct_imgs = ct_scan.remove_bg(scaled_ct_imgs)

    npz_dict = ct_scan.get_npz_dict(
        scaled_ct_imgs, ct_poses, ct_angles, phantom, fov_deg
    )
    return ct_scan.export_npz(npz_dict, phantom_idx)


def create_config(
    nerf_title,
    config_template_path="./configs/auto/ct_data_template.txt",
    use_view_dirs=True,
):
    with open(config_template_path, "r") as template_file:
        content = template_file.read()

    content = content.replace("{title}", nerf_title)
    content = content.replace("{viewdirs}", str(use_view_dirs))

    directory = os.path.dirname(config_template_path)
    output_filename = f"{nerf_title}.txt"
    output_path = os.path.join(directory, output_filename)

    with open(output_path, "w") as output_file:
        output_file.write(content)

    return Path(output_path)


def seed_rngs(global_rng_seed: int = 42):
    print(f"Using seed {global_rng_seed}")
    random.seed(global_rng_seed)
    np.random.seed(global_rng_seed)


# %%
TestType = Literal["sparse scan", "limited scan"]
TrainType = Literal["train new", "train existing", "load existing"]
ScanType = Literal["parallel", "cone beam"]
DEBUG = False

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
def train_and_save_nerf(
    # Hyper params
    n_iters=10_000,
    video_ckpt=5_000,
    weights_ckpt=5_000,
    # Phantom & scan params
    phantom_idx=4,
    num_scans=32,
    ph_size=256,
    radius=2.0,
    model_name: None | str = None,  # [MODIFY]
    test_type: TestType = "limited scan",  # [MODIFY]
    train_type: TrainType = "train new",  # Fixed for now
    scan_type: ScanType = "parallel",  # Fixed for now
):

    # GPU
    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Model
    proper_name = test_type.replace(" ", "_")
    test_title_fn = f"{proper_name}_ph-{phantom_idx}_scans-{num_scans}_iters-{n_iters}"
    print(f"Test title: {test_title_fn}")
    global_rng_seed = utils.str_title_hash(test_title_fn)
    seed_rngs(global_rng_seed)

    opt_kwargs = {}
    if test_type == "limited scan":
        test_desc_note = (
            "Limited Nerf using scans at limited angles as described in pattern"
        )
        (
            spherical_angles,
            ct_imgs,
            poses,
            use_view_dirs,
            use_mono_ct,
            model_path,
            phantom,
            psnrs,
            render_kwargs_test,
            limited_spherical_angles,
            limited_poses,
            limited_indices,
            pattern,
            # Added late
            ct_imgs_pred,
            recon_pred,
        ) = limited_scan(
            train_type=train_type,
            phantom_idx=phantom_idx,
            num_scans=num_scans,
            n_iters=n_iters,
            video_ckpt=video_ckpt,
            weights_ckpt=weights_ckpt,
            radius=radius,
            ph_size=ph_size,
            img_res=ph_size,
        )

        opt_kwargs.update(
            {
                "limited_spherical_angles": limited_spherical_angles,
                "limited_poses": limited_poses,
                "limited_indices": limited_indices,
                "pattern": pattern,
            }
        )

    if test_type == "sparse scan":
        test_desc_note = "Sparse Nerf using even images as GT scan images"

        (
            spherical_angles,
            ct_imgs,
            poses,
            use_view_dirs,
            use_mono_ct,
            model_path,
            phantom,
            psnrs,
            render_kwargs_test,
            spherical_angles_train,
            poses_train,
            # Added late
            ct_imgs_pred,
            recon_pred,
        ) = sparse_scan(
            train_type=train_type,
            phantom_idx=phantom_idx,
            num_scans=num_scans,
            n_iters=n_iters,
            video_ckpt=video_ckpt,
            weights_ckpt=weights_ckpt,
            radius=radius,
            ph_size=ph_size,
            img_res=ph_size,
        )
        opt_kwargs.update(
            {
                "spherical_angles_train": spherical_angles_train,
                "poses_train": poses_train,
            }
        )

    save_data(
        test_title_fn=test_title_fn,
        phantom_idx=phantom_idx,
        num_scans=num_scans,
        n_iters=n_iters,
        test_desc_note=test_desc_note,
        ph_size=ph_size,
        test_type=test_type,
        scan_type=scan_type,
        spherical_angles=spherical_angles,
        ct_imgs=ct_imgs,
        poses=poses,
        use_view_dirs=use_view_dirs,
        use_mono_ct=use_mono_ct,
        model_path=model_path,
        phantom=phantom,
        psnrs=psnrs,
        render_kwargs_test=render_kwargs_test,
        global_rng_seed=global_rng_seed,
        # Added late
        ct_imgs_pred=ct_imgs_pred,
        recon_pred=recon_pred,
        # Optionals
        **opt_kwargs,
    )


def limited_scan(
    # Core
    train_type: TrainType = "train new",
    # Independent
    phantom_idx: int = 4,
    num_scans: int = 32,
    # Hyper params
    n_iters: int = 10_000,
    video_ckpt: int = 5_000,
    weights_ckpt: int = 5_000,
    radius: float = 2.0,
    # Expected
    ph_size: int = 256,
    img_res: int = 256,
    # Optional
    model_name: None | str = "ct_data_13_256_16_600",  # [MODIFY]
    process_model: bool = True,
):
    # Phatom params
    assert num_scans % 8 == 0, "Number of scans must be a multiple of 8"

    hwf = (img_res, img_res, None)

    # TODO - Figure what limited angles to use
    phantom = ct_scan.load_phantom(phantom_idx, ph_size)
    theta_min, theta_max = 0, np.pi
    spherical_angles = np.array(
        [  # 0 - 180 deg
            [theta, 0.0]
            for theta in np.linspace(theta_min, theta_max, num_scans, endpoint=False)
        ]
    )
    poses = torch.stack(
        [
            load_blender.pose_spherical_deg(
                np.rad2deg(theta), np.rad2deg(phi), radius=radius
            )
            for theta, phi in spherical_angles
        ]
    )

    scan_full = ct_scan.AstraScanVec3D(  # 0 - 180 deg
        phantom.shape, spherical_angles, img_res=img_res
    )

    limited_indices = []
    limited_poses = []
    limited_spherical_angles = []

    pattern = "11011000"

    def angle_limit_condition(i, n, pattern="1010"):
        m = len(pattern)
        segment_idx = (i * m) // n
        return pattern[segment_idx] == "1"

    for i, curr_pose in enumerate(poses):
        if not angle_limit_condition(i, n=len(poses), pattern=pattern):
            continue

        limited_spherical_angles.append(spherical_angles[i])
        limited_indices.append(i)
        limited_poses.append(curr_pose)

    limited_poses = torch.stack(limited_poses)

    scan_partial = ct_scan.AstraScanVec3D(  # 0 - 180 deg \ 45 - 135 deg
        phantom.shape, limited_spherical_angles, img_res=img_res
    )

    ct_imgs = scan_full.generate_ct_imgs(phantom)
    ct_imgs = scale_astra_ct_outputs(ct_imgs)
    ct_imgs_limited = np.array([ct_imgs[i] for i in limited_indices])

    # Params
    use_view_dirs = True
    use_mono_ct = False

    # Plot Angles [DEBUG]
    utils.plot_angles(
        spherical_angles,
        limited_indices,
        title="Full Scan Angles",
    )
    # DEBUG

    psnrs = None
    if train_type == "train new":
        path_ds = create_nerf_ds(
            phantom_idx, phantom, ct_imgs_limited, limited_poses, None
        )
        path_cfg = create_config(
            path_ds.name,
            use_view_dirs=use_view_dirs,
        )
        psnrs = train_nerf(
            config_path=str(path_cfg),
            video_ckpt=video_ckpt,
            weights_ckpt=weights_ckpt,
            n_iters=n_iters,
            use_mono_ct=use_mono_ct,
            use_view_dirs=use_view_dirs,
        )
        model_path = f"./logs/auto/{path_ds.name}"

    if train_type == "train existing":
        path_cfg = f"./configs/auto/{model_name}.txt"
        psnrs = train_nerf(
            config_path=path_cfg,
            video_ckpt=video_ckpt,
            weights_ckpt=weights_ckpt,
            n_iters=n_iters,
            use_mono_ct=use_mono_ct,
            use_view_dirs=use_view_dirs,
        )
        model_path = f"./logs/auto/{model_name}"

    if train_type == "load existing":
        model_path = f"./logs/auto/{model_name}"

    # Load the model
    model, render_kwargs_test = get_model(model_path)

    ct_imgs_nerf = nerf_ct_imgs_limited(
        ct_imgs_partial=ct_imgs_limited,
        full_poses=poses,
        hwf=hwf,
        render_kwargs=render_kwargs_test,
        partial_indices=limited_indices,
    )
    recon_pred = scan_full.reconstruct_3d_volume_sirt(ct_imgs_nerf)
    if process_model:

        num_dp = 5
        bp_si_1 = ct_scan.plot_reconstructions(
            scan_full,
            ct_imgs,
            phantom,
            ph_size,
            title=f"[Full Orignial] Reconstructed Slice ({num_scans} views)",
            num_dp=num_dp,
        )
        bp_si_2 = ct_scan.plot_reconstructions(
            scan_partial,
            ct_imgs_limited,
            phantom,
            ph_size,
            title=f"[Part Orignial] Reconstructed Slice ({len(limited_spherical_angles)} views)",
            num_dp=num_dp,
        )
        bp_si_5 = ct_scan.plot_reconstructions(
            scan_full,
            ct_imgs_nerf,
            phantom,
            ph_size,
            title=f"[Part Orignial, Part Nerf] Reconstructed Slice ({num_scans} views)",
            num_dp=num_dp,
        )

        if DEBUG:
            ct_imgs_full_nerf = nerf_ct_imgs_full(poses, hwf, render_kwargs_test)
            bg_cg_6, bp_si_6 = ct_scan.plot_reconstructions(
                scan_full,
                ct_imgs_full_nerf,
                phantom,
                ph_size,
                title=f"[Full Nerf] Reconstructed Slice ({num_scans} views)",
                num_dp=num_dp,
            )

        # Create GIFs of one interpolation method at a tiem
        RENDER_GIF = False
        if RENDER_GIF:
            ct_imgs_infill = ct_imgs_nerf
            method = "nerf"

            suffix = f"[{num_scans}]_limited_v3_"
            utils.create_gif(
                ct_imgs,
                np.round(spherical_angles, 3),
                "Scan at angle {}",
                f"./figs/gifs/{suffix}ct_scan.gif",
                fps=8,
            )

            utils.create_gif(
                ct_imgs_infill,
                np.round(spherical_angles, 3),
                "Scan at angle {}",
                f"./figs/gifs/{suffix}ct_scan_{method}.gif",
                fps=8,
            )

            utils.create_gif(
                ct_imgs_full_nerf,
                np.round(spherical_angles, 3),
                "Scan at angle {}",
                f"./figs/gifs/{suffix}ct_scan_full_{method}.gif",
                fps=8,
            )

    return (
        spherical_angles,
        ct_imgs,
        poses,
        use_view_dirs,
        use_mono_ct,
        model_path,
        phantom,
        psnrs,
        render_kwargs_test,
        limited_spherical_angles,
        limited_poses,
        limited_indices,
        pattern,
        # Added late
        ct_imgs_nerf,
        recon_pred,
    )


def sparse_scan(
    # Core
    train_type: TrainType = "train new",
    # Independent
    phantom_idx: int = 4,
    num_scans: int = 32,
    # Hyper params
    n_iters: int = 10_000,
    video_ckpt: int = 5_000,
    weights_ckpt: int = 5_000,
    radius: float = 2.0,
    # Expected
    ph_size: int = 256,
    img_res: int = 256,
    # Optional
    model_name: None | str = "ct_data_13_256_16_600",  # [MODIFY]
    process_model: bool = True,
):

    # Phatom params
    assert num_scans % 8 == 0, "Number of scans must be a multiple of 8"

    hwf = (img_res, img_res, None)

    phantom = ct_scan.load_phantom(phantom_idx, ph_size)

    theta_min, theta_max = 0, np.pi
    spherical_angles = np.array(
        [
            [theta, 0.0]
            for theta in np.linspace(theta_min, theta_max, num_scans, endpoint=False)
        ]
    )
    spherical_angles_train = spherical_angles[::2]
    poses = torch.stack(
        [
            load_blender.pose_spherical_deg(
                np.rad2deg(theta), np.rad2deg(phi), radius=radius
            )
            for theta, phi in spherical_angles
        ]
    )

    scan_2n = ct_scan.AstraScanVec3D(phantom.shape, spherical_angles, img_res=img_res)
    scan_n = ct_scan.AstraScanVec3D(
        phantom.shape, spherical_angles_train, img_res=img_res
    )
    ct_imgs = scan_2n.generate_ct_imgs(phantom)
    ct_imgs = scale_astra_ct_outputs(ct_imgs)

    ct_imgs_train = ct_imgs[::2]
    poses_train = poses[::2]

    # Params
    use_view_dirs = True
    use_mono_ct = False

    psnrs = None
    if train_type == "train new":
        path_ds = create_nerf_ds(
            phantom_idx, phantom, ct_imgs_train, poses_train, spherical_angles_train
        )
        path_cfg = create_config(
            path_ds.name,
            use_view_dirs=use_view_dirs,
        )
        psnrs = train_nerf(
            config_path=str(path_cfg),
            video_ckpt=video_ckpt,
            weights_ckpt=weights_ckpt,
            n_iters=n_iters,
            use_mono_ct=use_mono_ct,
            use_view_dirs=use_view_dirs,
        )
        model_path = f"./logs/auto/{path_ds.name}"

    if train_type == "train existing":
        path_cfg = f"./configs/auto/{model_name}.txt"
        psnrs = train_nerf(
            config_path=path_cfg,
            video_ckpt=video_ckpt,
            weights_ckpt=weights_ckpt,
            n_iters=n_iters,
            use_mono_ct=use_mono_ct,
            use_view_dirs=use_view_dirs,
        )
        model_path = f"./logs/auto/{model_name}"

    if train_type == "load existing":
        model_path = f"./logs/auto/{model_name}"

    # Unload pytorch
    print("> Unloading pytorch")
    torch.cuda.empty_cache()

    # Load the model
    model, render_kwargs_test = get_model(model_path)
    ct_imgs_nerf = nerf_ct_imgs(ct_imgs_train, poses, hwf, render_kwargs_test)
    recon_sirt = scan_2n.reconstruct_3d_volume_sirt(ct_imgs_nerf)

    if process_model:
        # Every 2nd image
        ct_imgs_lanczos = ct_scan.lanczos_ct_imgs(ct_imgs_train)
        ct_imgs_lerp = ct_scan.lerp_ct_imgs(ct_imgs_train)

        num_dp = 5
        bp_si_1 = ct_scan.plot_reconstructions(
            scan_2n,
            ct_imgs,
            phantom,
            ph_size,
            title=f"[Full Orignial] Reconstructed Slice ({num_scans} views)",
            num_dp=num_dp,
        )
        bp_si_2 = ct_scan.plot_reconstructions(
            scan_n,
            ct_imgs_train,
            phantom,
            ph_size,
            title=f"[Half Orignial] Reconstructed Slice ({num_scans // 2} views)",
            num_dp=num_dp,
        )

        bp_si_3 = ct_scan.plot_reconstructions(
            scan_2n,
            ct_imgs_lerp,
            phantom,
            ph_size,
            title=f"[Half Orignial, Half Lerp] Reconstructed Slice ({num_scans} views)",
            num_dp=num_dp,
        )
        bp_si_4 = ct_scan.plot_reconstructions(
            scan_2n,
            ct_imgs_lanczos,
            phantom,
            ph_size,
            title=f"[Half Orignial, Half lanczos] Reconstructed Slice ({num_scans} views)",
            num_dp=num_dp,
        )
        bp_si_5 = ct_scan.plot_reconstructions(
            scan_2n,
            ct_imgs_nerf,
            phantom,
            ph_size,
            title=f"[Half Orignial, Half Nerf] Reconstructed Slice ({num_scans} views)",
            num_dp=num_dp,
        )

        # Create GIFs of one interpolation method at a tiem
        RENDER_GIF = False
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

    return (
        spherical_angles,
        ct_imgs,
        poses,
        use_view_dirs,
        use_mono_ct,
        model_path,
        phantom,
        psnrs,
        render_kwargs_test,
        spherical_angles_train,
        poses_train,
        # Added late
        ct_imgs_nerf,
        recon_sirt,
    )


# Save data for evaluation later
def save_data(
    test_title_fn,
    phantom_idx,
    num_scans,
    n_iters,
    test_desc_note,
    ph_size,
    test_type,
    scan_type,
    spherical_angles,
    ct_imgs,
    poses,
    use_view_dirs,
    use_mono_ct,
    model_path,
    phantom,
    psnrs,
    render_kwargs_test,
    global_rng_seed,
    # Added late
    ct_imgs_pred,
    recon_pred,
    # Limited only
    limited_spherical_angles=None,
    limited_poses=None,
    limited_indices=None,
    pattern=None,
    # Sparse only
    spherical_angles_train=None,
    poses_train=None,
):

    experiment_data = {
        "test title": test_title_fn,
        "test description": test_desc_note,
        "phantom index": phantom_idx,
        "phantom size": ph_size,
        "number of scans": num_scans,
        "test type": test_type,
        "scan type": scan_type,
        "GT scan angles": spherical_angles,
        "GT scan images": ct_imgs,
        "GT scan poses": poses.detach().cpu().numpy(),
        "train iterations": n_iters,
        "used viewdirs": use_view_dirs,
        "used mono ct": use_mono_ct,
        "model path": model_path,
        "used ortho": True,
        "fov": False,
        "GT phantom": phantom,
        "psnrs": psnrs,
        "render kwargs": str(render_kwargs_test),
        "created at": utils.get_concise_timestamp(),
        "seed": global_rng_seed,
        # Added late
        "pred images": ct_imgs_pred,
        "pred recon": recon_pred,
    }

    if test_type == "limited scan":
        experiment_data.update(
            {
                "limited scan angles": limited_spherical_angles,
                "limited scan poses": limited_poses.detach().cpu().numpy(),
                "limited indices": limited_indices,  # full indices are just list(range(num_scans))
                "train pattern": pattern,
            }
        )

    if test_type == "sparse scan":
        experiment_data.update(
            {
                "train scan angles": spherical_angles_train,
                "train scan poses": poses_train.detach().cpu().numpy(),
            }
        )

    data_save_path = Path(
        f"./results/{test_title_fn}@{utils.get_concise_timestamp()}.pkl"
    )

    with open(data_save_path, "wb") as f:
        pickle.dump(experiment_data, f)

    print(f"Experiment data saved at {data_save_path}")


# %%
if __name__ == "__main__":
    ph_indexes = [13, 4, 16]
    for ph_idx in ph_indexes:
        train_and_save_nerf(
            # Real
            n_iters=10_000,
            video_ckpt=2_500,
            weights_ckpt=2_500,
            # Test
            phantom_idx=ph_idx,
            num_scans=256,  # This the total number of scans in test
            test_type="sparse scan",
        )
