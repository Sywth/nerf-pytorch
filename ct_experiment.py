# %% [markdown]
# ## Setup

# %%
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import tqdm
import cv2

import ct_nerf_interp
import ct_scan
import utils

from skimage.restoration import inpaint_biharmonic
from skimage.metrics import structural_similarity as ssim
from typing import Any, Literal, Union
from pathlib import Path
from dataclasses import dataclass
import sys

# %%
# #GPU
torch.cuda.empty_cache()
torch.set_default_tensor_type("torch.cuda.FloatTensor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
ExperimentDataKey = Literal[
    "test title",
    "test description",
    "phantom index",
    "phantom size",
    "number of scans",
    "test type",
    "scan type",
    "GT scan angles",
    "GT scan images",
    "GT scan poses",
    "train iterations",
    "used viewdirs",
    "used mono ct",
    "model path",
    "used ortho",
    "fov",
    "GT phantom",
    "GT Reconstructed SIRT",
    "GT Reconstructed CGLS",
    "Train Reconstructed SIRT",
    "Train Reconstructed CGLS",
    "NVS SIRT",
    "NVS CGLS",
    "render kwargs",
    "created at",
    "limited scan angles",
    "limited scan poses",
    "limited indices",
    "train scan angles",
    "train scan poses",
    # Added late
    "pred images",
    "pred recon",
    # end
    "seed",
    "psnrs",
    "train pattern",
]
# %%
# Parse CLI arguments
SHOW_PLOT = False
CLI_MODE = True

if CLI_MODE:
    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python ct_experiment.py <path_to_base_folder> <phantom_index>"
        )
    fp_base = Path(sys.argv[1])
    ph_idx = int(sys.argv[2])
else:
    fp_base = Path("results/limited-256")
    ph_idx = 13

# Load Gaussian Data
nf_pkl_matches = list(fp_base.glob(f"*_ph-{ph_idx}_*.pkl"))
assert len(nf_pkl_matches) == 1, "There should exactly one pkl file"
path_to_data = nf_pkl_matches[0]

with open(path_to_data, "rb") as f:
    experiment_data: dict[ExperimentDataKey, Any] = pickle.load(f)


# Normalize NeRF and GS data
def normalize_data(data: np.ndarray) -> np.ndarray:
    return (data - data.min()) / (data.max() - data.min())


# fmt : off
# Load Gaussian Data
path_to_gs_folder = fp_base / Path(f"gs/{ph_idx}/")
ct_imgs_full_gs = normalize_data(np.load(path_to_gs_folder / "ct_imgs_gt.npy"))
gt_phantom_gs = normalize_data(np.load(path_to_gs_folder / "vol_gt.npy")).transpose(
    2, 0, 1
)
ct_imgs_gs = normalize_data(np.load(path_to_gs_folder / "ct_imgs_pred.npy"))
recon_gs = normalize_data(np.load(path_to_gs_folder / "recon_pred.npy")).transpose(
    2, 0, 1
)

# Load NeRF Data
ct_imgs_full_nf = normalize_data(experiment_data["GT scan images"])
gt_phantom_nf = normalize_data(experiment_data["GT phantom"])
ct_imgs_nf = normalize_data(experiment_data["pred images"])
recon_nf = normalize_data(experiment_data["pred recon"])
# fmt : on

# Misc Data from NeRF
phantom_idx = gt_phantom_nf.shape[0] // 2
img_res = ct_imgs_full_nf.shape[1]
test_type = experiment_data["test type"]
poses = torch.from_numpy(experiment_data["GT scan poses"]).to(device)
full_spherical_angles = experiment_data["GT scan angles"]
hwf = (img_res, img_res, experiment_data["fov"])

full_scanner = ct_scan.AstraScanVec3D(
    gt_phantom_nf.shape, full_spherical_angles, img_res=img_res
)
part_scaner = None
sino_idx = ct_imgs_full_nf.shape[1] // 2

psnrs = experiment_data["psnrs"]
plt.plot(psnrs)
# %%
if not (test_type == "limited scan" or test_type == "sparse scan"):
    raise ValueError("Invalid test type")

num_scans = experiment_data["number of scans"]
folder = f"./out/dump_[{experiment_data['test type']}][{experiment_data['phantom index']}][{num_scans}]#{utils.get_concise_timestamp()}"


def plot_fig(title: str):
    plt.savefig(f"{folder}/{title}.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOT:
        plt.show()
    plt.clf()


# Create and save desc
Path(folder).mkdir(parents=True, exist_ok=True)
with open(f"{folder}/desc.txt", "w") as f:
    f.write(experiment_data["test description"])


# %% Method Specific Constants
if test_type == "limited scan":
    part_spherical_angles = experiment_data["limited scan angles"]
    part_scaner = ct_scan.AstraScanVec3D(
        gt_phantom_nf.shape, part_spherical_angles, img_res=img_res
    )

    limited_indices = experiment_data["limited indices"]

    # angle biz
    part_scan_angles = np.array(experiment_data["limited scan angles"])
    full_scan_angles = np.array(experiment_data["GT scan angles"])
    unknown_angles = np.setdiff1d(full_scan_angles, part_scan_angles)

if test_type == "sparse scan":
    part_spherical_angles = experiment_data["train scan angles"]
    part_scaner = ct_scan.AstraScanVec3D(
        gt_phantom_nf.shape, part_spherical_angles, img_res=img_res
    )

    limited_indices = list(range(0, len(ct_imgs_full_nf), 2))

    # angle biz
    part_scan_angles = np.array(experiment_data["train scan angles"])
    full_scan_angles = np.array(experiment_data["GT scan angles"])
    unknown_angles = np.setdiff1d(full_scan_angles, part_scan_angles)

ct_imgs_train_set = np.zeros_like(ct_imgs_full_nf)
ct_imgs_train_set[limited_indices] = ct_imgs_full_nf[limited_indices]


# %% angle biz
# plot angles
utils.plot_angles(
    full_spherical_angles,
    limited_indices,
    title="Full Scan Angles",
)
plt.tight_layout()
plot_fig("full_scan_angles")


# %% [markdown]
# ## Inpainting Methods
# %% Get Mask
def get_mask(n, h, w, limited_indices):
    """
    Returns 2-D mask, 0 if known, 1 if missing.
    """
    missing_indices = set(range(n)) - set(limited_indices)
    mask = np.zeros((n, w))
    mask[sorted(missing_indices), :] = 1
    return mask


mask = get_mask(
    ct_imgs_full_nf.shape[0],
    ct_imgs_full_nf.shape[1],
    ct_imgs_full_nf.shape[2],
    limited_indices,
)


# %% Rolled data
# To Make algorithms work better by updating mask and ct_imgs_train_set with a
#   np.roll of offset=12 in axis=0


def roll_forward(arr: np.ndarray, roll_amount: int = -12) -> np.ndarray:
    return np.roll(arr, shift=roll_amount, axis=0)


def roll_backward(arr: np.ndarray, roll_amount: int = 12) -> np.ndarray:
    return np.roll(arr, shift=roll_amount, axis=0)


ct_imgs_train_set_rolled = roll_forward(ct_imgs_train_set)
mask_rolled = roll_forward(mask)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(
    ct_imgs_train_set_rolled.swapaxes(0, 1)[sino_idx], cmap="gray", aspect="auto"
)
ax[0].set_title("Rolled CT Images")
ax[1].imshow(mask_rolled, cmap="gray", aspect="auto")
ax[1].set_title("Rolled Mask")
plt.tight_layout()
plot_fig("rolled_mask")


# %% Biharmonic
def inpaint_sinogram_slice(sinogram: np.ndarray, mask: np.ndarray):
    return inpaint_biharmonic(sinogram, mask, channel_axis=None)


def inpaint_sinogram(sinogram: np.ndarray, mask: np.ndarray):
    inpainted_sinogram = np.zeros_like(sinogram)
    for i in range(sinogram.shape[0]):
        sinogram_slice = sinogram[i]
        inpainted_sinogram[i] = inpaint_sinogram_slice(sinogram_slice, mask)
    return inpainted_sinogram


ct_imgs_biharmonic_rolled = inpaint_sinogram(
    ct_imgs_train_set_rolled.swapaxes(0, 1),
    mask_rolled,
).swapaxes(0, 1)
ct_imgs_biharmonic = roll_backward(ct_imgs_biharmonic_rolled)

# %% Tela FMM & Navier Stokes
InpaintMethodFlagType = Literal[
    cv2.INPAINT_NS,
    cv2.INPAINT_TELEA,
]

INFIL_INPAINT_RADIUS = 6
INTERP_INPAINT_RADIUS = 3


def inpaint_sinogram_slice_opencv(
    sinogram_slice: np.ndarray,
    mask_slice: np.ndarray,
    method: InpaintMethodFlagType,
    inpaint_radius: int = 3,
) -> np.ndarray:
    """
    NOTE : `sinogram_slice` must be normalized to [0, 1] before inpainting.
    """
    assert sinogram_slice.min() >= 0 and sinogram_slice.max() <= 1

    # OpenCV expects 8-bit or 32-bit single-channel float
    sinogram_uint8 = (sinogram_slice * 255).clip(0, 255).astype(np.uint8)
    mask_uint8 = (mask_slice > 0).astype(np.uint8)

    inpainted_sinogram_uint8 = cv2.inpaint(
        sinogram_uint8, mask_uint8, inpaintRadius=inpaint_radius, flags=method
    )
    return inpainted_sinogram_uint8.astype(np.float32) / 255.0


def inpaint_sinogram_opencv(
    sinogram: np.ndarray,
    mask: np.ndarray,
    method: InpaintMethodFlagType,
    inpaint_radius: int = 3,
) -> np.ndarray:
    inpainted_sinogram = np.zeros_like(sinogram, dtype=np.float32)
    for i in range(sinogram.shape[0]):
        sinogram_slice = sinogram[i]
        inpainted_slice = inpaint_sinogram_slice_opencv(
            sinogram_slice, mask, method, inpaint_radius
        )
        inpainted_sinogram[i] = inpainted_slice
    return inpainted_sinogram


ct_imgs_ns_rolled = inpaint_sinogram_opencv(
    ct_imgs_train_set_rolled.swapaxes(0, 1),
    mask_rolled,
    method=cv2.INPAINT_NS,
    inpaint_radius=INFIL_INPAINT_RADIUS,
).swapaxes(0, 1)
ct_imgs_ns = roll_backward(ct_imgs_ns_rolled)

ct_imgs_fmm = inpaint_sinogram_opencv(
    ct_imgs_train_set_rolled.swapaxes(0, 1),
    mask_rolled,
    method=cv2.INPAINT_TELEA,
    inpaint_radius=16,
).swapaxes(0, 1)
ct_imgs_fmm = roll_backward(ct_imgs_fmm)


# %% TVR De-Noising
def tv_norm(U, epsilon=1e-6):
    """
    Compute a differentiable 3D TV norm on a tensor U.
    U is assumed to be of shape (D, H, W).
    """
    # Compute forward differences along each axis
    dx = U[:, :, 1:] - U[:, :, :-1]
    dy = U[:, 1:, :] - U[:, :-1, :]
    dz = U[1:, :, :] - U[:-1, :, :]

    # Compute L2 norm of the gradients with a small epsilon to avoid singularities
    tv_x = torch.sqrt(dx**2 + epsilon)
    tv_y = torch.sqrt(dy**2 + epsilon)
    tv_z = torch.sqrt(dz**2 + epsilon)

    return tv_x.sum() + tv_y.sum() + tv_z.sum()


def inpaint_tv_pytorch_core(sinogram, mask, num_iters=200, lr=1e-2, tv_weight=0.1):
    """
    Inpaint a 3D sinogram (voxel grid) with missing regions specified by mask.
    - sinogram: np.ndarray of shape (D, H, W)
    - mask: np.ndarray of shape (D, H, W) with 1 indicating missing data and 0 known.

    The mask is inverted so that known pixels are weighted in the data fidelity term.
    """
    # Convert inputs to torch tensors and move to GPU
    sinogram_t = torch.tensor(sinogram, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    # Invert mask: now 1 = known, 0 = missing
    mask_t = 1 - mask_t

    # Initialize U with the sinogram values; U is our optimization variable
    U = sinogram_t.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([U], lr=lr)
    losses: list[float] = []
    for i in tqdm.tqdm(range(num_iters), desc="TV Inpainting Optimization"):
        optimizer.zero_grad()

        # Data fidelity: enforce that U remains close to sinogram on known voxels.
        data_loss = torch.norm(mask_t * (U - sinogram_t)) ** 2
        # TV regularization promotes piecewise smoothness while preserving edges.
        tv_loss = tv_norm(U)

        loss = data_loss + tv_weight * tv_loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 500 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")

    return U.detach().cpu().numpy(), losses


def inpaint_tv_pytorch(
    sinogram: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    # Apply the TV inpainting method on sinogram data.
    sino_tv, losses = inpaint_tv_pytorch_core(
        sinogram,
        mask,
        tv_weight=0.1,
    )
    sino_tv_original = sino_tv.copy()

    for i in range(sinogram.shape[0]):
        sino_tv[i][mask == 0] = sinogram[i][mask == 0]

    plt.plot(losses)
    plt.yscale("log")
    plt.title("TV Inpainting Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.tight_layout()
    plot_fig("tv_loss")

    return sino_tv, sino_tv_original


sino_tv, sino_tv_original = inpaint_tv_pytorch(
    ct_imgs_train_set_rolled.swapaxes(0, 1),
    mask_rolled,
)
ct_imgs_tv_rolled = sino_tv.swapaxes(0, 1)
ct_imgs_tv = roll_backward(ct_imgs_tv_rolled)

plt.imshow(sino_tv_original[sino_idx])
plt.show()
plt.imshow(sino_tv[sino_idx])
plt.show()
plt.imshow(sino_tv.swapaxes(0, 1)[ph_idx])
plt.show()
plt.imshow(sino_tv_original.swapaxes(0, 1)[ph_idx])
plt.show()


# %% Lerp
def lerp_ct_imgs(
    ct_imgs: np.ndarray,
    indices_mask: list[int] = None,
    num_total_imgs: int = 64,
) -> np.ndarray:
    # 2. Precompute interpolated images
    interpolated_imgs = ct_imgs.copy()
    for i in range(num_total_imgs):
        if indices_mask[i] == 1:
            continue

        # NOTE : This should need to be flipped if
        #   we go over the end of the array on either side
        #   but adding that logic appears to be wrong

        # 3. Find nearest valid frames (cyclically)
        prev_idx = (i - 1) % num_total_imgs
        while indices_mask[prev_idx] == 0:
            prev_idx = (prev_idx - 1) % num_total_imgs

        next_idx = (i + 1) % num_total_imgs
        while indices_mask[next_idx] == 0:
            next_idx = (next_idx + 1) % num_total_imgs

        # 4. Compute interpolation weights
        total_dist = (next_idx - prev_idx) % num_total_imgs
        dist_to_prev = (i - prev_idx) % num_total_imgs
        assert total_dist > 0, "Invalid distance between frames"
        alpha = dist_to_prev / total_dist

        # 5. Linear interpolation
        prev_img = ct_imgs[prev_idx]
        next_img = ct_imgs[next_idx]

        interpolated_imgs[i] = utils.lerp(prev_img, next_img, alpha)

    return interpolated_imgs


indices_mask = np.zeros(len(ct_imgs_full_nf))
indices_mask[limited_indices] = 1

# Rolls automatically
ct_imgs_lerp = lerp_ct_imgs(
    ct_imgs_train_set,
    indices_mask,
    ct_imgs_full_nf.shape[0],
)

plt.imshow(
    np.roll(ct_imgs_lerp.swapaxes(0, 1)[ct_imgs_lerp.shape[1] // 2], 12, axis=0),
    aspect=3,
)
plt.tight_layout()
plot_fig("rolled_lerp")


# %% Sinogram Data Setup
titles_sino_plot = [
    "Full Sinogram",
    "Training Set Sinogram",
    "LERP Sinogram",
    "NeRF Sinogram",
    "Biharmonic Sinogram",
    "NS Sinogram",
    "FMM Sinogram",
    "TV Sinogram",
    "GS Sinogram",
]
all_ct_imgs = [
    ct_imgs_full_nf,
    ct_imgs_train_set,
    ct_imgs_lerp,
    ct_imgs_nf,
    ct_imgs_biharmonic,
    ct_imgs_ns,
    ct_imgs_fmm,
    ct_imgs_tv,
    ct_imgs_gs,  # DO NOT USE THIS FOR RECON
]
# %% Reoncstruction
titles_phantom_plot = [
    "GT Slice",
    "Full Scan Reconstruction",
    "Train Set Reconstruction",
    "LERP Reconstruction",
    "NeRF Reconstruction",
    "Biharmonic Reconstruction",
    "NS Reconstruction",
    "FMM Reconstruction",
    "TV Reconstruction",
    "GS Reconstruction",
]
all_reconstructions = [gt_phantom_nf]
use_part_scanner_idx = 1
gaussian_idx = len(all_ct_imgs) - 1
for i, ct_imgs in enumerate(tqdm.tqdm(all_ct_imgs, desc="Reconstructing volumes")):
    if i == use_part_scanner_idx:
        ct_imgs = ct_imgs[limited_indices]
        scanner = part_scaner
    else:
        scanner = full_scanner

    if i == gaussian_idx:
        recon = recon_gs
    else:
        recon = scanner.reconstruct_3d_volume_sirt(ct_imgs, num_iterations=256)

    recon = normalize_data(recon)
    all_reconstructions.append(recon)

print(
    "Reconstruction Global min:", min(phantom.min() for phantom in all_reconstructions)
)
print(
    "Reconstruction Global max:", max(phantom.max() for phantom in all_reconstructions)
)
# %%
# Make Sinograms
gt_sino_nf = ct_imgs_full_nf.swapaxes(0, 1)
gt_sino_gs = ct_imgs_full_gs.swapaxes(0, 1)
assert gt_sino_nf.shape == gt_sino_gs.shape, "GT sinograms should be the same shape"


def plot_images(
    data_list,
    titles,
    save_path,
    cmap="gray",
    fig_title="",
    colorbar_label="Intensity (a.u.)",
    vmin=0,
    vmax=1,
):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), constrained_layout=True)
    fig.suptitle(fig_title, fontsize=26, y=1.05)
    axes = axes.flatten()

    data_plot_diff = len(axes) - len(data_list)

    for ax, title, img in zip(axes[data_plot_diff:], titles, data_list):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=20)

    for ax in axes:
        ax.axis("off")

    cbar = fig.colorbar(
        im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02, aspect=30
    )
    cbar.set_label(colorbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    plot_fig(save_path)


# Example usage:
# Plot Phantoms
plot_images(
    data_list=[recon[phantom_idx] for recon in all_reconstructions],
    titles=titles_phantom_plot,
    cmap="gray",
    colorbar_label="Intensity (a.u.)",
    save_path="phantom",
    fig_title="Reconstructed Phantoms Sliced",
)

# Plot Sinograms
plot_images(
    data_list=[img.swapaxes(0, 1)[sino_idx] for img in all_ct_imgs],
    titles=titles_sino_plot,
    cmap="gray",
    colorbar_label="Intensity (a.u.)",
    save_path="sinogram",
    fig_title="Sinograms Sliced",
)

# Pot Phantom AD
plot_images(
    data_list=[
        np.abs(recon[phantom_idx] - gt_phantom_nf[phantom_idx])
        for recon in all_reconstructions[1:]
    ],
    titles=titles_phantom_plot[1:],
    cmap="magma",
    colorbar_label="Absolute Deviation (a.u.)",
    save_path="phantom_abs_diff",
    fig_title="Phantom Absolute Differences",
)

# Plot Sinogram AD
plot_images(
    data_list=[
        np.abs(img.swapaxes(0, 1)[sino_idx] - gt_sino_nf[sino_idx])
        for img in all_ct_imgs[1:]
    ],
    titles=titles_sino_plot[1:],
    cmap="magma",
    colorbar_label="Absolute Deviation (a.u.)",
    save_path="sinogram_abs_diff",
    fig_title="Sinogram Absolute Differences",
)

# Plot Phantom SSIM Gradients
ssim_phantoms = np.array(
    [
        ssim(
            gt_phantom_nf[phantom_idx],
            recon[phantom_idx],
            gradient=True,
            full=True,
            data_range=1.0,
        )[1:3]
        for recon in all_reconstructions[1:]
    ]
)
ssim_ph_gradient_imgs = ssim_phantoms[:, 0, :, :]
ssim_ph_conv_imgs = ssim_phantoms[:, 1, :, :]

plot_images(
    data_list=ssim_ph_gradient_imgs,
    titles=titles_phantom_plot[1:],
    cmap="magma",
    vmin=np.min(ssim_ph_gradient_imgs),
    vmax=np.max(ssim_ph_gradient_imgs),
    colorbar_label="Pointwise SSIM Gradient (a.u.)",
    save_path="phantom_ssim_grad",
    fig_title="Phantom SSIM Gradients",
)

# Plot Sinogram SSIM Gradients
ssim_sinos = np.array(
    [
        ssim(
            gt_sino_nf[sino_idx],
            img.swapaxes(0, 1)[sino_idx],
            gradient=True,
            full=True,
            data_range=1.0,
        )[1:3]
        for img in all_ct_imgs[1:]
    ]
)
ssim_si_gradient_imgs = ssim_sinos[:, 0, :, :]
ssim_si_conv_imgs = ssim_sinos[:, 1, :, :]

plot_images(
    data_list=ssim_si_gradient_imgs,
    titles=titles_sino_plot[1:],
    cmap="magma",
    vmin=np.min(ssim_si_gradient_imgs),
    vmax=np.max(ssim_si_gradient_imgs),
    colorbar_label="Pointwise SSIM Gradient (a.u.)",
    save_path="sinogram_ssim_grad",
    fig_title="Sinogram SSIM Gradients",
)

# Plot Phantom SSIM Convolutions
plot_images(
    data_list=ssim_ph_conv_imgs,
    titles=titles_phantom_plot[1:],
    cmap="magma",
    vmin=np.min(ssim_ph_conv_imgs),
    vmax=np.max(ssim_ph_conv_imgs),
    colorbar_label="Pointwise SSIM Convolution (a.u.)",
    save_path="phantom_ssim_conv",
    fig_title="Phantom SSIM Convolutions",
)

# Plot Sinogram SSIM Convolutions
plot_images(
    data_list=ssim_si_conv_imgs,
    titles=titles_sino_plot[1:],
    cmap="magma",
    vmin=np.min(ssim_si_conv_imgs),
    vmax=np.max(ssim_si_conv_imgs),
    colorbar_label="Pointwise SSIM Convolution (a.u.)",
    save_path="sinogram_ssim_conv",
    fig_title="Sinogram SSIM Convolutions",
)


# %%
# Get metrics
def get_metrics_phantom(
    titles: list[str], recons: list[np.ndarray], gs_idx: int
) -> pd.DataFrame:
    metrics = []
    for i, (title, recon) in tqdm.tqdm(
        enumerate(zip(titles, recons)), total=len(titles)
    ):
        gt = gt_phantom_gs if i == gs_idx else gt_phantom_nf
        ssim_val = utils.ssim_3d(recon, gt)
        psnr_val = utils.psnr(recon, gt)

        metrics.append((title, ssim_val, psnr_val))
    return pd.DataFrame(
        metrics,
        columns=["Method", "SSIM", "PSNR"],
    )


df_metrics = get_metrics_phantom(
    titles_phantom_plot,
    all_reconstructions,
    9,
)
# %%
# Save metrics
title_df = f"metrics-{'limited' if test_type == 'limited scan' else 'sparse'}-{experiment_data['phantom index']}-{num_scans}"
path_df = Path(folder) / Path(f"{title_df}.csv")
df_metrics.to_csv(path_df, index=False)
df_metrics

# %% DEBUG IGNORE
if __name__ == "__main__" and False:
    fig, ax = plt.subplots(figsize=(4, 5))  # Height = 2 × Width
    ax.set_title("Initial Sinogram")
    img = ct_imgs_train_set[limited_indices].swapaxes(0, 1)[sino_idx]

    im = ax.imshow(
        img, cmap="gray", aspect="auto"
    )  # Let the figure size dictate the aspect
    ax.set_xlabel("Width ($W$)")
    ax.set_ylabel(r"Projection Index ($N$)")

    limited_indices = np.array(limited_indices)
    tick_step = 10
    tick_positions = np.arange(0, len(limited_indices), tick_step)
    tick_labels = limited_indices[tick_positions]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()
    plt.show()

    # %%

    fig, ax = plt.subplots(figsize=(4, 5))  # Height = 2 × Width
    ax.set_title("Re-stitched Sinogram")

    img = ct_imgs_train_set.swapaxes(0, 1)[sino_idx]

    ax.imshow(img, cmap="gray", aspect="auto")  # Let figure size control appearance
    ax.set_xlabel("Width ($W$)")
    ax.set_ylabel(r"Projection Index ($N$)")

    plt.tight_layout()
    plt.show()

    # %%
    # Mask
    fig, ax = plt.subplots(figsize=(4, 5))  # Height = 2 × Width
    ax.set_title("Mask")
    img = mask
    ax.imshow(img, cmap="gray", aspect="auto")  # Let figure size control appearance
    ax.set_xlabel("Width ($W$)")
    ax.set_ylabel(r"Projection Index ($N$)")
    plt.tight_layout()
    plt.show()

    # %%
    num_slices = 10
    phantom_indices = [4, 13, 16]
    ph_size = 128
    slice_indices = np.linspace(0, ph_size - 1, num_slices).astype(np.int32)

    fig, axes = plt.subplots(
        len(phantom_indices),
        num_slices,
        figsize=(num_slices * 2, len(phantom_indices) * 2),
    )

    for row_idx, ph_idx in enumerate(phantom_indices):
        tomo_ph = ct_scan.load_phantom(ph_idx, ph_size)
        for col_idx, slice_idx in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(tomo_ph[slice_idx], cmap="gray")
            ax.axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.suptitle("Phantom Slices", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
