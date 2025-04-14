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
    "seed",
    "psnrs",
    "train pattern",
]

# %%
filename = "prev/limited_nerf_ph-16_scans-32_iters-10000@2025-03-12#22-39.pkl"
path_to_data = Path(f"./results/{filename}")

with open(path_to_data, "rb") as f:
    experiment_data: dict[ExperimentDataKey, Any] = pickle.load(f)

# %%
gt_phantom = experiment_data["GT phantom"]
phantom_idx = experiment_data["phantom index"]
ct_imgs_full = experiment_data["GT scan images"]
img_res = ct_imgs_full.shape[1]
test_type = experiment_data["test type"]
poses = torch.from_numpy(experiment_data["GT scan poses"]).to(device)
full_spherical_angles = experiment_data["GT scan angles"]

full_scaner = ct_scan.AstraScanVec3D(
    gt_phantom.shape, full_spherical_angles, img_res=img_res
)
part_scaner = None


def lerp_ct_imgs(
    ct_imgs_partial,
    N,
    partial_indices,
):
    # 1. Initialization and Sorting
    H, W = ct_imgs_partial.shape[1:]
    ct_imgs_full = np.zeros((N, H, W), dtype=ct_imgs_partial.dtype)

    partial_indices = np.array(partial_indices)
    sort_order = np.argsort(partial_indices)
    partial_indices = partial_indices[sort_order]
    ct_imgs_partial = ct_imgs_partial[sort_order]
    K = len(partial_indices)

    # 2. Interpolate for each cyclic segment
    for j in range(K):
        start_idx = partial_indices[j]
        start_img = ct_imgs_partial[j]

        # Determine the next anchor (cyclic order)
        if j < K - 1:
            end_idx = partial_indices[j + 1]
            end_img = ct_imgs_partial[j + 1]
        else:
            # Wrap-around: adjust index and flip first image horizontally
            end_idx = partial_indices[0] + N
            end_img = np.flip(ct_imgs_partial[0], axis=1)

        gap = end_idx - start_idx

        # 3. Interpolate linearly for indices in the segment
        for i in range(start_idx, end_idx):
            t = (i - start_idx) / gap
            interp_img = (1 - t) * start_img + t * end_img
            ct_imgs_full[i % N] = interp_img

    return ct_imgs_full


# %%
if not (test_type == "limited scan" or test_type == "sparse scan"):
    raise ValueError("Invalid test type")

# %%
if test_type == "limited scan":
    part_spherical_angles = experiment_data["limited scan angles"]
    part_scaner = ct_scan.AstraScanVec3D(
        gt_phantom.shape, part_spherical_angles, img_res=img_res
    )

    model, render_kwargs_test = ct_nerf_interp.get_model(experiment_data["model path"])
    limited_indices = experiment_data["limited indices"]
    ct_train_set_imgs = ct_imgs_full[limited_indices]
    hwf = (img_res, img_res, experiment_data["fov"])

    ct_imgs_nvs = ct_nerf_interp.nerf_ct_imgs_limited(
        ct_imgs_partial=ct_train_set_imgs,
        full_poses=poses,
        hwf=hwf,
        render_kwargs=render_kwargs_test,
        partial_indices=limited_indices,
    )
    ct_imgs_lerp = lerp_ct_imgs(ct_train_set_imgs, len(ct_imgs_full), limited_indices)

# %%
if test_type == "sparse scan":
    part_spherical_angles = experiment_data["train scan angles"]
    part_scaner = ct_scan.AstraScanVec3D(
        gt_phantom.shape, part_spherical_angles, img_res=img_res
    )

    full_spherical_angles = experiment_data["train scan angles"]
    full_scaner = ct_scan.AstraScanVec3D(
        gt_phantom.shape, full_spherical_angles, img_res=img_res
    )
    ct_train_set_imgs = ct_imgs_full[::2]

    # TODO : Is this right, never tested it
    limited_indices = list(range(0, len(ct_imgs_full), 2))

# %% Create Train Set
ct_imgs_train_set = np.zeros_like(ct_imgs_full)
ct_imgs_train_set[limited_indices] = ct_imgs_full[limited_indices]


# %% [markdown]
# # Inpainting Methods
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
    ct_imgs_full.shape[0],
    ct_imgs_full.shape[1],
    ct_imgs_full.shape[2],
    limited_indices,
)

# %% Biharmonic


def inpaint_sinogram_slice(sinogram: np.ndarray, mask: np.ndarray):
    return inpaint_biharmonic(sinogram, mask, channel_axis=None)


def inpaint_sinogram(sinogram: np.ndarray, mask: np.ndarray):
    inpainted_sinogram = np.zeros_like(sinogram)
    for i in range(sinogram.shape[0]):
        sinogram_slice = sinogram[i]
        inpainted_sinogram[i] = inpaint_sinogram_slice(sinogram_slice, mask)
    return inpainted_sinogram


ct_imgs_biharmonic = inpaint_sinogram(
    ct_imgs_train_set.swapaxes(0, 1),
    mask,
).swapaxes(0, 1)


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


ct_imgs_ns = inpaint_sinogram_opencv(
    ct_imgs_train_set.swapaxes(0, 1),
    mask,
    method=cv2.INPAINT_NS,
    inpaint_radius=INFIL_INPAINT_RADIUS,
).swapaxes(0, 1)

ct_imgs_fmm = inpaint_sinogram_opencv(
    ct_imgs_train_set.swapaxes(0, 1),
    mask,
    method=cv2.INPAINT_TELEA,
    inpaint_radius=16,
).swapaxes(0, 1)


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
        num_iters=5_000,
    )

    for i in range(sinogram.shape[0]):
        sino_tv[i][mask == 0] = sinogram[i][mask == 0]

    plt.plot(losses)
    plt.show()

    return sino_tv


ct_imgs_tv = inpaint_tv_pytorch(
    ct_imgs_train_set.swapaxes(0, 1),
    mask,
).swapaxes(0, 1)

# %%
sirt_iter = 256

recon_sirt_full = full_scaner.reconstruct_3d_volume_sirt(
    ct_imgs_full, num_iterations=sirt_iter
)
recon_sirt_nvs = full_scaner.reconstruct_3d_volume_sirt(
    ct_imgs_nvs, num_iterations=sirt_iter
)
recon_sirt_train_set = part_scaner.reconstruct_3d_volume_sirt(
    ct_train_set_imgs, num_iterations=sirt_iter
)
recon_sirt_biharmonic = full_scaner.reconstruct_3d_volume_sirt(
    ct_imgs_biharmonic, num_iterations=sirt_iter
)
recon_sirt_ns = full_scaner.reconstruct_3d_volume_sirt(
    ct_imgs_ns, num_iterations=sirt_iter
)
recon_sirt_fmm = full_scaner.reconstruct_3d_volume_sirt(
    ct_imgs_fmm, num_iterations=sirt_iter
)
recon_sirt_tv = full_scaner.reconstruct_3d_volume_sirt(
    ct_imgs_tv, num_iterations=sirt_iter
)


# %% Setup
# normalize all gt, nvs and part reconstructions

titles_phantom_plot = [
    "GT Slice",
    "Full Scan Reconstruction",
    "Train Set Reconstruction",
    "NVS Reconstruction",
    "Biharmonic Reconstruction",
    "NS Reconstruction",
    "FMM Reconstruction",
    "TV Reconstruction",
]
all_phantoms = [
    gt_phantom,
    recon_sirt_full,
    recon_sirt_train_set,
    recon_sirt_nvs,
    recon_sirt_biharmonic,
    recon_sirt_ns,
    recon_sirt_fmm,
    recon_sirt_tv,
]
all_phantoms = [
    (phantom - phantom.min()) / (phantom.max() - phantom.min())
    for phantom in all_phantoms
]


titles_sino_plot = [
    "Full Sinogram",
    "Training Set Sinogram",
    # "Lerp Sinogram",
    "NVS Sinogram",
    "Biharmonic Sinogram",
    "NS Sinogram",
    "FMM Sinogram",
    "TV Sinogram",
]
all_ct_imgs = [
    ct_imgs_full,
    ct_imgs_train_set,
    # ct_imgs_lerp,
    ct_imgs_nvs,
    ct_imgs_biharmonic,
    ct_imgs_ns,
    ct_imgs_fmm,
    ct_imgs_tv,
]

print("Global min:", min(phantom.min() for phantom in all_phantoms))
print("Global max:", max(phantom.max() for phantom in all_phantoms))


# %%
@dataclass
class Method:
    title: str
    reconstructed_images: np.ndarray


def plot_methods(target_idx: int, methods: list[Method], is_sinogram: bool = False):
    n_cols = 4
    len_methods = len(methods) + (1 if is_sinogram else 0)
    n_rows = math.ceil(len_methods / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5.1, n_rows * 5), constrained_layout=True
    )
    axs = axs.flatten()

    method_idx = 0
    for i, ax in enumerate(axs):
        # Skip the first subplot if it's a sinogram, as this where the GT is shown
        if is_sinogram and i == 0:
            continue

        method = methods[method_idx]
        method_idx += 1

        im = ax.imshow(
            method.reconstructed_images[target_idx],
            cmap="gray",
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        ax.set_title(method.title, fontsize=20)
        ax.axis("off")

    # remove empty subplots
    for i in range(len(axs)):
        axs[i].axis("off")

    # Add a colorbar
    cbar = fig.colorbar(
        im, ax=axs, orientation="vertical", fraction=0.046, pad=0.04, aspect=40
    )
    cbar.set_label("Intensity (a.u.)", labelpad=20, fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    plt.show()


MetricMethods = Literal["SSIM", "Absoulte Difference"]


def compare_methods(
    target_idx: int,
    metric: MetricMethods,
    methods: list[Method],
    is_sinogram: bool = False,
):
    gt_method = methods[0]

    n_cols = 4
    len_methods = len(methods) + (1 if is_sinogram else 0)
    n_rows = math.ceil(len_methods / n_cols)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 5.1, n_rows * 5), constrained_layout=True
    )
    axs = axs.flatten()

    method_idx = 1
    for i, ax in enumerate(axs):
        if i == 0:
            # we skip this as its where the GT is shown
            continue
        if is_sinogram and i == 1:
            # we skip this on sinogram as this what we compare to
            continue

        method = methods[method_idx]
        method_idx += 1

        if metric == "SSIM":
            _, cmp_img = ssim(
                gt_method.reconstructed_images[target_idx],
                method.reconstructed_images[target_idx],
                full=True,
                data_range=1.0,
            )
        if metric == "Absoulte Difference":
            cmp_img = np.abs(
                gt_method.reconstructed_images[target_idx]
                - method.reconstructed_images[target_idx]
            )

        im = ax.imshow(cmp_img, vmin=0, vmax=1, cmap="magma", aspect="auto")
        ax.set_title(method.title, fontsize=20)

    # remove empty subplots
    for i in range(len(axs)):
        axs[i].axis("off")

    # add a colorbar
    cbar = fig.colorbar(
        im, ax=axs, orientation="vertical", fraction=0.046, pad=0.04, aspect=40
    )
    cbar.set_label(f"{metric} (a.u.)", labelpad=20, fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    plt.show()


def evaluate_methods(target_idx: int, methods: list[Method], is_sinogram: bool = False):
    plot_methods(target_idx, methods, is_sinogram)
    compare_methods(target_idx, "Absoulte Difference", methods, is_sinogram)
    compare_methods(target_idx, "SSIM", methods, is_sinogram)


sliced_methods = [
    Method(title, phantom) for title, phantom in zip(titles_phantom_plot, all_phantoms)
]
evaluate_methods(
    len(gt_phantom) // 2,
    sliced_methods,
    is_sinogram=False,
)


# %% # Sinogram
sinogramed_methods = [
    Method(title, ct_imgs.swapaxes(0, 1))
    for title, ct_imgs in zip(titles_sino_plot, all_ct_imgs)
]
idx = sinogramed_methods[0].reconstructed_images.shape[0] // 2

evaluate_methods(
    idx,
    sinogramed_methods,
    is_sinogram=True,
)


# %% Get metrics
def get_metrics_phantom(gt_phantom: np.ndarray, methods: list[Method]) -> pd.DataFrame:
    metrics = []
    for method in methods:
        ssim_val = utils.ssim_3d(method.reconstructed_images, gt_phantom)
        psnr_val = utils.psnr(method.reconstructed_images, gt_phantom)
        metrics.append((method.title, ssim_val, psnr_val))
    return pd.DataFrame(
        metrics,
        columns=["Method", "SSIM", "PSNR"],
    )


df_metrics = get_metrics_phantom(gt_phantom, sliced_methods)
# %% Save metrics
# metrics_{phantom_idx}_{experiment type}_{experiment name}
title_df = f"metrics_{experiment_data['phantom index']}_{experiment_data['test type']}_{utils.get_concise_timestamp()}"
title_df = title_df.replace(" ", "_")

df_metrics.to_csv(f"tables/{title_df}.csv", index=False)
