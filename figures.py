# %%
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import astra
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.mplot3d import Axes3D
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from matplotlib import font_manager


# %% Show absolute deviaton
# load data/baboon.png


def fmt(x, pos):
    a, b = "{:.1e}".format(x).split("e")
    b = int(b)
    return r"${} \times 10^{{{}}}$".format(a, b)


formatter = ticker.FuncFormatter(fmt)
font_prop = font_manager.FontProperties(family="serif", size=12)


def plot_iamges(
    image,
    noised_image,
    diff_image,
    diff_title,
):
    n_imgs = 3
    fig, axes = plt.subplots(1, n_imgs, figsize=(12, 4), gridspec_kw={"wspace": 0.05})

    # Plot images
    axes[0].imshow(image, cmap="gray", interpolation="nearest")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(noised_image, cmap="gray", interpolation="nearest")
    axes[1].set_title("Noised Image")
    axes[1].axis("off")

    im2 = axes[2].imshow(diff_image, cmap="magma", interpolation="nearest")
    axes[2].set_title(diff_title)
    axes[2].axis("off")

    # Colorbar 1 (same height as figure, right aligned)
    cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
    cbar = fig.colorbar(im2, cax=cbar_ax, format=formatter)
    cbar.set_label(
        "a.u.",
        rotation=270,
        labelpad=22,
        fontproperties=font_prop,
    )
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(rect=[0, 0, 0.9, 1])


image = Image.open("data/baboon.png").convert("L")
image = np.array(image, dtype=np.float32) / 255.0
np.random.seed(8)

# Add blotched nosie the image
noised_image = image.copy()
h, w = noised_image.shape
for _ in range(5):  # Add 5 blotches
    rand_h = np.random.randint(0, h // 2)
    rand_w = np.random.randint(0, w // 2)
    start_h = np.random.randint(0, h - rand_h)
    start_w = np.random.randint(0, w - rand_w)
    noised_image[
        start_h : start_h + rand_h, start_w : start_w + rand_w
    ] += np.random.rand()

# add gaussian noise
sigma = 0.1
noise = np.random.normal(0, sigma, image.shape) * 1.5
noised_image += noise


noised_image = np.clip(noised_image, 0, 1)
img_abs_diff = np.abs(image - noised_image)
abs_dev = np.sum(img_abs_diff) / (h * w)

ssim_val, img_ssim_grad, img_ssim_convs = ssim(
    image,
    noised_image,
    full=True,
    data_range=1.0,
    gradient=True,
)
avg_grad = np.mean(img_ssim_grad)

# Compute pointwise PSNR
epsilon = 1e-10  # Small constant to avoid log(0)
mse_pointwise = (image - noised_image) ** 2
img_max = 1.0
img_psnr = 10 * np.log10(img_max / (mse_pointwise + epsilon))
psnr_val = np.mean(img_psnr)

print("SSIM: ", ssim_val)
plot_iamges(
    image,
    noised_image,
    img_abs_diff,
    f"Pointwise Absolute Deviation ({float(abs_dev):.4f})",
)
plot_iamges(image, noised_image, img_psnr, f"Pointwise PSNR ({float(psnr_val):.4f})")
plot_iamges(
    image,
    noised_image,
    img_ssim_grad,
    f"Pointwise SSIM Gradient ({float(avg_grad):.2g})",
)
plot_iamges(
    image,
    noised_image,
    img_ssim_convs,
    f"Pointwise SSIM Convolutions ({float(ssim_val):.4f})",
)

# %% Transmittance vs Time
# Time domains
import numpy as np
import matplotlib.pyplot as plt

# Time domain
t_min = 0.0
t_max = 10.0
steps = 1000
dt = (t_max - t_min) / steps
t = np.linspace(t_min, t_max, steps)


# Ground truth function
def sigma_gt_fn(t):
    def gauss(mu, sigma, amp=1.0):
        return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    box = np.where((t >= 2) & (t <= 8), 100, 0)
    spike1 = gauss(5, 0.5, 200)
    spike2 = gauss(6, 0.18, 130)
    noise = np.random.normal(scale=2.0, size=t.shape)
    return box + spike1 + spike2 + noise


# Evaluate true volume density and transmittance
sigma_gt = sigma_gt_fn(t)
transmittance = np.cumsum(sigma_gt) * dt

# Sample points for discretization
sample_indices = np.sort(np.random.choice(len(t), size=30, replace=False))
t_samples = t[sample_indices]
sigma_samples = sigma_gt[sample_indices]

# Stepwise volume density
t_step = []
sigma_step = []
for i in range(len(t_samples) - 1):
    t_step.extend([t_samples[i], t_samples[i + 1]])
    sigma_step.extend([sigma_samples[i], sigma_samples[i]])
t_step.append(t_samples[-1])
sigma_step.append(sigma_samples[-1])

# Stepwise transmittance (discrete integral approximation)
T_step = [0.0]
for i in range(1, len(t_samples)):
    delta_t = t_samples[i] - t_samples[i - 1]
    T_step.append(T_step[-1] + sigma_samples[i - 1] * delta_t)

# Create ZOH-style transmittance curve
t_trans_step = []
T_trans_step = []
for i in range(len(t_samples) - 1):
    t_trans_step.extend([t_samples[i], t_samples[i + 1]])
    T_trans_step.extend([T_step[i], T_step[i]])
t_trans_step.append(t_samples[-1])
T_trans_step.append(T_step[-1])

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Left: sigma (volume density)
axs[0].plot(t, sigma_gt, label="Ground Truth", color="orange", linewidth=1.5)
axs[0].plot(
    t_step, sigma_step, label="Discretized", color="blue", linestyle="--", linewidth=1.5
)
axs[0].set(
    xlabel="t",
    ylabel=r"Volume Density ($\sigma$)",
    title="Volume Density ($\sigma$) against Time ($t$)",
)
axs[0].legend()

# Right: transmittance (integrated)
axs[1].plot(t, transmittance, label="Ground Truth", color="orange", linewidth=1.5)
axs[1].plot(
    t_trans_step,
    T_trans_step,
    label="Discretized",
    color="blue",
    linestyle="--",
    linewidth=1.5,
)
axs[1].set(
    xlabel="t",
    ylabel="Transmittance (T)",
    title="Transmittance ($T$) against Time ($t$)",
)
axs[1].legend()
plt.tight_layout()
plt.show()


# %% SPARSE
# --- 1. Load & Resize Image ---
image = Image.open("data/shepp.png").convert("L")
N = 128
image = image.resize((N, N))
image_array = np.array(image, dtype=np.float32) / 255.0

# --- 2. Define Geometry ---
vol_geom = astra.create_vol_geom(N, N)
angles = np.linspace(0, np.pi, 32, endpoint=False)
# angles = np.linspace(0, np.pi / 2, 64, endpoint=False)
proj_geom = astra.create_proj_geom("parallel", 1.0, N, angles)

# 3. Create Projector
projector_id = astra.create_projector("linear", proj_geom, vol_geom)

# 4. Create Sinogram
proj_id, sinogram = astra.create_sino(image_array, projector_id)

# 5. Data Objects
sinogram_id = astra.data2d.create("-sino", proj_geom, sinogram)
recon_id = astra.data2d.create("-vol", vol_geom)

# 6. FBP Config
cfg = astra.astra_dict("FBP")
cfg["ReconstructionDataId"] = recon_id
cfg["ProjectionDataId"] = sinogram_id
cfg["ProjectorId"] = projector_id  # ← REQUIRED
cfg["option"] = {"FilterType": "Ram-Lak"}

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# After use
astra.projector.delete(projector_id)

# --- 7. Retrieve and Display Reconstruction ---
reconstruction = astra.data2d.get(recon_id)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_array, cmap="gray", interpolation="nearest")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Sinogram")

display_sinogram = np.zeros((sinogram.shape[0] * 2, *sinogram.shape[1:]))
display_sinogram[::2] = sinogram

plt.imshow(display_sinogram, cmap="gray", aspect="auto", interpolation="nearest")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Reconstruction (FBP)")
plt.imshow(reconstruction, cmap="gray", interpolation="nearest")
plt.axis("off")

plt.tight_layout()
plt.show()

# --- 8. Clean up ---
astra.algorithm.delete(alg_id)
astra.data2d.delete(recon_id)
astra.data2d.delete(sinogram_id)
# %% Limited
image = Image.open("data/shepp.png").convert("L")
N = 128
image = image.resize((N, N))
image_array = np.array(image, dtype=np.float32) / 255.0

# --- 2. Define Geometry ---
vol_geom = astra.create_vol_geom(N, N)
m = 64  # Total number of angles
lhs = np.linspace(0, np.pi / 4, m // 4, endpoint=False)
rhs = np.linspace(3 * (np.pi / 4), np.pi, m // 4, endpoint=False)
angles = np.concatenate((lhs, rhs))
proj_geom = astra.create_proj_geom("parallel", 1.0, N, angles)

# 3. Create Projector
projector_id = astra.create_projector("linear", proj_geom, vol_geom)

# 4. Create Sinogram
proj_id, sinogram = astra.create_sino(image_array, projector_id)

# 5. Data Objects
sinogram_id = astra.data2d.create("-sino", proj_geom, sinogram)
recon_id = astra.data2d.create("-vol", vol_geom)

# 6. FBP Config
cfg = astra.astra_dict("FBP")
cfg["ReconstructionDataId"] = recon_id
cfg["ProjectionDataId"] = sinogram_id
cfg["ProjectorId"] = projector_id  # ← REQUIRED
cfg["option"] = {"FilterType": "Ram-Lak"}

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

# After use
astra.projector.delete(projector_id)

# --- 7. Retrieve and Display Reconstruction ---
reconstruction = astra.data2d.get(recon_id)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_array, cmap="gray", interpolation="nearest")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Sinogram")

display_sinogram = np.zeros((m, *sinogram.shape[1:]))
display_sinogram[: m // 4] = sinogram[: m // 4]
display_sinogram[3 * (m // 4) :] = sinogram[m // 4 :]

plt.imshow(display_sinogram, cmap="gray", aspect="auto", interpolation="nearest")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Reconstruction (FBP)")
plt.imshow(reconstruction, cmap="gray", interpolation="nearest")
plt.axis("off")

plt.tight_layout()
plt.show()

# --- 8. Clean up ---
astra.algorithm.delete(alg_id)
astra.data2d.delete(recon_id)
astra.data2d.delete(sinogram_id)

# %%
# Parameters
r = 1  # Sphere radius
num_points = 1000  # Number of desired scanner positions
num_camera_circle = 50  # Number of camera vectors in equatorial circle

# Generate uniformly distributed points on a sphere using the Fibonacci lattice
indices = np.arange(0, num_points, dtype=float) + 0.5
phi = np.arccos(1 - 2 * indices / num_points)
theta = np.pi * (1 + 5**0.5) * indices

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# Camera circle on equator
theta_cam = np.linspace(0, np.pi, num_camera_circle, endpoint=False)
x_cam = r * np.cos(theta_cam)
y_cam = r * np.sin(theta_cam)
z_cam = np.zeros_like(x_cam)

# Vectors pointing to origin
u = -x_cam
v = -y_cam
w = -z_cam

# %%
# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, color="blue", s=10, label="Potential Scanner Position", alpha=0.4)
ax.quiver(
    x_cam,
    y_cam,
    z_cam,
    u,
    v,
    w,
    length=0.3,
    color="red",
    normalize=True,
    label="Potential Scanner Pose",
)
# Add a marker 2 to represent the object at origin
ax.scatter(0, 0, 0, color="red", s=100, marker="x")


ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Scanner geometry and poses visualized")
ax.legend()
ax.set_box_aspect([1, 1, 1])  #

plt.tight_layout()
plt.show()
