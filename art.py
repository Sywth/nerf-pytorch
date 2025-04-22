# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import ct_scan

# %%
phantom_idx = 13
ph_size = 256
img_res = ph_size
num_scans = 64
theta_range = [0, np.pi]
phi_range = [0, 0]
spherical_angles = np.stack(
    (
        np.linspace(theta_range[0], theta_range[1], num_scans),
        np.linspace(phi_range[0], phi_range[1], num_scans),
    ),
    axis=1,
)

phantom = ct_scan.load_phantom(phantom_idx, ph_size)
scan_2n = ct_scan.AstraScanVec3D(phantom.shape, spherical_angles, img_res=img_res)
ct_imgs = scan_2n.generate_ct_imgs(phantom)


# %%
def save_phantom(phantom_idx, ph_size=ph_size):
    phantom = ct_scan.load_phantom(phantom_idx, ph_size)
    np.save(f"phantoms/phantom_{phantom_idx}.npy", phantom)


for idx in [4, 13, 16]:
    save_phantom(idx)

# %%
plot_images_n = 8

fig, axs = plt.subplots(1, plot_images_n, figsize=(10, 5))
for idx, i in enumerate(np.arange(0, num_scans, num_scans // plot_images_n)):
    axs[idx].imshow(ct_imgs[int(i)], cmap="gray")
    axs[idx].axis("off")
plt.show()


# %% My current library using SIRT
ct_recon = scan_2n.reconstruct_3d_volume_sirt(ct_imgs)
fig, axs = plt.subplots(1, plot_images_n, figsize=(10, 5))
for idx, i in enumerate(np.arange(0, ph_size, ph_size // plot_images_n)):
    axs[idx].imshow(ct_recon[int(i), :, :], cmap="gray")
    axs[idx].axis("off")
plt.show()
