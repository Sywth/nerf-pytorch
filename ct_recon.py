# %%
import numpy as np
import matplotlib.pyplot as plt
import ct_scan
import astra

if __name__ == "__main__":
    # %%
    phantom_idx = 13
    size = 128
    num_scans = 32

    # %%
    phantom = ct_scan.load_phantom(phantom_idx, size)
    scan_params = ct_scan.AstraScanParameters(
        phantom_shape=phantom.shape,
        num_scans=num_scans,
    )
    ct_imgs = scan_params.generate_ct_imgs(phantom)
    sinogram = ct_imgs.swapaxes(0, 1)
    ct_recon = scan_params.reconstruct_3d_volume_alg(sinogram, 64)

    # %%
    show_idx = (size - 1) // 2

    plt.subplot(1, 2, 1)
    plt.title(f"CT Image {show_idx}")
    plt.imshow(phantom[show_idx])
    plt.show()

    plt.subplot(1, 2, 2)
    plt.title("Sinogram")
    plt.imshow(ct_recon[show_idx])
    plt.show()

    # %%

