import torch
import matplotlib.pyplot as plt
import matplotlib


def plot_rays(ray_o: torch.Tensor, ray_d: torch.Tensor, scan_lim: float = 1.0):
    ray_o = ray_o.cpu().numpy()
    ray_d = ray_d.cpu().numpy()

    ax = plt.figure().add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.quiver(
        ray_o[..., 0],
        ray_o[..., 1],
        ray_o[..., 2],
        ray_d[..., 0],
        ray_d[..., 1],
        ray_d[..., 2],
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-scan_lim, scan_lim)
    ax.set_ylim(-scan_lim, scan_lim)
    ax.set_zlim(-scan_lim, scan_lim)

    plt.show()
