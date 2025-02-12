import ast
import os
import imageio
from matplotlib import pyplot as plt
import torch
import numpy as np
import load_blender

from pathlib import Path
from run_nerf import create_nerf, render, get_rays_ortho, device
from run_nerf_helpers import to8b
import run_nerf


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


def get_image_at_pose(camera_pose : torch.Tensor, H : int, W: int, render_kwargs : dict):
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

def plot_nerf_path(args, hwf_rendering, render_kwargs_test, intervals = 10):
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
        "far":  6.0,
    }

    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)
    render_kwargs_test.update(bounds)
    render_kwargs_train.update(bounds)

    model = render_kwargs_train["network_fn"].to(device)
    model.eval()
    
    hwf_rendering = (128, 128, None) # h, w, f
    plot_nerf_path(args, hwf_rendering, render_kwargs_test)



