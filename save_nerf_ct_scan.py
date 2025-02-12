import ast
import torch
import numpy as np
import load_blender

from pathlib import Path
from run_nerf import create_nerf, render, get_rays_ortho, device


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

    # TODO : Figure out how to handle device placement cause idek what the actual fuck the original nerf script does 
    rays = rays.to(device)
    camera_pose = camera_pose.to(device)

    print("Rays device:", rays.device)
    print("Camera pose device:", camera_pose.device)

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

    return np.clip(rgb_image, 0, 1)  # Ensure valid image range


if __name__ == "__main__":
    torch.cuda.empty_cache()

    base_path = Path("./logs/blender_paper_lego")

    # Load trained NeRF model
    args = parse_args(base_path / "args.txt")
    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)
    model = render_kwargs_train["network_fn"].to(device)
    model.eval()
    H = render_kwargs_test

    render_resolution = 256, 256
    camera_pose = load_blender.pose_spherical(
        theta=0.0,
        phi=0.0,
        radius=2.0,
    )

    get_image_at_pose(
        camera_pose,
        render_resolution[0],
        render_resolution[1],
        render_kwargs_test,
    )
