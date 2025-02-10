import ast
import torch
import numpy as np
from pathlib import Path
from run_nerf import create_nerf

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


def generate_voxel_grid(model, grid_size, bbox_min, bbox_max, device):
    """
    Generates a voxel grid from a trained NeRF model.

    Args:
        model: Trained NeRF model.
        grid_size (tuple): (H, W, D) dimensions of the voxel grid.
        bbox_min (tuple): Minimum XYZ bounds of the volume.
        bbox_max (tuple): Maximum XYZ bounds of the volume.
        device: Torch device (CPU/GPU).

    Returns:
        voxel_grid: (H, W, D, 4) array containing (r, g, b, sigma) values.
    """
    H, W, D = grid_size
    x_lin = torch.linspace(bbox_min[0], bbox_max[0], H, device=device)
    y_lin = torch.linspace(bbox_min[1], bbox_max[1], W, device=device)
    z_lin = torch.linspace(bbox_min[2], bbox_max[2], D, device=device)

    # Create a 3D grid of points
    x, y, z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
    points = torch.stack([x, y, z], dim=-1).view(-1, 3)  # Reshape to (H*W*D, 3)

    # Query NeRF model (no view directions needed for static scene)
    with torch.no_grad():
        raw_output = model(points)
        rgb = torch.sigmoid(raw_output[..., :3])  # (H*W*D, 3)
        sigma = raw_output[..., 3:4]  # (H*W*D, 1)
        voxel_values = torch.cat([rgb, sigma], dim=-1)  # (H*W*D, 4)

    return voxel_values.view(H, W, D, 4).cpu().numpy()  # Reshape back to 3D


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path("./logs/blender_paper_lego")

    # Load trained NeRF model
    args = parse_args(base_path / "args.txt")
    render_kwargs_train, render_kwargs_test, _, _, _ = create_nerf(args)
    model = render_kwargs_train["network_fn"].to(device)
    model.eval()

    # Define voxel grid properties
    grid_size = (128, 128, 128)  # Adjust as needed
    bbox_min = (-1.0, -1.0, -1.0)  # Adjust bounds based on dataset
    bbox_max = (1.0, 1.0, 1.0)

    # Generate voxel grid
    # i dont what the fuck is worng with this line but this function needs to be fucking fixed lol aahaaha 
    voxel_grid = generate_voxel_grid(model, grid_size, bbox_min, bbox_max, device)

    # Save to file
    save_path = base_path / "voxel_grid.npz"
    np.savez_compressed(save_path, voxel_grid=voxel_grid)
    print(f"Voxel grid saved at {save_path}")

