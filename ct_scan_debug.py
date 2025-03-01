
import trimesh
import pyrender
import numpy as np

def create_axis_arrow(axis: str, length=1.0, radius=0.01):
    """Creates an arrow mesh for a given axis."""
    # Cylinder for the arrow shaft
    shaft = trimesh.creation.cylinder(radius=radius, height=length * 0.8)
    shaft.apply_translation([0, 0, length * 0.4])  # Move it up along Z

    # Cone for the arrowhead
    head = trimesh.creation.cone(radius=radius * 2, height=length * 0.2)
    head.apply_translation([0, 0, length * 0.9])  # Place on top of shaft

    # Combine shaft and head
    arrow = trimesh.util.concatenate([shaft, head])

    # Rotate arrows to correct axis
    if axis == "x":
        arrow.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        arrow.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    return arrow

def add_cardinal_axes(scene, axis_length=1.0):
    """Adds cardinal axes (X, Y, Z) to the pyrender scene."""

    # Create arrows with different colors
    x_axis = pyrender.Mesh.from_trimesh(create_axis_arrow("x", axis_length), smooth=False)
    y_axis = pyrender.Mesh.from_trimesh(create_axis_arrow("y", axis_length), smooth=False)
    z_axis = pyrender.Mesh.from_trimesh(create_axis_arrow("z", axis_length), smooth=False)

    # Add to scene with colors
    scene.add(x_axis, pose=np.eye(4), name="X-axis")
    scene.add(y_axis, pose=np.eye(4), name="Y-axis")
    scene.add(z_axis, pose=np.eye(4), name="Z-axis")
