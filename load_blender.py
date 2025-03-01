import os
from warnings import warn
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from pathlib import Path, PureWindowsPath

import utils

trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()

# NOTE : Consider copying this 
def pose_spherical_deg(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (
        torch.Tensor(
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        )
        @ c2w
    )
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            file_path = PureWindowsPath(basedir) / PureWindowsPath(frame["file_path"])
            if file_path.suffix != ".png":
                file_path = file_path.with_suffix(".png")
            fname = file_path.as_posix()

            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))

        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # NOTE : Added so that i can empty train and val splits
    #   if train or val has len 0 then give them shape (0, *all_imgs[0].shape)
    all_imgs = [
        np.empty((0, *(all_imgs[0].shape[1:]))) if len(imgs) == 0 else imgs
        for imgs in all_imgs
    ]
    all_poses = [
        np.empty((0, *(all_poses[0].shape[1:]))) if len(poses) == 0 else poses
        for poses in all_poses
    ]
    # END of my addition

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]

    camera_angle_x = float(meta["camera_angle_x"])
    print(f"Using Camera X : {camera_angle_x}")

    ortho_mode = False 
    # check if nan or inf
    if np.isnan(camera_angle_x) or np.isinf(camera_angle_x):
        warn(
            "Camera angle is nan or inf. Assuming use of flag --use_ortho"
        )
        ortho_mode = True
    
    if not ortho_mode:
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

# NOTE : This is the original code, but i replaced it to work with my ct-rendering code 
    # render_poses = torch.stack(
    #     [
    #         pose_spherical(angle, -30.0, 4.0)
    #         for angle in np.linspace(-180, 180, 40 + 1)[:-1]
    #     ],
    #     0,
    # )

    number_of_angles = 8
    angles_deg = np.linspace(-180, 180, number_of_angles, endpoint=False)
    render_poses = utils.generate_camera_poses(
        np.deg2rad(angles_deg), 
        4.0,
        "x",
    )
    # cast each pose to a torch and stack them
    render_poses = torch.stack(
        [torch.Tensor(pose) for pose in render_poses],
        0,
    )

    # Translate each pose by trans 
    # translation = torch.Tensor([0, 0.5, 0])
    # render_poses[:, :3, 3] += translation
    

    if half_res:
        H = H // 2
        W = W // 2
        if not ortho_mode:
            focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    if ortho_mode:
        return imgs, poses, render_poses, [H,W], i_split
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split
    
    raise Exception("Unreachable")
