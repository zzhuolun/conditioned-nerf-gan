import torch.nn as nn
import torch
import sys

sys.path.append("../")
from generators.volumetric_rendering import *
import numpy as np
from datasets import ShapeNet
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm


def unique_idx(lst):
    result = []
    seen = set()
    for idx, x in enumerate(lst):
        if x not in seen:
            result.append(idx)
            seen.add(x)
    return result


def voxel_interpolate(voxels, points, img_size, num_steps, device):
    voxel_rgb_sigma = torch.zeros_like(voxels).to(device)
    voxel_rgb_sigma[:, :3] = voxels[:, 1:]
    voxel_rgb_sigma[:, -1] = voxels[:, 0]

    voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
    batch_size = points.shape[0]
    grid = points / (voxel_length / 2)  # normalize the sampling point location
    grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)
    points_feature = F.grid_sample(
        voxel_rgb_sigma,
        grid,
        mode="nearest",
        align_corners=False,
        padding_mode="border",
    )
    sigma = points_feature[:, -1]
    color = points_feature[:, :-1]
    b, h, w, n = torch.where(sigma != 0)
    bhw = torch.stack([b, h, w], 1).tolist()
    bhw = [tuple(i) for i in bhw]
    bhwn = torch.stack([b, h, w, n], 1).tolist()
    bhw_s = unique_idx(bhw)
    bhwn_s = [bhwn[i] for i in bhw_s]
    bhwn_s = torch.as_tensor(bhwn_s, dtype=torch.long)
    color_render = torch.ones(batch_size, 3, img_size, img_size).to(device)
    color_render[bhwn_s[:, 0], :, bhwn_s[:, 1], bhwn_s[:, 2]] = color[
        bhwn_s[:, 0], :, bhwn_s[:, 1], bhwn_s[:, 2], bhwn_s[:, 3]
    ]
    return color_render

def voxel_surface_render(
    voxels,
    cam2worlds,
    img_size,
    fov,
    ray_start,
    ray_end,
    num_steps,
    device,
):

    batch_size = cam2worlds.shape[0]
    # Generate initial camera rays and sample points.
    with torch.no_grad():
        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
            batch_size,
            num_steps,
            resolution=(img_size, img_size),
            device=device,
            fov=fov,
            ray_start=ray_start,
            ray_end=ray_end,
        )  # batch_size, pixels, num_steps, 1
        (
            transformed_points,
            z_vals,
            transformed_ray_directions,
            transformed_ray_origins,
        ) = transform_sampled_points(
            points_cam,
            z_vals,
            rays_d_cam,
            device=device,
            cam2worlds=cam2worlds,
        )
        transformed_points = transformed_points.reshape(
            batch_size, img_size * img_size * num_steps, 3
        )
    # Model prediction on course points
    rgb = voxel_interpolate(
        voxels,
        transformed_points,
        img_size,
        num_steps,
        device,
    )
    return rgb

if __name__=="__main__":
    for category in ["Car", "Chair", "Plane"]:
        metadata = {
            "dataset_path": f"/storage/user/zhouzh/data/ShapeNet{category}/ShapeNet{category}",
            "target_size": 128,
            "subset": "val",
            "load_voxel": True,
            "load_pcl": False,
            "load_img": False,
            "load_depth": False,
            "load_cam": True,
            "voxel_resolution": 32,
            "batch_size": 1,
            "num_workers": 12,
            "fov": 49.134342641202636,
        }

        dataset = ShapeNet(
            metadata["dataset_path"],
            metadata["target_size"],
            metadata["subset"],
            metadata=metadata,
            load_voxel=metadata["load_voxel"],
            load_pcl=metadata["load_pcl"],
            dataset_mode="test",
            load_img=True,
            load_depth=False,
            load_cam=True,
            voxel_resolution=metadata["voxel_resolution"],
            shuffle=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=metadata["batch_size"],
            shuffle=False,
            drop_last=True,
            pin_memory=False,
            num_workers=metadata["num_workers"],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save_dir = Path(f"/storage/slurm/zhouzh/thesis/voxel_render_32/{category}")
        save_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            for data in tqdm(dataloader):
                rgbs = voxel_surface_render(
                    data["voxel"].to(device),
                    data["cam2world"].to(device),
                    metadata["target_size"],
                    metadata["fov"],
                    ray_start=0.2,
                    ray_end=2.0,
                    num_steps=256,
                    device=device,
                )
                # output = torch.cat([data["img"] * 0.5 + 0.5, rgb.cpu()], 0)
                rgbs = rgbs.cpu()
                for idx, rgb in enumerate(rgbs):
                    img_dir = Path(data["img_path"][idx])
                    save_image(
                        rgb,
                        save_dir / (img_dir.parent.parent.stem + "_" + img_dir.name),
                        nrow=4,
                        normalize=False,
                    )
