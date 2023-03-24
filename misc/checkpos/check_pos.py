from os import RTLD_LAZY
import sys

sys.path.append("../")
import numpy as np
import random
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from generators.math_utils_torch import *
from generators.volumetric_rendering import create_cam2world_matrix
import math
import open3d as o3d
from pathlib import Path
import pickle
from misc.tsne import custom_color
import argparse


def load_cam2world_gt(id):
    dir = Path(
        "/usr/stud/zhouzh/stanford-shapenet-renderer/shapenet/baa1e44e6e086c233e320a6703976361"
    )
    cams = np.load(dir / "cameras.npz")
    cam2world_gt = cams[f"world_mat_inv_{int(id)}"]
    K_inv = cams[f"camera_mat_inv_{int(id)}"]
    cam2world_gt[:,1]*=-1
    cam2world_gt[:,2]*=-1
    return cam2world_gt, K_inv


def load_pcl():
    dir = Path("/usr/stud/zhouzh/data/ShapeNetCar/baa1e44e6e086c233e320a6703976361/")
    # dir = Path("/usr/stud/zhouzh/stanford-shapenet-renderer/shapenet/baa1e44e6e086c233e320a6703976361/")
    pcl_npz = np.load(dir / "pointcloud.npz")
    pcl = pcl_npz["points"]
    scale = pcl_npz["scale"]
    pcl = pcl[torch.randperm(len(pcl)).numpy()[:2500]]
    return pcl


def check_ray_direction(depths_pts, rays, coords, cam2world, H, W, show=True):
    cam_origin = cam2world[:3, -1]
    rays = rays[:, 0, :].reshape(128, 128, 3)[coords[:, 0], coords[:, 1]] - cam_origin
    depths_pts = depths_pts - cam_origin
    rays /= np.linalg.norm(
        rays, axis=-1, keepdims=True
    )  # nerf generated ray direction at world space
    depths_pts /= np.linalg.norm(
        depths_pts, axis=-1, keepdims=True
    )  # depth unprojected ray direction at world space
    if show:
        ray_direction_map = np.zeros((H, W, 3))
        depthpts_direction_map = np.zeros((H, W, 3))
        ray_direction_map[coords[:, 0], coords[:, 1]] = rays
        depthpts_direction_map[coords[:, 0], coords[:, 1]] = depths_pts
        fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        mini = min(np.min(ray_direction_map), np.min(depthpts_direction_map))
        maxi = max(np.max(ray_direction_map), np.max(depthpts_direction_map))
        ray_direction_map = (ray_direction_map - mini) / (maxi - mini)
        depthpts_direction_map = (depthpts_direction_map - mini) / (maxi - mini)
        difference = np.sum(np.abs(ray_direction_map - depthpts_direction_map), axis=-1)
        axes[0].imshow(ray_direction_map)
        axes[1].imshow(depthpts_direction_map)
        im = axes[2].imshow(difference)
        # im = axes[1][0].imshow(ray_direction_map[:,:,0])
        # im = axes[1][1].imshow(ray_direction_map[:,:,1])
        # im = axes[1][2].imshow(ray_direction_map[:,:,2])

        # im = axes[2][0].imshow(depthpts_direction_map[:,:,0])
        # im = axes[2][1].imshow(depthpts_direction_map[:,:,1])
        # im = axes[2][2].imshow(depthpts_direction_map[:,:,2])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
    assert np.allclose(rays, depths_pts, atol=0.005, rtol=0)
    # for i in range(depths_pts.shape[0]):
    #     depth_pt = depths_pts[i] - cam_origin
    #     ray_pt = rays[coords[i]] - cam_origin
    #     depth_pt /= np.linalg.norm(depth_pt)
    #     ray_pt /= np.linalg.norm(ray_pt)
    #     if not np.allclose(depth_pt, ray_pt, atol=0.005, rtol=0):
    #         print(depth_pt, ray_pt)


def get_points(depth_dir, rgb_dir, cam2world_dir, rays_dir):
    depth = torch.load(depth_dir)[0]
    rgb = torch.load(rgb_dir)[0].permute(1, 2, 0)
    cam2world = torch.load(cam2world_dir).detach().cpu().numpy()[0]
    rays = torch.load(rays_dir).detach().cpu().numpy()[0]
    depth_pts = []
    color_pts = []
    H, W = depth.shape
    coords = []
    for x in range(H):
        for y in range(W):
            if depth[x, y] != 0:
                coords.append([x, y])
                z = depth[x, y].item()
                v = -(2 * x - (H - 1)) / (H - 1) * z / 2.1875
                u = (2 * y - (W - 1)) / (W - 1) * z / 2.1875
                depth_pts.append([u, v, -z])
                color_pts.append(rgb[x, y].numpy())
    depth_pts = np.array(depth_pts)
    coords = np.asarray(coords)
    assert depth_pts[:, 0].max() <= 1
    assert depth_pts[:, 1].max() <= 1
    assert depth_pts[:, 0].min() >= -1
    assert depth_pts[:, 1].min() >= -1
    depth_pts_homo = np.ones((depth_pts.shape[0], 4))
    depth_pts_homo[:, :3] = depth_pts
    depths_pts_w = (cam2world @ depth_pts_homo.T).T[:, :3]
    color_pts = np.asarray(color_pts)
    check_ray_direction(depths_pts_w, rays, coords, cam2world, H, W, show=False)
    rays = rays.reshape(-1, 3)
    rays = rays[torch.randperm(rays.shape[0]).numpy()[: int(rays.shape[0] * 0.05)]]
    return depths_pts_w, color_pts, rays, cam2world


def img2cam(depth, K_inv):
    img_coord = np.ones_like(depth)
    img_coord[:, :2] = depth[:, :2]  # Nx3
    xyz = img_coord @ K_inv.T
    xyz *= depth[:, -1:]
    return xyz


def get_points_gt(depth_dir, rgb_dir, rays_dir, id):
    depth = torch.load(depth_dir)[0]
    rgb = torch.load(rgb_dir)[0].permute(1, 2, 0)
    cam2world , _ = load_cam2world_gt(id)
    rays = torch.load(rays_dir).detach().cpu().numpy()[0]
    depth_pts = []
    color_pts = []
    H, W = depth.shape
    coords = []
    for x in range(H):
        for y in range(W):
            if depth[x, y] != 0:
                coords.append([x, y])
                z = depth[x, y].item()
                v = (2 * x - (H - 1)) / (H - 1) * z / 2.1875
                u = (2 * y - (W - 1)) / (W - 1) * z / 2.1875
                depth_pts.append([u, v, z])
                color_pts.append(rgb[x, y].numpy())
    depth_pts = np.array(depth_pts)
    coords = np.asarray(coords)
    assert depth_pts[:, 0].max() <= 1
    assert depth_pts[:, 1].max() <= 1
    assert depth_pts[:, 0].min() >= -1
    assert depth_pts[:, 1].min() >= -1
    depth_pts_homo = np.ones((depth_pts.shape[0], 4))
    depth_pts_homo[:, :3] = depth_pts
    depths_pts_w = (cam2world @ depth_pts_homo.T).T[:, :3]
    color_pts = np.asarray(color_pts)
    check_ray_direction(depths_pts_w, rays, coords, cam2world, H, W, show=False)
    rays = rays.reshape(-1, 3)
    rays = rays[torch.randperm(rays.shape[0]).numpy()[: int(rays.shape[0] * 0.05)]]
    return depths_pts_w, color_pts, rays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Renders given obj file by rotation a camera around it."
    )
    parser.add_argument(
        "-v", "--views", type=int, default=4, help="number of views to be rendered"
    )
    opt = parser.parse_args()

    # tableau20, tableau20blind = custom_color()
    depths = []
    colors = []
    rays = []
    # rays_color = []
    # depths_gts = []
    # start = random.randint(0, 100-NUM_VIEWS)
    for i in random.sample(range(120), opt.views):
        if i==106:
            continue
        print(i)
        depths_pts_w, color_pts, ray, cam2world = get_points(
            f"data/depths_128_{i}", f"data/rgb_128_{i}", f"data/cam2world_128_{i}", f"data/ray_128_{i}"
        )
        cam2world_gt, _ = load_cam2world_gt(i)
        assert np.allclose(cam2world_gt, cam2world, atol=0.005, rtol=0)
        # depths_pts_w, color_pts, ray = get_points_gt(
        #     f"data/depths_128_{i}",
        #     f"data/rgb_128_{i}",
        #     f"data/ray_128_{i}",
        #     i,
        # )
        depths.append(depths_pts_w)
        colors.append(color_pts)
        rays.append(ray)
        # rays_color.append(np.ones_like(ray) * np.asarray(tableau20[i - start]))
        # depths_gts.append(depths_gt)
        # cam2world_gt = load_cam2world_gt(paths[i])
        # # assert np.allclose(cam2world, cam2world_gt)
        # assert np.sum(np.abs(cam2world-cam2world_gt))<0.01

    # pcl = load_pcl()

    depths = np.concatenate(depths, 0)
    colors = np.concatenate(colors, 0)
    rays = np.concatenate(rays, 0)
    # rays_color = np.concatenate(rays_color, 0)
    # depths_gts = np.concatenate(depths_gts, 0)
    print("start visualizing")
    pt_depth = o3d.geometry.PointCloud()
    pt_depth.points = o3d.utility.Vector3dVector(depths)
    # pt_depth.paint_uniform_color([0, 0, 1])
    pt_depth.colors = o3d.utility.Vector3dVector(colors)

    # pt_depth_gt = o3d.geometry.PointCloud()
    # pt_depth_gt.points = o3d.utility.Vector3dVector(depths_gts)
    # # pt_depth.paint_uniform_color([0, 0, 1])
    # pt_depth_gt.colors = o3d.utility.Vector3dVector(colors)

    pt_render = o3d.geometry.PointCloud()
    pt_render.points = o3d.utility.Vector3dVector(rays)
    pt_render.paint_uniform_color([0, 0.5, 0.8])
    # pt_render.colors = o3d.utility.Vector3dVector(rays_color)

    # pt_pcl = o3d.geometry.PointCloud()
    # pt_pcl.points = o3d.utility.Vector3dVector(pcl)
    # pt_pcl.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pt_depth, pt_render], width=1800, height=950)
