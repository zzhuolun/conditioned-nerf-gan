# --------------------------------------------------------------------
# Reproject the depth map and rgb to colored points in world space.
# Render the reprojected points with pcl, to check if the pose matches.
# --------------------------------------------------------------------
from os import RTLD_LAZY
import sys
sys.path.append('../')
from datasets import _read_resize_shapenet, readEXR
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
import cv2 as cv
import OpenEXR
import argparse


def load_cam2world_gt(fpath, id):
    cams = np.load(fpath / "cameras.npz")
    cam2world_gt = cams[f"world_mat_inv_{int(id)}"]
    # K_inv = cams[f"camera_mat_inv_{int(id)}"]
    return cam2world_gt


def backproject_points(
    fpath: Path, id: int, img_size: int, show: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject depth and rgb image to world space

    Args:
        fpath (Path): path to the car data folder
        id (int): the idx of the view.
        img_size (int): the size to load depth and rgb image.
        show (bool): if show the depth and rgb image of each view.

    Returns:
        tuple[np.ndarray, np.ndarray]: points coordinates in world space and the corresponding color of each point.
    """
    depth = readEXR(str(fpath / "depth" / f"{id:04d}0001.exr"), img_size)
    rgb = _read_resize_shapenet(str(fpath / "image" / f"{id:04d}.png"), img_size)
    K = np.load(fpath / "cameras.npz")[f"camera_mat_{id}"]
    if show:
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        axes[0].imshow(depth)
        axes[1].imshow(rgb)
        plt.show()
    cam2world = load_cam2world_gt(fpath, id)
    depth_pts = []
    color_pts = []
    H, W = depth.shape
    for i in range(H):
        for j in range(W):
            if depth[i, j] > 1e-3:
                Z = depth[i, j].item()
                Y = (2 * i - (H - 1)) / (H - 1) * Z / K[1, 1]
                X = (2 * j - (W - 1)) / (W - 1) * Z / K[0, 0]
                depth_pts.append([X, Y, Z])
                color_pts.append(rgb[i, j])
    depth_pts = np.asarray(depth_pts)
    color_pts = np.asarray(color_pts)
    depth_pts_homo = np.ones((depth_pts.shape[0], 4))
    depth_pts_homo[:, :3] = depth_pts
    depth_pts_w = (depth_pts_homo @ cam2world.T)[:, :3]
    return depth_pts_w, color_pts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Renders given obj file by rotation a camera around it."
    )
    parser.add_argument(
        "path",
        type=str,
        help="the absolute path of the model directory which has the cameras.npz, image/, depth/, etc.",
    )
    parser.add_argument(
        "-v", "--views", type=int, default=4, help="number of views to be rendered"
    )
    parser.add_argument(
        "-s", "--img_size", type=int, default=256, help="number of views to be rendered"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If true, show the depth and rgb image of each view.",
    )
    opt = parser.parse_args()

    DIR = Path(opt.path)

    depths = []
    colors = []
    for i in random.sample(range(opt.views), opt.views):
        depth_pts, color_pts = backproject_points(DIR, i, opt.img_size, opt.show)
        depths.append(depth_pts)
        colors.append(color_pts)
    depths = np.concatenate(depths, 0)
    colors = np.concatenate(colors, 0)

    print("start visualizing")
    pt_depth = o3d.geometry.PointCloud()
    pt_depth.points = o3d.utility.Vector3dVector(depths)
    pt_depth.colors = o3d.utility.Vector3dVector(colors)

    # visualize gt pcl
    pcl_gt = np.load(DIR / "pointcloud.npz")["points"]
    pcl_render = o3d.geometry.PointCloud()
    pcl_render.points = o3d.utility.Vector3dVector(pcl_gt)
    pcl_render.paint_uniform_color([0, 0.5, 0.8])

    o3d.visualization.draw_geometries([pt_depth, pcl_render], width=1600, height=750)
