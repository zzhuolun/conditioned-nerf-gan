import sys

sys.path.append("../")
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import OpenEXR
import torch

car_dir = "/usr/stud/zhouzh/data/ShapeNetCar/baa1e44e6e086c233e320a6703976361/"


def load_cam2world_gt(id):
    cams = np.load(Path(car_dir) / "cameras.npz")
    cam2world_gt = cams[f"world_mat_inv_{int(id)}"]
    K_inv = cams[f"camera_mat_inv_{int(id)}"]
    return cam2world_gt, K_inv


def load_pcl():
    pcl_npz = np.load(Path(car_dir) / "pointcloud.npz")
    pcl = pcl_npz["points"]
    scale = pcl_npz["scale"]
    pcl = pcl[torch.randperm(100000).numpy()[:50000]]
    return pcl


def read_depth_rgb(id) -> np.ndarray:
    """Read depth map from the .exr file

    Args:
        fpath (str): the path to the .exr depth map

    Returns:
        np.ndarray: 256*256, depth map
    """
    fpath = str(Path(car_dir) / "depth" / f"{id:04}0001.exr")
    readFile = OpenEXR.InputFile(fpath)
    (r, g, b) = readFile.channels("RGB")
    dr = np.copy(np.frombuffer(r, dtype=np.half))
    db = np.copy(np.frombuffer(b, dtype=np.half))
    dg = np.copy(np.frombuffer(g, dtype=np.half))
    assert np.allclose(dr, db)
    assert np.allclose(db, dg)
    h = (
        readFile.header()["displayWindow"].max.y
        + 1
        - readFile.header()["displayWindow"].min.y
    )
    w = (
        readFile.header()["displayWindow"].max.x
        + 1
        - readFile.header()["displayWindow"].min.x
    )
    depth = np.reshape(dr, (h, w)).astype(np.float64)
    depth[depth == np.inf] = 0

    rgb_path = Path(car_dir) / "image" / f"{id:04}.png"
    rgb = cv.imread(str(rgb_path), cv.IMREAD_UNCHANGED)
    rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
    rgb = (rgb / 256).astype("uint8") / 256
    assert rgb.shape[0] == depth.shape[0]
    assert rgb.shape[1] == depth.shape[1]
    assert rgb.shape[2] == 3
    fig = plt.figure(id)
    ax = fig.add_subplot(121)
    ax.imshow(rgb)
    ax = fig.add_subplot(122)
    ax.imshow(depth)
    return depth, rgb


def reproject(id):
    """Reproject the depth and rgb image to world coordinate system."""
    cam2world, K_inv = load_cam2world_gt(id)
    depth, rgb = read_depth_rgb(id)
    H, W = depth.shape
    pts = []
    pts_color = []
    for x in range(H):
        for y in range(W):
            if depth[x, y] != 0:
                z = depth[x, y]
                v = (2 * x - (H - 1)) / (H - 1) * z / 2.1875
                u = (2 * y - (W - 1)) / (W - 1) * z / 2.1875
                pts.append([u, v, z])
                pts_color.append(rgb[x, y])
    pts = np.array(pts)
    assert pts[:, 0].max() <= 1
    assert pts[:, 1].max() <= 1
    assert pts[:, 0].min() >= -1
    assert pts[:, 1].min() >= -1
    pts_color = np.array(pts_color)
    pts_homo = np.ones((pts.shape[0], 4))
    pts_homo[:, :-1] = pts
    pts_w = (cam2world @ pts_homo.T).T[:, :3]
    return pts_w, pts_color


def visualize(ids):
    pts = []
    pts_color = []
    for i in ids:
        pt, pt_color = reproject(i)
        pts.append(pt)
        pts_color.append(pt_color)
    # plt.show()
    pts = np.concatenate(pts, 0)
    pts_color = np.concatenate(pts_color, 0)
    pcl = load_pcl()

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pts)
    # pt_depth.paint_uniform_color([0, 0, 1])
    points.colors = o3d.utility.Vector3dVector(pts_color)

    pt_pcl = o3d.geometry.PointCloud()
    pt_pcl.points = o3d.utility.Vector3dVector(pcl)
    pt_pcl.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([points, pt_pcl], width=1800, height=950)


if __name__ == "__main__":
    ids = list(range(24))
    visualize(ids)
