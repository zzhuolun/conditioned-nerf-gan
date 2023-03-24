import numpy as np
import sys
sys.path.append('../')
import random
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from generators.math_utils_torch import *
from generators.volumetric_rendering import create_cam2world_matrix
import math
import open3d as o3d
import cv2 as cv
import OpenEXR
def read_depth_rgb(id) -> np.ndarray:
    """Read depth map from the .exr file

    Args:
        fpath (str): the path to the .exr depth map

    Returns:
        np.ndarray: 256*256, depth map
    """
    car_dir = "/usr/stud/zhouzh/data/ShapeNetCar/baa1e44e6e086c233e320a6703976361/"
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
    return depth, rgb
dir = Path("../../data/ShapeNetCar/baa1e44e6e086c233e320a6703976361/")
cameras = np.load(dir / "cameras.npz")
points = np.load(dir / "pointcloud.npz")
pcl = points["points"]
scale = points["scale"]
for id in range(24):
    extrinsics = cameras[f"world_mat_{id}"]
    # extrinsics = extrinsics.numpy()
    K = cameras[f"camera_mat_{0}"][:3, :3]
    # K = np.eye(3)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, -1:]

    pcl_c = ((R @ pcl.T) + t).T
    # pcl_cam = K @ pcl_c / (K @ pcl_c)[-1]
    # img = pcl_cam.T[:,:-1]
    # img[:,1]*=-1
    z = pcl_c[:, -1]
    x = pcl_c[:, 0]
    y = pcl_c[:, 1]
    u = x * 2.1875 / z
    v = y * 2.1875 / z
    H = 256
    W = 256
    i = ((H - 1) * u + (H - 1)) / 2
    j = ((W - 1) * v + (W - 1)) / 2
    pcl_projected = np.zeros((256, 256))
    for m in range(i.shape[0]):
        if 0 <= i[m] <= 255 and 0 <= j[m] <= 255:
            pcl_projected[int(j[m]), int(i[m])] = z[m]

    depth, rgb = read_depth_rgb(id)
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    im1 = axes[0].imshow(pcl_projected)
    im2 = axes[1].imshow(depth)
    axes[2].imshow(rgb)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    plt.show()