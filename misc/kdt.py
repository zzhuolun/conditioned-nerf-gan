# -------------------------------------------------------------------------------- 
# Pick a random car and visualizes the sampled negative points and the postive pcl.
# --------------------------------------------------------------------------------

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
import numpy as np
import torch
import open3d as o3d
import time
import random
from pathlib import Path


def visualize_sampled_negative_points(pcl_path, radius, num_pos_points, neg_to_pos_ratio):
    pcl = np.load(pcl_path)["points"]
    pcl = torch.from_numpy(pcl)
    pcl = pcl.type(torch.float32)
    pcl_ds = pcl[torch.randperm(100000)[:num_pos_points]]
    # Y = pdist(pcl_ds, 'euclidean')
    # print(Y.shape)
    # print(Y.min().item(), Y.max().item())

    start = time.time()
    tree = KDTree(pcl_ds)
    neg = torch.rand(int(num_pos_points * neg_to_pos_ratio), 3) - 0.5
    results = tree.query_ball_point(neg, radius)
    ids = [idx for idx, i in enumerate(results) if not i]
    real_neg = neg[ids]
    end = time.time()
    
    print(f"KDTree query time: {round(end - start, 5)}s")
    print("neg ratio: ", len(ids) / (num_pos_points * neg_to_pos_ratio), " | neg num: ", len(ids), "| pos num: ", num_pos_points)
    print(
        f"The average number of pos neighbors within {radius} ball per point:",
        len(tree.query_pairs(r=radius)) * 2 / num_pos_points,
    )

    real_neg = real_neg[torch.where(real_neg[:, 0] > -0.3)]
    real_neg = real_neg[torch.where(real_neg[:, 0] < 0.3)]

    neg = o3d.geometry.PointCloud()
    neg.points = o3d.utility.Vector3dVector(real_neg)
    # neg.paint_uniform_color([1, 0.706, 0])
    neg.colors = o3d.utility.Vector3dVector(real_neg + 0.5)

    pos = o3d.geometry.PointCloud()
    pos.points = o3d.utility.Vector3dVector(pcl_ds)
    pos.paint_uniform_color([1, 0.706, 0])

    o3d.visualization.draw_geometries([neg, pos], width=1800, height=950)


if __name__ == "__main__":
    dir = "../data/ShapeNetCar/"
    pcl_path = random.sample(list(Path(dir).iterdir()), 1)[0] / "pointcloud.npz"
    pos_points_ratio = 0.1
    radius = 0.025
    neg_to_pos_ratio = 1/0.7
    visualize_sampled_negative_points(pcl_path, radius, int(100000*pos_points_ratio), neg_to_pos_ratio)
