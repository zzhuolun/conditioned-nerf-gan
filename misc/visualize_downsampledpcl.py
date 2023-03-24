import open3d as o3d
import argparse
import numpy as np


def visualize(filename):
    xyz = np.load(filename)
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("/usr/stud/zhouzh/tmp/sync.ply", pcd)

    # # Load saved point cloud and visualize it
    # pcd_load = o3d.io.read_point_cloud("/usr/stud/zhouzh/tmp/sync.ply")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize shapent car pointcloud"
    )
    parser.add_argument("file", help="The file .npy to be visualized")
    opt = parser.parse_args()
    visualize(opt.file)