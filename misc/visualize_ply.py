import open3d as o3d
import argparse


def show_ply(path):
    pcl = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcl])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    opt = parser.parse_args()
    show_ply(opt.path)
