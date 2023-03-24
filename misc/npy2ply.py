import open3d as o3d
from pathlib import Path
import numpy as np
import argparse


def npy2ply(npy_path: Path, show: bool, save_dir: Path = None):
    pcl_np = np.load(npy_path)
    pcd = o3d.geometry.PointCloud()
    if pcl_np.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(pcl_np)
    elif pcl_np.shape[1] == 6:
        pcd.points = o3d.utility.Vector3dVector(pcl_np[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pcl_np[:, 3:])
    if show:
        o3d.visualization.draw_geometries([pcd])
    else:
        print("Saving point cloud at ", str(
            save_dir/(npy_path.parent.stem+'.ply')))
        o3d.io.write_point_cloud(save_dir/(npy_path.parent.stem+'.ply'), pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="absolute path to the .npy points file",
    )
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--show", type=bool, default=True)
    opt = parser.parse_args()
    npy2ply(Path(opt.path), opt.show, opt.save_dir)
