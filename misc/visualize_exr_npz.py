import open3d as o3d
from pathlib import Path

import numpy as np
import OpenEXR as exr
import Imath
from matplotlib import pyplot as plt
import argparse


def readEXR(filename):
    """Read color + depth data from EXR image file.

    Parameters
    ----------
    filename : str
        File path.

    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.

    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.

    From https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
    """

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header["dataWindow"]
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    # convert all channels in the image to numpy arrays
    for c in header["channels"]:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.HALF))
        C = np.frombuffer(C, dtype=np.int16)
        C = np.reshape(C, isize)

        channelData[c] = C

    colorChannels = (
        ["R", "G", "B", "A"] if "A" in header["channels"] else ["R", "G", "B"]
    )
    img = np.concatenate(
        [channelData[c][..., np.newaxis] for c in colorChannels], axis=2
    )
    img = (img / 256).astype("uint8")
    img = img / 256
    # linear to standard RGB
    img[..., :3] = np.where(
        img[..., :3] <= 0.0031308,
        12.92 * img[..., :3],
        1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055,
    )

    # sanitize image to be in range [0, 1]
    img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))

    # Z = None if "Z" not in header["channels"] else channelData["Z"]

    return img


def visualize_pcl(filename):
    """Visualize the .npz point cloud with open3d"""
    pcd_npz = np.load(filename)
    xyz = pcd_npz["points"]
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("/usr/stud/zhouzh/tmp/sync.ply", pcd)

    # # Load saved point cloud and visualize it
    # pcd_load = o3d.io.read_point_cloud("/usr/stud/zhouzh/tmp/sync.ply")
    o3d.visualization.draw_geometries([pcd])

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    # xyz_load = np.asarray(pcd_load.points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize shapent car pointcloud/depth"
    )
    parser.add_argument("file", help="The file (.npz or .exr) to be visualized")
    opt = parser.parse_args()
    file_type = Path(opt.file).suffix

    if file_type == '.npz':
        visualize_pcl(opt.file)
    elif file_type == '.exr':
        img = readEXR(opt.file)
        plt.imshow(img)
        plt.show()
    else:
        raise ValueError("Unsupported file type!")
