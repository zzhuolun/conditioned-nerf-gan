import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import argparse
import mrcfile
import cv2 as cv
import torch

def np_to_cv(img):
    img = torch.from_numpy(img)
    img = (
        img.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .to("cpu", torch.uint8)
        .numpy()
    )
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def pcl2_voxel(
    root_dir,
    car_name,
    resolution,
    length,
    voxel_name,
    noise_color=0,
    noise_xyz=0,
    save=True,
):
    pcl_dir = root_dir / car_name / "pcl_color.npy"
    pcl = np.load(pcl_dir)
    points = pcl[:, :3]
    colors = pcl[:, 3:]
    # Add noise to the point cloud
    points += np.random.randn(points.shape[0], points.shape[1]) * noise_xyz
    colors += np.random.randn(colors.shape[0], colors.shape[1]) * noise_color
    points = np.clip(points, -length / 2 + 1e-4, length / 2 - 1e-4)
    colors = np.clip(colors, 0, 1)

    assert points[:, 0].max() <= length / 2
    assert points[:, 1].max() <= length / 2
    assert points[:, 2].max() <= length / 2
    assert points[:, 0].min() >= -length / 2
    assert points[:, 1].min() >= -length / 2
    assert points[:, 2].min() >= -length / 2

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=length / resolution,
        min_bound=np.array([-length / 2, -length / 2, -length / 2]),
        max_bound=np.array([length / 2, length / 2, length / 2]),
    )
    if save:
        voxel = np.zeros((resolution, resolution, resolution, 4), dtype=float)
        for v in voxel_grid.get_voxels():
            # print(v)
            voxel[v.grid_index[0], v.grid_index[1], v.grid_index[2], 0] = 1
            voxel[v.grid_index[0], v.grid_index[1], v.grid_index[2], 1:] = v.color
        output = {}
        output["length"] = length
        output["resolution"] = resolution
        output["noise_color"] = noise_color
        output["noise_xyz"] = noise_xyz
        output["voxel"] = voxel
        np.savez(
            root_dir / car_name / voxel_name,
            **output,
        )
        print("Saving voxel of ", car_name)
    else:
        o3d.visualization.draw_geometries([voxel_grid])


def pcl2_voxelvideo(
    root_dir,
    car_name,
    cat,
    resolution,
    length,
    voxel_name,
    noise_color=0,
    noise_xyz=0,
):
    pcl_dir = root_dir / car_name / "pcl_color.npy"
    pcl = np.load(pcl_dir)
    # idx = np.random.randint(100000, size=2048)
    # pcl = pcl[idx]
    points = pcl[:, :3]
    colors = pcl[:, 3:]
    # Add noise to the point cloud
    points += np.random.randn(points.shape[0], points.shape[1]) * noise_xyz
    colors += np.random.randn(colors.shape[0], colors.shape[1]) * noise_color
    points = np.clip(points, -length / 2 + 1e-4, length / 2 - 1e-4)
    colors = np.clip(colors, 0, 1)

    assert points[:, 0].max() <= length / 2
    assert points[:, 1].max() <= length / 2
    assert points[:, 2].max() <= length / 2
    assert points[:, 0].min() >= -length / 2
    assert points[:, 1].min() >= -length / 2
    assert points[:, 2].min() >= -length / 2

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=length / resolution,
        min_bound=np.array([-length / 2, -length / 2, -length / 2]),
        max_bound=np.array([length / 2, length / 2, length / 2]),
    )
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=256, height=256)
    vis.add_geometry(voxel_grid)
    ctr = vis.get_view_control()
    
    output_name = Path('voxel_video') / cat / (car_name+'.mp4')
    output_name.parent.mkdir(parents=True, exist_ok=True)
    cam2world = np.load('/home/stud/zhouzh/pi-gan/cam_video.npy')
    num_frames = cam2world.shape[0]

    camera_intrisic=o3d.camera.PinholeCameraIntrinsic(256, 256, 2.1875*128, 2.1875*128, 127.5, 127.5)
    # Render each frame
    fourcc = cv.VideoWriter_fourcc(*"MPEG")
    video = cv.VideoWriter(
        str(output_name),
        fourcc,
        24,
        (256, 256),
    )
    for i in range(num_frames):
        # Change the camera position, orientation, and zoom level
        # azimuth = i * 360.0 / num_frames
        # elevation = i * 180.0 / num_frames
        # zoom = 0.5
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_zoom(zoom)
        # ctr.rotate(azimuth, elevation)
        cam = o3d.camera.PinholeCameraParameters()
        cam.extrinsic = np.linalg.inv(cam2world[i])
        cam.intrinsic = camera_intrisic
        ctr.convert_from_pinhole_camera_parameters(cam)
        # Save the frame as an image
        # filename = os.path.join(output_dir, f'frame_{i:04d}.png')
        # vis.capture_screen_image(filename, do_render=True)
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        video.write(np_to_cv(frame))
    cv.destroyAllWindows()
    video.release()
    print("Writing video to ", str(output_name))
    # Close the visualization window
    vis.destroy_window()


def voxel2mrc(voxel_dir, save_path):
    voxel = np.load(voxel_dir)["voxel"]
    voxel = voxel[:, :, :, 0]
    with mrcfile.new_mmap(
        save_path,
        overwrite=True,
        shape=voxel.shape,
        mrc_mode=2,
    ) as mrc:
        mrc.data[:] = voxel
    print("Save voxel to ", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the voxel grid from pcl_color.npy"
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=64,
        help="The resolution of the generated voxel grid",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=float,
        default=1.2,
        help="The length of the cube for voxel grid",
    )
    parser.add_argument(
        "--cat",
        type=str,
    )
    parser.add_argument(
        "--object_name",
        type=str,
    )
    opt = parser.parse_args()
    noise_xyz = 0  # 0.01
    noise_color = 0  # 0.01
    if opt.cat == 'car':
        root_dir = Path("/storage/user/zhouzh/data/ShapeNetCar/ShapeNetCar")
    elif opt.cat =='chair':
        root_dir = Path("/storage/user/zhouzh/data/ShapeNetChair/ShapeNetChair")
    elif opt.cat=='plane':
        root_dir = Path("/storage/user/zhouzh/data/ShapeNetPlane/ShapeNetPlane")

    pcl2_voxel(
            root_dir,
            opt.object_name,
            opt.cat,
            opt.resolution,
            opt.length,
            voxel_name=None,
            noise_color=0,
            noise_xyz=0,
        )