import sys

sys.path.append("../")
from utils import *
from pathlib import Path
import numpy as np
from blender_renderer.project import backproject_points
from tqdm import tqdm

# fp = Path("/usr/stud/zhouzh/data/ShapeNetCar/")
# fp = Path("/storage/user/zhouzh/data/ShapeNetChair/ShapeNetChair")
fp = Path("/storage/user/zhouzh/data/ShapeNetPlane/ShapeNetPlane")
with open(fp.parent / "thousand.lst", "r") as f:
    thousand = f.readlines()
with open(fp.parent / "twenty.lst", "r") as f:
    twenty = f.readlines()
with open(fp.parent / "val.lst", "r") as f:
    val = f.readlines()
thousand = [i[:-1] for i in thousand]
twenty = [i[:-1] for i in twenty]
val = [i[:-1] for i in val]

cars_path = [fp / i for i in thousand + val + twenty]
# cars_path = [fp/i for i in twenty]
num_views = 24
pcl_size = 100000
error_car = []


for car_path in tqdm(cars_path):  # fp.iterdir()):
    # car_path = fp / car[:-1]
    if car_path.is_dir():
        pcl_name = car_path / "pcl_color.npy"
        if not (pcl_name.exists() and pcl_name.stat().st_size > 0):
            try:
                points_ls = []
                for i in range(num_views):
                    depht_pts, color_pts = backproject_points(car_path, i, 256, False)
                    points = np.concatenate([depht_pts, color_pts], axis=1)
                    points_ls.append(points)
                points = np.concatenate(points_ls, axis=0)
                if points.shape[0] >= pcl_size:
                    points_downsampled = points[
                        random.sample(range(points.shape[0]), pcl_size)
                    ]
                else:
                    print("total points less than", pcl_size)
                    print(car_path)
                    points_downsampled = points

                np.save(pcl_name, points_downsampled)
                print("Saving pcl to ", str(pcl_name))
            except Exception as e:
                print(e)
                print(str(pcl_name))
                error_car.append(str(pcl_name))
print("Unfinished cars:")
print(error_car)
