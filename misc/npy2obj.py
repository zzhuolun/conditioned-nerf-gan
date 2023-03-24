from logging import raiseExceptions
import numpy as np
from pathlib import Path
import argparse
from typing import Union


def npy2obj(points: np.ndarray, obj_dir: Union[str, Path]) -> None:
    if points.shape[1] == 6:
        color = (points[:, 3:] * 255).astype("uint8")
        with open(obj_dir, "w+") as f:
            for i in range(points.shape[0]):
                f.write(
                    "v "
                    + str(points[i, 0])
                    + " "
                    + str(points[i, 1])
                    + " "
                    + str(points[i, 2])
                    + " "
                    + str(color[i, 0])
                    + " "
                    + str(color[i, 1])
                    + " "
                    + str(color[i, 2])
                    + "\n"
                )
        print("Save to ", obj_dir)
    elif points.shape[1] == 3:
        with open(obj_dir, "w+") as f:
            for i in range(points.shape[0]):
                f.write(
                    "v "
                    + str(points[i, 0])
                    + " "
                    + str(points[i, 1])
                    + " "
                    + str(points[i, 2])
                    + "\n"
                )
        print("Save to ", obj_dir)
    else:
        raise ValueError("Points dimension should be either 3 or 6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="absolute path to the .npy points file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="directory to save the .obj file",
    )
    opt = parser.parse_args()
    Path(opt.save_dir).mkdir(exist_ok=True)
    pcl = np.load(opt.path)
    obj_dir = Path(opt.save_dir) / (Path(opt.path).parent.stem + ".obj")
    npy2obj(pcl, obj_dir)
