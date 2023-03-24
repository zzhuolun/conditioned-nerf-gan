import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path
import argparse


def show_train_test_cam_dist(
    train_cam_path: str, test_cam_path: str, trainset_size: int, testset_size: int
) -> None:
    """Plot the camera origin distribution of trainset and testset.
    Used to check the distribution of the camera origins in the rendered dataset.

    Args:
        train_cam_path (str): Path to the cameras.npz of trainset.
        test_cam_path (str): Path to the cameras.npz of testset.
        trainset_size (int): Length of the trainset.
        testset_size (int): Length of the testset.
    """
    cams_train = np.load(train_cam_path)
    cams_test = np.load(test_cam_path)
    origin_train = []
    origin_test = []
    for i in range(trainset_size):
        origin_train.append(cams_train[f"world_mat_inv_{i}"][:3, -1])
    for i in range(testset_size):
        origin_test.append(cams_test[f"world_mat_inv_{i}"][:3, -1])
    origin_train = np.asarray(origin_train)
    origin_test = np.asarray(origin_test)
    print(origin_train.shape)
    print(origin_test.shape)
    color_train = np.ones_like(origin_train) * np.array([0, 0, 1])
    color_test = np.ones_like(origin_test) * np.array([1, 0, 0])
    origins = np.concatenate((origin_train, origin_test), axis=0)
    colors = np.concatenate((color_train, color_test), axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(origins[:, 0], origins[:, 1], origins[:, 2], c=colors)
    plt.title("Camera origin in world space")
    plt.show()


def show_img_in_grid(image_folder_dir: str, nrows: int, ncols: int) -> None:
    """Plot the images in image_folder_dir in grid.

    Args:
        image_folder_dir (str): folder path
        nrows (int): number of rows
        ncols (int): number of columns
    """

    img_ls = []
    for p in Path(image_folder_dir).iterdir():
        img_ls.append(plt.imread(p))

    fig = plt.figure(figsize=(4.0, 4.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),
        axes_pad=0,  # pad between axes in inch.
    )

    for ax, im in zip(grid, img_ls):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="absolute path to the images folder",
    )
    parser.add_argument("--nrows", type=int, default=1, help="size of generated images")
    opt = parser.parse_args()

    ncols = len(list(Path(opt.path).iterdir())) // opt.nrows
    if len(list(Path(opt.path).iterdir())) % opt.nrows != 0:
        ncols += 1
    show_img_in_grid(opt.path, opt.nrows, ncols)
