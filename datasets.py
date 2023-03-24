# --------------
# Datasets
# --------------

import glob
import random
from pathlib import Path
from re import S

import cv2 as cv
import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Sampler
import OpenEXR


class ShapeNet(Dataset):
    def __init__(
        self,
        dataset_path: str,
        img_size: int,
        sampling_mode: str,
        metadata: dict,
        load_voxel: bool,
        load_pcl: bool,
        dataset_mode: str = "train",
        shuffle: bool = True,
        multifold: int = 1,
        load_img: bool = True,
        load_depth: bool = True,
        load_cam: bool = True,
        cond_mode: str = None,
        voxel_resolution: int = 64,
        # **kwargs,
    ):
        """
        Args:
            dataset_path (str): The path to the dataset.
            img_size (int): Loaded image resolution.
            sampling_mode (str): Which part of shapenet cars to load. thousand | twenty | single | ...
            dataset_mode (str, optional): train | val | test. Train and val belongs to the same cars, the only difference is that the pose of valset is novel. Testset has unknown cars.
            shuffle (bool, optional): If false, use seed to control the randomness. Defaults to True.
            voxel_res (int, optional): The resolution of loaded voxel. Defaults to 64.
        """
        super().__init__()
        test_view_per_car = 1
        print(
            "Loading cars from: ",
            str(Path(dataset_path).parent / f"{sampling_mode}.lst"),
        )
        with open(
            Path(dataset_path).parent / f"{sampling_mode}.lst", "r"
        ) as f:
            seq_ls = f.readlines()
        cars_dir_ls = [Path(dataset_path) / i[:-1] for i in seq_ls]
        self.data = []
        for car in cars_dir_ls:
            all_imgs = sorted(
                [i for i in (car / "image").iterdir() if i.suffix == ".png"]
            )
            if dataset_mode == "train":
                self.data += all_imgs[:-test_view_per_car]
            elif dataset_mode == "val":
                self.data += all_imgs[-test_view_per_car:]
            elif dataset_mode == "test":
                self.data += all_imgs
            else:
                raise AssertionError("Undefined dataset mode!")

        # if too few cars in the trainset, training would be slow
        print("Multifold: ", multifold)
        self.data *= multifold
        if shuffle:
            random.shuffle(self.data)
        print(f"{dataset_mode}set size: ", len(self.data))
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.img_size = img_size
        self.load_img = load_img
        self.load_depth = load_depth
        self.load_voxel = load_voxel
        self.load_pcl = load_pcl
        self.load_cam = load_cam
        self.cond_mode = cond_mode
        self.voxel_resolution = voxel_resolution
        self.metadata = metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {}
        img_path = self.data[index]
        sample["img_path"] = str(img_path)

        if self.load_img:
            X = _read_resize_shapenet(str(img_path), self.img_size)
            X = self.transform(X)
            sample["img"] = X.type(torch.float32)

        if self.cond_mode or self.load_cam:
            cameras_npz = np.load(img_path.parent.parent / "cameras.npz")
            if self.load_cam:
                cam2world = torch.from_numpy(
                    cameras_npz[f"world_mat_inv_{int(img_path.stem)}"]
                ).type(torch.float32)
                sample["cam2world"] = cam2world

            if self.cond_mode:
                cond_path = sample_cond(img_path, cameras_npz, self.cond_mode)
                if (
                    self.metadata["discriminator"]
                    == "ProgressiveDiscriminator_projection"
                ):
                    cond = _read_resize_shapenet(str(cond_path), 256)
                else:
                    cond = _read_resize_shapenet(str(cond_path), self.img_size)
                cond = self.transform(cond)
                sample["cond"] = cond.type(torch.float32)

        if self.load_depth:
            depth = readEXR(
                str(img_path.parent.parent / "depth" / (img_path.stem + "0001.exr")),
                self.img_size,
            )
            sample["depth"] = depth

        if self.load_voxel:
            if self.voxel_resolution == 64:
                sample["voxel"] = (
                    torch.from_numpy(
                        np.load(img_path.parent.parent / "voxel.npz")["voxel"]
                    )
                    .permute(3, 2, 1, 0)
                    .type(torch.float)
                )
            elif self.voxel_resolution == 32:
                sample["voxel"] = (
                    torch.from_numpy(
                        np.load(img_path.parent.parent / "voxel_32.npz")["voxel"]
                    )
                    .permute(3, 2, 1, 0)
                    .type(torch.float)
                )
            elif self.voxel_resolution == 128:
                sample["voxel"] = (
                    torch.from_numpy(
                        np.load(img_path.parent.parent / "voxel_128.npz")["voxel"]
                    )
                    .permute(3, 2, 1, 0)
                    .type(torch.float)
                )            
            else:
                raise ValueError("Unsupported voxel resolution")

        if self.load_pcl:
            pcl = torch.from_numpy(
                np.load(img_path.parent.parent / "pcl_color.npy")
            ).type(torch.float32)
            pcl = pcl[
                torch.randperm(pcl.shape[0])[
                    : int(pcl.shape[0] * self.metadata["num_points_ratio"])
                ],
            ]
            sample["pcl"] = pcl
        return sample


def _read_resize_shapenet(path, img_size):
    """Read and resize images from shapenet"""
    X = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    X = cv.cvtColor(X, cv.COLOR_BGRA2RGB)
    if img_size != X.shape[0]:
        mode = cv.INTER_LINEAR if img_size > X.shape[0] else cv.INTER_AREA
        X = cv.resize(X, dsize=(img_size, img_size), interpolation=mode)
    X = X / (2**16 - 1)
    return X


def readEXR(fpath: str, img_size: int) -> np.ndarray:
    """Read depth map from the .exr file

    Args:
        fpath (str): the path to the .exr depth map

    Returns:
        np.ndarray: 256*256, depth map
    """
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
    # set inf depth values(background) as 0
    depth = np.reshape(dr, (h, w)).astype(float)
    depth[depth == 65504] = np.inf
    if img_size != depth.shape[0]:
        mode = cv.INTER_LINEAR if img_size > depth.shape[0] else cv.INTER_AREA
        depth = cv.resize(depth, dsize=(img_size, img_size), interpolation=mode)
    depth[depth == np.inf] = 0
    return depth


def sample_cond(img_path: Path, cameras, mode: str) -> str:
    current_id = int(img_path.stem)
    if mode == "random":
        new_id = (current_id + random.randint(1, 23)) % 24
        return str(img_path.parent / f"{new_id:04}.png")

    cam_origins = np.zeros((24, 3))
    for i in range(24):
        cam_origins[i] = cameras[f"world_mat_inv_{i}"][:3, -1]
    argsort = np.argsort(np.linalg.norm(cam_origins - cam_origins[current_id], axis=1))
    if mode == "nearest":
        return str(img_path.parent / f"{argsort[1]:04}.png")
    elif mode == "furthest":
        return str(img_path.parent / f"{argsort[-1]:04}.png")
    else:
        raise ValueError("Unsupported mode type!")


def get_dataset(metadata: dict, subsample=None):
    # name should be CelebA|Cats|Carla|CO3D|shapenet, dataset is an instance of these classes
    if metadata["dataset"]["trainset"] in ["twenty", "single"]:
        multifold = 3000
    else:
        multifold = 1
    dataset = ShapeNet(
        metadata["dataset"]["path"],
        metadata["img_size"],
        metadata["dataset"]["trainset"],
        metadata,
        metadata["dataset"]["load_voxel"],
        metadata["dataset"]["load_pcl"],
        shuffle=True,
        multifold=multifold,
        cond_mode=metadata["dataset"]["cond_mode"],
        voxel_resolution=metadata["voxel_resolution"],
        load_depth=metadata["dataset"]["load_depth"],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=metadata["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=metadata["num_workers"],
    )
    return dataloader, 3


def get_dataset_distributed(metadata: dict, world_size: int, rank: int):
    if metadata["dataset"]["trainset"] in ["twenty", "single"]:
        multifold = 3000
    else:
        multifold = 1
    dataset = ShapeNet(
        metadata["dataset"]["path"],
        metadata["img_size"],
        metadata["dataset"]["trainset"],
        metadata,
        metadata["dataset"]["load_voxel"],
        metadata["dataset"]["load_pcl"],
        shuffle=True,
        multifold=multifold,
        cond_mode=metadata["dataset"]["cond_mode"],
        voxel_resolution=metadata["voxel_resolution"],
        load_depth=metadata["dataset"]["load_depth"],
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=metadata["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=metadata["num_workers"],
    )

    return dataloader, 3


class TestsetSampler(Sampler):
    """Customised Sampler to sample specific indices from the testset, to reduce computation time."""

    def __init__(self, indices: list):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
