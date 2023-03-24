"""
Contains code for logging approximate FID scores during training.
If you want to output ground-truth images from the training dataset, you can
run this file as a script.
"""

from faulthandler import cancel_dump_traceback_later
import os
import shutil
from turtle import Shape
import torch
import copy
import argparse

from torchvision.utils import save_image
from pytorch_fid import fid_score
from tqdm import tqdm

import datasets
from datasets import ShapeNet
from configs import curriculums
from pathlib import Path

from generators.volumetric_rendering import (
    create_cam2world_matrix,
    sample_camera_positions,
)
from generators.pointnet import encode_pcl


def normalize_before_save(img):
    # This step is to counter the transforms.Normalize([0.5], [0.5]) in the dataset definition
    img = (img * 0.5) + 0.5
    # return torch.clamp(img, 0, 1)
    return img


def output_real_images(dataloader, real_dir):
    for sample in dataloader:
        real_imgs = sample["img"]
        real_imgs = normalize_before_save(real_imgs)
        for idx, img in enumerate(real_imgs):
            save_image(
                img,
                os.path.join(
                    real_dir,
                    sample["img_path"][idx].split("/")[-3]
                    + "_"
                    + sample["img_path"][idx].split("/")[-1],
                ),
                normalize=False,
                # range=(-1, 1),
            )


def setup_evaluation(metadata: dict):
    # Only make real images if they haven't been made yet
    real_dir = os.path.join(
        "/usr/stud/zhouzh/EvalImages",
        Path(metadata["dataset"]["path"]).stem,
        metadata["fid"]["subset"] + "_" + str(metadata["fid"]["target_size"]),
    )
    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
        dataset = ShapeNet(
            metadata["dataset"]["path"],
            metadata["fid"]["target_size"],
            metadata["fid"]["subset"],
            metadata,
            load_pcl=False,
            load_voxel=False,
            dataset_mode="test",
            load_depth=False,
            load_cam=False,
        )
        # assert len(dataset) >= metadata["fid"]["num_real_imgs"]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            drop_last=False,
            pin_memory=False,
            num_workers=metadata["num_workers"],
        )
        print("Outputting real images...")
        output_real_images(dataloader, real_dir)
        print("...done")
    return real_dir


def output_images(
    generator, encoder, dataset, input_metadata, rank, world_size, output_dir, device
):

    metadata = copy.deepcopy(input_metadata)
    metadata["img_size"] = 128
    if input_metadata["num_steps"] > 48:
        metadata["batch_size"] = 2
    else:
        metadata["batch_size"] = 8

    # img_counter = rank
    generator.eval()

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
    print(f"Generating {len(dataset)} images at {output_dir}")
    if rank == 0:
        pbar = tqdm("generating images", total=len(dataloader))
    with torch.no_grad():
        for sample in dataloader:
            if metadata["dataset"]["load_voxel"]:
                voxel = sample["voxel"].to(device)
                z = encoder(voxel)
            if metadata["dataset"]["load_pcl"]:
                pcl = sample["pcl"].to(device)
                z, _ = encode_pcl(encoder, pcl, device, 0)
            # for _ in range(metadata["fid"]["generated_imgs_per_car"]):
            # cam_origins = sample_camera_positions(
            #     device,
            #     "y",
            #     metadata["cam_r_start"],
            #     metadata["cam_r_end"],
            #     metadata["batch_size"],
            # )
            # cam2worlds = create_cam2world_matrix(cam_origins, "y", device)
            imgs = generator(z, sample["cam2world"].to(device), **metadata)[0].cpu()
            # imgs = (imgs * 0.5) + 0.5
            imgs = normalize_before_save(imgs)
            for idx, img in enumerate(imgs):
                save_image(
                    img,
                    os.path.join(
                        output_dir,
                        sample["img_path"][idx].split("/")[-3]
                        + "_"
                        + sample["img_path"][idx].split("/")[-1],
                    ),
                    normalize=False,
                    # range=(-1, 1),
                )
            if rank == 0:
                pbar.update(1)
        # imgs = torch.cat(imgs_ls, dim=0)
        # for img in imgs:
        #     save_image(
        #         img,
        #         os.path.join(output_dir, f"{img_counter:0>5}.jpg"),
        #         normalize=True,
        #         range=(-1, 1),
        #     )
        #     img_counter += world_size
    if rank == 0:
        pbar.close()


def calculate_fid(real_dir, generated_dir):
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, generated_dir], 128, "cuda", 2048
    )
    torch.cuda.empty_cache()
    return fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Carla")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--num_imgs", type=int, default=8000)
    parser.add_argument("--dataset_path", type=str, default='"./data/carla/*.png"')
    opt = parser.parse_args()

    real_images_dir = setup_evaluation(
        opt.dataset,
        None,
        data_path=opt.dataset_path,
        target_size=opt.img_size,
        num_imgs=opt.num_imgs,
    )
