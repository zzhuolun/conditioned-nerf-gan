import argparse
import copy
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Union
from xml.sax import default_parser_list

import numpy as np
import torch
from importlib_metadata import metadata
from scipy.spatial import KDTree
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.distributed as dist
import configs
from configs import thesis
# from configs import sixthousand, thousand

import fid_evaluation
from configs import curriculums
from datasets import ShapeNet, TestsetSampler
from discriminators import discriminators
from generators import generators, siren, unet3d
from generators.pointnet import ResnetPointnet, encode_pcl
# from generators.encoder import VoxelEncoder, UNet3D
from generators.volumetric_rendering import (
    sample_camera_positions,
    create_cam2world_matrix,
)
from metric_utils import compute_metrics_from_dirs


def jsonKeys2int(x):
    def to_int(y):
        if isinstance(y, int):
            return y
        if y.isdigit():
            return int(y)
        else:
            return y

    if isinstance(x, dict):
        return {to_int(k): v for k, v in x.items()}
    return x


def z_sampler(shape: tuple[int, int], device: torch.device, dist: str) -> torch.Tensor:
    if dist == "gaussian":
        z = torch.randn(shape, device=device)
    elif dist == "uniform":
        z = torch.rand(shape, device=device) * 2 - 1
    return z

def generate_img(
    generator,
    z: torch.Tensor,
    cam2worlds: torch.Tensor,
    metadata: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate rgb and depth map with latent code z

    Args:
        generator: generator
        z (torch.Tensor): latent code
        cam2world (torch.Tensor, optional): The origin of camera in world space. Defaults to None.
        metadata (dict): parameters for generator inference
    Returns:
        tuple[torch.Tensor, torch.Tensor]: generated rgb and depth image
    """
    with torch.no_grad():
        # assert cam_origin is not None
        img, depth_map = generator(z, cam2worlds, **metadata)
        img = img.cpu()
        depth_map = depth_map.cpu()
        depth_map = torch.stack([depth_map] * 3, 1)
    return img, depth_map


def _find_newest_ckpoints(dir: Path) -> Union[None, str]:
    """dir: parent path of the checkpoints"""
    steps = []
    if dir.exists():
        if list(dir.glob("*.tar")):
            for name in dir.glob("*.tar"):
                steps.append(int(name.stem))
            return f"{max(steps)}.tar"
    return None


def loss_depth(gt: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
    """Compute depth loss (mse loss) at the foreground pixels in ground truth."""
    non_zero_idx = torch.where(gt != 0)
    return ((gt[non_zero_idx] - preds[non_zero_idx]) ** 2).mean()


def loss_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the mse photometric loss. Make sure the two tensor has same size and is on the same device."""
    return ((x - y) ** 2).mean()
    # return torch.abs(img_gt - img_gen).mean()


def loss_geometry(sigma_preds: torch.Tensor, num_pos_points: int) -> torch.Tensor:
    """Compute the geometry loss by punishing small sigma prediction for positive points and large sigma prediction for negative poionts.

    Args:
        sigma_preds (torch.Tensor): the sigma output of siren network
        num_pos_points (int): number of positive points
    """
    geometry_loss = (
        torch.nn.functional.softplus(-sigma_preds[:, :num_pos_points]).mean()
        + torch.nn.functional.softplus(sigma_preds[:, num_pos_points:]).mean()
    )
    return geometry_loss


class Trainer:
    """
    Class to train and evaluate piGAN, load models, save checkpoints, write generated images, print out logs, etc
    """

    def __init__(self, parser: argparse.ArgumentParser, rank: int, world_size: int):
        self.parser = parser
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(rank)
        self._load_curriculum(self.parser)
        self.metadata = curriculums.extract_metadata(self.curriculum, 0)
        self._mkdirs()
        if self.rank == 0:
            print(self.curriculum)
            print(self.parser)
            # self.writer = SummaryWriter(self.tensorboard_path)
        self.trainset = ShapeNet(
            self.metadata["dataset"]["path"],
            img_size=128,
            sampling_mode=self.metadata["dataset"]["trainset"],
            metadata=self.metadata,
            load_voxel=self.metadata["dataset"]["load_voxel"],
            load_pcl=self.metadata["dataset"]["load_pcl"],
            shuffle=True,
            multifold=1,
            voxel_resolution=self.metadata["voxel_resolution"],
        )

        self.valset = ShapeNet(
            self.metadata["dataset"]["path"],
            img_size=128,
            sampling_mode=self.metadata["dataset"]["trainset"],
            metadata=self.metadata,
            load_voxel=self.metadata["dataset"]["load_voxel"],
            load_pcl=self.metadata["dataset"]["load_pcl"],
            dataset_mode="val",
            shuffle=False,
            multifold=1,
            voxel_resolution=self.metadata["voxel_resolution"],
        )

        self.testset = ShapeNet(
            self.metadata["dataset"]["path"],
            img_size=128,
            sampling_mode=self.metadata["dataset"]["testset"],
            metadata=self.metadata,
            load_voxel=self.metadata["dataset"]["load_voxel"],
            load_pcl=self.metadata["dataset"]["load_pcl"],
            dataset_mode="test",
            shuffle=False,
            multifold=1,
            voxel_resolution=self.metadata["voxel_resolution"],
        )

        self.fidset = ShapeNet(
            self.metadata["dataset"]["path"],
            self.metadata["fid"]["target_size"],
            self.metadata["fid"]["subset"],
            metadata=self.metadata,
            load_voxel=self.metadata["dataset"]["load_voxel"],
            load_pcl=self.metadata["dataset"]["load_pcl"],
            dataset_mode="test",
            load_img=False,
            load_depth=False,
            load_cam=True,
            voxel_resolution=self.metadata["voxel_resolution"],
        )

    def _load_curriculum(self, parser):
        """
        Load curriculum (i.e. hyperparameters) dictionary.
        """
        if self.parser.load_curriculum:
            with open(self.parser.load_curriculum, "r") as f:
                print("Loading curriculum from", self.parser.load_curriculum)
                self.curriculum = json.load(f, object_hook=jsonKeys2int)
        elif (Path(self.parser.output_dir) / "curriculum.json").exists():
            with open(Path(self.parser.output_dir) / "curriculum.json", "r") as f:
                print(
                    "Loading curriculum from",
                    Path(self.parser.output_dir) / "curriculum.json",
                )
                self.curriculum = json.load(f, object_hook=jsonKeys2int)
        elif (
            self.parser.load_dir
            and (Path(self.parser.load_dir).parent.parent / "curriculum.json").exists()
        ):
            with open(
                Path(self.parser.load_dir).parent.parent / "curriculum.json", "r"
            ) as f:
                print(
                    "Loading curriculum from",
                    Path(self.parser.load_dir).parent.parent / "curriculum.json",
                )
                self.curriculum = json.load(f, object_hook=jsonKeys2int)
        else:
            print(f"Creating curriculum from configs.{parser.config_base}")
            self.curriculum = curriculums.update_recursive(
                eval(f"{parser.config_base}.default"),
                eval(f"{parser.config_base}.special"),
            )
            if parser.config:
                print(f"Update curriculum with {parser.config}")
                self.curriculum = curriculums.update_recursive(
                    self.curriculum, eval(f"{parser.config_base}.{parser.config}")
                )

    def _mkdirs(self) -> None:
        """
        Make subdirectories at the parser.output_dir, the output folder is organized as follows:
        checkpoints/: the dir to store model & optimizer checkpoints
        logs/: the slurm output dir
        samples/: the frequently generated sample images
        evaluation/: the generated images to compute fid score
        """
        output_dir = Path(self.parser.output_dir)
        self.checkpoints_path = output_dir / "checkpoints"
        self.logs_path = output_dir / "logs"
        self.samples_train_path = output_dir / "samples" / "trainview"
        self.samples_val_path = output_dir / "samples" / "valview"
        self.samples_test_path = output_dir / "samples" / "testview"
        self.evaluation_path = output_dir / "evaluation"
        # self.tensorboard_path = output_dir.parent / "tensorboard" / output_dir.stem
        # assert self.logs_path.exists()
        if self.rank == 0:
            if not self.checkpoints_path.exists():
                print(
                    "Creating directory at ",
                    self.checkpoints_path,
                    " to save checkpoints",
                )
            self.checkpoints_path.mkdir(exist_ok=True)
            if not self.samples_train_path.exists():
                print(
                    "Creating directory at ",
                    self.samples_train_path,
                    " to save generated sample images",
                )
            self.samples_train_path.mkdir(parents=True, exist_ok=True)

            if not self.evaluation_path.exists():
                print(
                    "Creating directory at ",
                    self.evaluation_path,
                    " to save generatd images for fid computation",
                )
            self.evaluation_path.mkdir(parents=True, exist_ok=True)

            if not self.samples_val_path.exists():
                print(
                    "Creating directory at ",
                    self.samples_val_path,
                    " to save generatd images from test set",
                )
            self.samples_val_path.mkdir(exist_ok=True)

            if not self.samples_test_path.exists():
                print(
                    "Creating directory at ",
                    self.samples_test_path,
                    " to save generatd images from test set",
                )
            self.samples_test_path.mkdir(exist_ok=True)

    def load_models(self) -> None:
        """
        Load or create empty generator, discriminator, optimizer_G, optimizer_D, scaler, ema, ema2, generator_losses and discriminator_losses.
        Those models and optimizers are stored as attriutes of LogTrain class

        Loading logic:
        If checkpoints exist in self.parser.output_dir/checkpoints, then load from this directory; (this step is to avoid slurm reschedule)
        else if self.parser.load_dir actually store a .tar model, then load from here;
        else, create empty models, optimizers, etc.
        """
        newest_ck_output = _find_newest_ckpoints(self.checkpoints_path)
        # newest_ck_load = (
        #     _find_newest_ckpoints(Path(self.parser.load_dir) / "checkpoints")
        #     if self.parser.load_dir
        #     else None
        # )
        if newest_ck_output:
            true_load_dir = self.checkpoints_path / newest_ck_output
            print("Loading pretrained models from: ", str(true_load_dir))
        elif self.parser.load_dir:
            true_load_dir = Path(self.parser.load_dir)
            print("Loading pretrained models from: ", str(true_load_dir))
        else:
            print("Creating untrained models")
            true_load_dir = None

        # Load generator
        generator = generators.ImplicitGenerator3d(**self.metadata["generator"])

        if true_load_dir:
            checkpoint = torch.load(true_load_dir)
            generator.load_state_dict(checkpoint["generator_state_dict"])
            generator.step = checkpoint["step"]
            # self.update_metadata()
        generator.to(self.device)
        generator_ddp = (
            DDP(generator, device_ids=[self.rank], find_unused_parameters=True)
            if self.parser.ddp
            else generator
        )
        scaler = torch.cuda.amp.GradScaler()
        optimizer_G = torch.optim.Adam(
            generator_ddp.parameters(),
            lr=self.metadata["gen_lr"],
            betas=self.metadata["betas"],
            weight_decay=self.metadata["weight_decay"],
        )
        if true_load_dir:
            optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.generator_ddp = generator_ddp
        self.generator = generator_ddp.module if self.parser.ddp else generator_ddp
        self.optimizer_G = optimizer_G

        # Loading discriminator
        if self.metadata["enable_discriminator"]:
            discriminator = getattr(discriminators, self.metadata["discriminator"])()
            if true_load_dir:
                discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
                discriminator.step = checkpoint["step"]
            discriminator.to(self.device)
            discriminator_ddp = (
                DDP(discriminator, device_ids=[self.rank], find_unused_parameters=True)
                if self.parser.ddp
                else discriminator
            )
            optimizer_D = torch.optim.Adam(
                discriminator_ddp.parameters(),
                lr=self.metadata["disc_lr"],
                betas=self.metadata["betas"],
                weight_decay=self.metadata["weight_decay"],
            )
            if true_load_dir:
                optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
            self.discriminator_ddp = discriminator_ddp
            self.discriminator = (
                discriminator_ddp.module if self.parser.ddp else discriminator_ddp
            )
            self.optimizer_D = optimizer_D

        # Loading encoder
        # encoder = VoxelEncoder(c_dim=16)
        if self.metadata["dataset"]["load_voxel"]:
            encoder = getattr(unet3d, self.metadata["unet"]["type"])(
                **self.metadata["unet"]
            )
        elif self.metadata["dataset"]["load_pcl"]:
            encoder=ResnetPointnet(dim=6, c_dim=self.metadata["generator"]["z_dim"])
        else:
            raise ValueError("Either load pcl or load voxel")

        if true_load_dir:
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
        # else:
        #     encoder.load_state_dict(
        #         torch.load(
        #             "/usr/stud/zhouzh/pi-gan/pretrained_models/PointNet_ONetEncoder.pt"
        #         )
        #     )
        encoder.to(self.device)
        encoder_ddp = (
            DDP(encoder, device_ids=[self.rank], find_unused_parameters=False)
            if self.parser.ddp
            else encoder
        )
        optimizer_E = torch.optim.Adam(
            encoder_ddp.parameters(),
            lr=self.metadata["enc_lr"],
            betas=self.metadata["betas"],
            weight_decay=self.metadata["weight_decay"],
        )
        if true_load_dir:
            optimizer_E.load_state_dict(checkpoint["optimizer_E_state_dict"])
        self.encoder_ddp = encoder_ddp
        self.encoder = encoder_ddp.module if self.parser.ddp else encoder_ddp
        self.optimizer_E = optimizer_E

        self.scaler = scaler

        if true_load_dir:
            self.photometry_losses_val = checkpoint["photometry_losses_val"]
            self.photometry_losses_test = checkpoint["photometry_losses_test"]
            self.depth_losses_val = checkpoint["depth_losses_val"]
            self.depth_losses_test = checkpoint["depth_losses_test"]
        else:

            self.photometry_losses_val = []
            self.depth_losses_val = []
            self.photometry_losses_test = []
            self.depth_losses_test = []

        if self.metadata["photo_loss"]:
            if true_load_dir:
                self.photometry_losses = checkpoint["photometry_losses"]
            else:
                self.photometry_losses = []
        if self.metadata["enable_discriminator"]:
            if true_load_dir:
                self.generator_losses = checkpoint["generator_losses"]
                self.discriminator_losses = checkpoint["discriminator_losses"]
            else:
                self.generator_losses = []
                self.discriminator_losses = []
        if self.metadata["depth_loss"]:
            if true_load_dir:
                self.depth_losses = checkpoint["depth_losses"]
            else:
                self.depth_losses = []

        # if self.parser.set_step != None:
        #     self.generator.step = self.parser.set_step
        self.generator.set_device(self.device)

    def write_train_settings(self):
        """
        Write the training settings to "train_spec.txt" and the current curriculm to curriculum.json.
        """
        if self.rank == 0:
            with open(os.path.join(self.parser.output_dir, "train_spec.txt"), "w") as f:
                f.write(str(self.parser))
                f.write("\n\n")
                f.write(str(self.curriculum))
                f.write("\n\n")
                f.write(str(self.generator))
                f.write("\n\n")
                if self.metadata["enable_discriminator"]:
                    f.write(str(self.discriminator))
                    f.write("\n\n")
                f.write(str(self.encoder))
            with open(
                os.path.join(self.parser.output_dir, "curriculum.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.curriculum, f, ensure_ascii=False, indent=4)

    def save_models(self) -> None:
        """
        Save models at every self.parser.eval_freq steps and at only self.rank=0
        """
        if self.rank == 0 and (self.generator.step + 1) % self.parser.eval_freq == 0:
            step = self.generator.step
            print(
                f"Saving models and optimizers in {self.checkpoints_path} at step {step}"
            )

            ck_dict = {
                "step": step,
                "generator_state_dict": self.generator.state_dict(),
                "optimizer_G_state_dict": self.optimizer_G.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
                "optimizer_E_state_dict": self.optimizer_E.state_dict(),
            }
            if self.metadata["photo_loss"]:
                ck_dict["photometry_losses"] = self.photometry_losses

            ck_dict["photometry_losses_val"] = self.photometry_losses_val
            ck_dict["depth_losses_val"] = self.depth_losses_val
            ck_dict["photometry_losses_test"] = self.photometry_losses_test
            ck_dict["depth_losses_test"] = self.depth_losses_test

            if self.metadata["depth_loss"]:
                ck_dict["depth_losses"] = self.depth_losses

            if self.metadata["enable_discriminator"]:
                ck_dict["discriminator_state_dict"] = self.discriminator.state_dict()
                ck_dict["optimizer_D_state_dict"] = self.optimizer_D.state_dict()
                ck_dict["generator_losses"] = self.generator_losses
                ck_dict["discriminator_losses"] = self.discriminator_losses

            torch.save(
                ck_dict,
                self.checkpoints_path / f"{step}.tar",
            )

    def set_learning_rate(self) -> None:
        """
        Set learning rate according the step at beginning of each epoch
        """
        for param_group in self.optimizer_G.param_groups:
            if param_group.get("name", None) == "mapping_network":
                param_group["lr"] = self.metadata["gen_lr"] * 5e-2
            else:
                param_group["lr"] = self.metadata["gen_lr"]
            param_group["betas"] = self.metadata["betas"]
            param_group["weight_decay"] = self.metadata["weight_decay"]
            
        for param_group in self.optimizer_E.param_groups:
            param_group["lr"] = self.metadata["enc_lr"]
            param_group["betas"] = self.metadata["betas"]
            param_group["weight_decay"] = self.metadata["weight_decay"]

        if self.metadata["enable_discriminator"]:
            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] = self.metadata["disc_lr"]
                param_group["betas"] = self.metadata["betas"]
                param_group["weight_decay"] = self.metadata["weight_decay"]

    def update_metadata(self) -> tuple[dict, bool]:
        """Update metadata according to step number.

        Returns:
            tuple[dict, bool]: The second element indicates if the metadata is changed, i.e. the key steps reached and hyperparameters are updated.
        """
        new_metadata = curriculums.extract_metadata(
            self.curriculum, self.generator.step
        )
        if (
            new_metadata["batch_size"] != self.metadata["batch_size"]
            or new_metadata["img_size"] != self.metadata["img_size"]
        ):
            metadata_updated = True
        else:
            metadata_updated = False
        self.metadata = new_metadata

        return metadata_updated

    def print_stats(self, steps_elapsed_time: float) -> Union[list, None]:
        """
        Print statistics during training every self.parser.print_freq
        """
        if self.rank == 0 and (self.generator.step + 1) % self.parser.print_freq == 0:
            stats = [
                [
                    "Time",
                    "Epoch",
                    "Step",
                    "Sec per step",
                    "Alpha",
                    "Img Size",
                    "Batch Size",
                    "Num steps",
                    # "TopK",
                    "Scale",
                ],
                [
                    datetime.now().strftime("%d--%H:%M"),
                    self.generator.epoch,
                    self.generator.step,
                    round(steps_elapsed_time / self.parser.print_freq, 3),
                    f"{self.alpha:.2f}",
                    self.metadata["img_size"],
                    self.metadata["batch_size"] * self.world_size,
                    self.metadata["num_steps"],
                    # self.topk_num,
                    self.scaler.get_scale(),
                ],
            ]
            if self.metadata["dataset"]["load_voxel"]:
                stats[0].append("Voxel Res.")
                stats[1].append(self.metadata["voxel_resolution"])
            if self.metadata["photo_loss"]:
                stats[0].append("Photo loss")
                stats[1].append(f"{self.photometry_losses[-1]:.4f}")
            if self.photometry_losses_val:
                stats[0].append("Photo loss val")
                stats[1].append(f"{self.photometry_losses_val[-1][1]:.4f}")
            if self.depth_losses_val:
                stats[0].append("Depth loss val")
                stats[1].append(f"{self.depth_losses_val[-1][1]:.4f}")
            if self.photometry_losses_test:
                stats[0].append("Photo loss test")
                stats[1].append(f"{self.photometry_losses_test[-1][1]:.4f}")
            if self.depth_losses_test:
                stats[0].append("Depth loss test")
                stats[1].append(f"{self.depth_losses_test[-1][1]:.4f}")

            if self.metadata["depth_loss"]:
                stats[0].append("Depth loss")
                stats[1].append(f"{self.depth_losses[-1]:.4f}")

            if self.metadata["enable_discriminator"]:
                if self.generator_losses:
                    stats[0].append("Loss_G")
                    stats[1].append(f"{self.generator_losses[-1]:.4f}")
                if self.discriminator_losses:
                    stats[0].append("Loss_D")
                    stats[1].append(f"{self.discriminator_losses[-1]:.4f}")
            print(tabulate(stats, tablefmt="fancy_grid"))
            return stats

    def set_alpha(self):
        step_last_upsample = curriculums.last_upsample_step(
            self.curriculum, self.generator.step
        )
        self.alpha = min(
            1,
            (self.generator.step - step_last_upsample) / (self.metadata["fade_steps"]),
        )
        self.metadata["nerf_noise"] = max(0, 1.0 - self.generator.step / 5000.0)

    # @profile
    def train_generator(self, sample: dict) -> None:
        """
        Train the generator
        """
        imgs = sample["img"].to(self.device, non_blocking=True)
        cam2worlds = sample["cam2world"].to(self.device)
        if self.metadata["dataset"]["load_voxel"]:
            voxels = sample["voxel"].to(self.device)
        if self.metadata["dataset"]["load_pcl"]:
            pcls = sample["pcl"].to(self.device)
        if self.metadata["dataset"]["load_depth"]:
            depths = sample["depth"]
        conds = (
            sample["cond"].to(self.device)
            if self.metadata["dataset"]["cond_mode"]
            else [None] * imgs.shape[0]
        )
        split_batch_size = imgs.shape[0] // self.metadata["batch_split"]
        photometry_loss_subbatch_buffer = 0
        generator_loss_subbatch_buffer = 0
        depth_loss_subbatch_buffer = 0
        for split in range(self.metadata["batch_split"]):
            with torch.cuda.amp.autocast():
                if self.metadata["dataset"]["load_voxel"]:
                    subset_z = self.encoder_ddp(
                        voxels[split * split_batch_size : (split + 1) * split_batch_size]
                    )
                if self.metadata["dataset"]["load_pcl"]:
                    subset_z, z_reg = encode_pcl(
                    self.encoder_ddp,
                    pcls[split * split_batch_size : (split + 1) * split_batch_size],
                    self.device,
                    noise_weight=self.metadata["noise_weight"],
                )
                subset_cam = cam2worlds[
                    split * split_batch_size : (split + 1) * split_batch_size
                ]
                # forward pass
                gen_imgs, gen_depths = self.generator_ddp(
                    subset_z,
                    subset_cam,
                    **self.metadata,
                )
                if self.metadata["enable_discriminator"]:
                    g_preds = self.discriminator_ddp(
                        gen_imgs,
                        self.alpha,
                        cond=conds[
                            split * split_batch_size : (split + 1) * split_batch_size
                        ],
                        **self.metadata,
                    )
                    loss_G = torch.nn.functional.softplus(-g_preds).mean()
                else:
                    loss_G = torch.zeros(1, device=self.device)
                if self.metadata["photo_loss"]:
                    photometry_loss = loss_mse(
                        imgs[split * split_batch_size : (split + 1) * split_batch_size],
                        gen_imgs,
                    )
                else:
                    photometry_loss = torch.zeros(1, device=self.device)

                if self.metadata["depth_loss"]:
                    depth_loss = loss_depth(
                        depths[
                            split * split_batch_size : (split + 1) * split_batch_size
                        ].to(self.device),
                        gen_depths,
                    )
                else:
                    depth_loss = torch.zeros(1, device=self.device)

                if self.metadata["dataset"]["load_pcl"]:
                    z_reg_loss = z_reg * self.metadata["z_reg_weight"]
                    if self.generator.step % 100 == 0 and self.rank == 0:
                        print("z_reg: ", z_reg.item())
                else:
                    z_reg_loss = torch.zeros(1, device=self.device)
                    
                loss = (
                    loss_G
                    + photometry_loss
                    + depth_loss * self.metadata["depth_loss_weight"]
                    + z_reg_loss
                )
                generator_loss_subbatch_buffer += loss_G.item()
                photometry_loss_subbatch_buffer += photometry_loss.item()
                depth_loss_subbatch_buffer += depth_loss.item()

            self.scaler.scale(loss).backward()
        if self.metadata["enable_discriminator"]:
            self.generator_losses.append(
                generator_loss_subbatch_buffer / self.metadata["batch_split"]
            )
        if self.metadata["photo_loss"]:
            self.photometry_losses.append(
                photometry_loss_subbatch_buffer / self.metadata["batch_split"]
            )

        if self.metadata["depth_loss"]:
            self.depth_losses.append(
                depth_loss_subbatch_buffer / self.metadata["batch_split"]
            )

        self.scaler.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(
            self.generator_ddp.parameters(), self.metadata.get("grad_clip", 0.3)
        )
        self.scaler.step(self.optimizer_G)
        self.optimizer_G.zero_grad()

        # if self.generator.step > self.metadata["freeze_encoder_until"]:
        self.scaler.unscale_(self.optimizer_E)
        torch.nn.utils.clip_grad_norm_(
            self.encoder_ddp.parameters(), self.metadata.get("grad_clip", 0.3)
        )
        self.scaler.step(self.optimizer_E)
        self.optimizer_E.zero_grad()

        self.scaler.update()

    def train_discriminator(self, sample: dict) -> None:
        """
        Train the discriminator
        """
        imgs = sample["img"]
        conds = (
            sample["cond"].to(self.device)
            if self.metadata["dataset"]["cond_mode"]
            else [None] * imgs.shape[0]
        )
        split_batch_size = imgs.shape[0] // self.metadata["batch_split"]
        with torch.cuda.amp.autocast():
            # Generate images for discriminator training
            real_imgs = imgs.to(self.device, non_blocking=True)
            if self.metadata["dataset"]["load_voxel"]:
                voxels = sample["voxel"].to(self.device)
            if self.metadata["dataset"]["load_pcl"]:
                pcls = sample["pcl"].to(self.device)
            with torch.no_grad():
                if self.metadata.get("random_gen_img", True):
                    cam_origins = sample_camera_positions(
                        self.device,
                        up_direction="y",
                        cam_r_start=self.metadata["cam_r_start"],
                        cam_r_end=self.metadata["cam_r_end"],
                        n=imgs.shape[0],
                    )
                    cam2worlds = create_cam2world_matrix(cam_origins, "y", self.device)
                else:
                    cam2worlds = sample["cam2world"].to(self.device)
                gen_imgs = []
                gen_positions = []
                for split in range(self.metadata["batch_split"]):
                    if self.metadata["dataset"]["load_voxel"]:
                        subset_z = self.encoder_ddp(
                            voxels[split * split_batch_size : (split + 1) * split_batch_size]
                        )
                    if self.metadata["dataset"]["load_pcl"]:
                        subset_z, z_reg = encode_pcl(
                        self.encoder_ddp,
                        pcls[split * split_batch_size : (split + 1) * split_batch_size],
                        self.device,
                        noise_weight=self.metadata["noise_weight"],
                    )
                    subset_cam = cam2worlds[
                        split * split_batch_size : (split + 1) * split_batch_size
                    ]
                    gen_img, gen_depth = self.generator_ddp(
                        subset_z,
                        subset_cam,
                        **self.metadata,
                    )

                    gen_imgs.append(gen_img)
                    gen_positions.append(gen_depth)

                gen_imgs = torch.cat(gen_imgs, dim=0)
                gen_positions = torch.cat(gen_positions, dim=0)

            real_imgs.requires_grad = True
            r_preds = self.discriminator_ddp(
                real_imgs, self.alpha, cond=conds, **self.metadata
            )

        if self.metadata["r1_lambda"] > 0:
            # Gradient penalty
            grad_real = torch.autograd.grad(
                outputs=self.scaler.scale(r_preds.sum()),
                inputs=real_imgs,
                create_graph=True,
            )
            inv_scale = 1.0 / self.scaler.get_scale()
            grad_real = [p * inv_scale for p in grad_real][0]
        with torch.cuda.amp.autocast():
            if self.metadata["r1_lambda"] > 0:
                grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()
                grad_penalty = 0.5 * self.metadata["r1_lambda"] * grad_penalty
            else:
                grad_penalty = 0

            g_preds = self.discriminator_ddp(
                gen_imgs, self.alpha, cond=conds, **self.metadata
            )

            d_loss = (
                torch.nn.functional.softplus(g_preds).mean()
                + torch.nn.functional.softplus(-r_preds).mean()
                + grad_penalty
            )
            self.discriminator_losses.append(d_loss.item())

        self.optimizer_D.zero_grad()
        self.scaler.scale(d_loss).backward()
        self.scaler.unscale_(self.optimizer_D)
        torch.nn.utils.clip_grad_norm_(
            self.discriminator_ddp.parameters(), self.metadata["grad_clip"]
        )
        self.scaler.step(self.optimizer_D)

    def sample_imgs(self):
        """
        Generate sample images to check the training results
        """
        if (
            self.rank == 0
            and (self.generator.step + 1) % self.parser.sampling_interval == 0
        ):
            assert self.generator.siren.training == False
            print("Inference on known camera poses (trainset):")
            self._sample_from_cameras(self.trainset, self.samples_train_path)
            print("Inference on novel camera poses (valset):")
            self._sample_from_cameras(self.valset, self.samples_val_path)
            print("Inference on unseen cars (testset)")
            self._sample_from_cameras(self.testset, self.samples_test_path)
            print("Compute val losses:")
            self.val_loss(
                self.valset, self.photometry_losses_val, self.depth_losses_val
            )
            print("Compute test losses:")
            self.val_loss(
                self.testset, self.photometry_losses_test, self.depth_losses_test
            )

    def _sample_from_cameras(self, dataset, save_dir, num_imgs=12):
        """Generate a few images conditioned on voxel and with train camera"""
        # TODO: subbatch in case of OOM
        copied_metadata = copy.deepcopy(self.metadata)
        copied_metadata["img_size"] = 128
        copied_metadata["num_steps"] = 48
        copied_metadata["nerf_noise"] = 0
        # copied_metadata["num_points_ratio"] = 1
        copied_metadata["last_back"] = copied_metadata.get("eval_last_back", False)
        if num_imgs > len(dataset):
            num_imgs = len(dataset)
        step = self.generator.step
        rgbs_r = []
        depths_r = []
        rgbs_g = []
        depths_g = []
        # random.seed(0)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i in random.sample(range(len(dataset)), num_imgs):
                    sample = dataset[i]
                    rgbs_r.append((sample["img"] * 0.5) + 0.5)
                    depths_r.append(
                        torch.stack([torch.from_numpy(sample["depth"])] * 3, 0)
                        / copied_metadata["ray_end"]
                    )
                    if copied_metadata["dataset"]["load_voxel"]:
                        voxel = sample["voxel"].unsqueeze(0).to(self.device)
                        z = self.encoder_ddp(voxel)
                    if copied_metadata["dataset"]["load_pcl"]:
                        pcl = sample["pcl"].unsqueeze(0).to(self.device)
                        z, _ = encode_pcl(self.encoder_ddp, pcl, self.device, 0)
                    cam = sample["cam2world"].unsqueeze(0).to(self.device)
                    
                    rgb, depth = generate_img(
                        self.generator,
                        z,
                        cam,
                        copied_metadata,
                    )
                    rgbs_g.append((rgb[0] * 0.5) + 0.5)
                    depths_g.append(depth[0] / copied_metadata["ray_end"])
        save_image(
            torch.stack(rgbs_r + depths_r + rgbs_g + depths_g, 0),
            save_dir / f"{step:06}.png",
            nrow=num_imgs,
            normalize=False,
        )

    def val_loss(self, dataset, photo_loss_ls, depth_loss_ls):
        batch_size = 1
        num_imgs = 48
        # Note here this should be true random, shouldn't be affected by seeds
        # randomly select validation and test data
        # indices = random.sample(
        #     range(len(dataset)), num_imgs if num_imgs < len(dataset) else len(dataset)
        # )

        # use fixed validation and test data for better comparision
        if len(dataset) <= num_imgs:
            indices = list(range(len(dataset)))
        else:
            indices = list(range(num_imgs))
        sampler = TestsetSampler(indices)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            pin_memory=False,
            num_workers=self.metadata["num_workers"],
        )
        copied_metadata = copy.deepcopy(self.metadata)
        copied_metadata["img_size"] = 128
        copied_metadata["num_steps"] = 48
        copied_metadata["nerf_noise"] = 0
        # copied_metadata["num_points_ratio"] = 0.5
        copied_metadata["last_back"] = copied_metadata.get("eval_last_back", False)
        photo_loss = 0
        depth_loss = 0

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for idx, sample in enumerate(dataloader):
                    imgs = sample["img"]
                    cams = sample["cam2world"]
                    depths = sample["depth"]
                    if copied_metadata["dataset"]["load_voxel"]:
                        voxel = sample["voxel"].to(self.device)
                        z = self.encoder_ddp(voxel)
                    if copied_metadata["dataset"]["load_pcl"]:
                        pcl = sample["pcl"].to(self.device)
                        z, _ = encode_pcl(self.encoder_ddp, pcl, self.device, 0)
                    gen_imgs, gen_depths = self.generator_ddp(
                        z, cam2worlds=cams.to(self.device), **copied_metadata
                    )
                    photo_loss += loss_mse(imgs, gen_imgs.cpu()).item()
                    depth_loss += loss_depth(depths, gen_depths.cpu()).item()
                    # if idx >= 25:
                    #     break
        photo_loss /= len(dataloader)
        depth_loss /= len(dataloader)
        photo_loss_ls.append([self.generator.step, photo_loss])
        depth_loss_ls.append([self.generator.step, depth_loss])

    def evaluate(self) -> None:
        """
        Evaluate the generator (i.e. compute fid score, output many generated images at self.evaluation_path) every self.parser.eval_freq steps
        """
        if (self.generator.step + 1) % self.parser.eval_freq == 0:

            copied_metadata = copy.deepcopy(self.metadata)
            copied_metadata["img_size"] = 128
            copied_metadata["num_steps"] = 96
            copied_metadata["nerf_noise"] = 0
            
            if self.rank == 0:
                real_dir = fid_evaluation.setup_evaluation(copied_metadata)
            if self.parser.ddp:
                dist.barrier()
            fid_evaluation.output_images(
                self.generator_ddp,
                self.encoder_ddp,
                self.fidset,
                copied_metadata,
                self.rank,
                self.world_size,
                str(self.evaluation_path),
                self.device,
            )
            if self.parser.ddp:
                dist.barrier()
            if self.rank == 0:
                # fid = fid_evaluation.calculate_fid(real_dir, str(self.evaluation_path))
                fid, ofid, lpips_score, psnr = compute_metrics_from_dirs(
                    real_dir,
                    str(self.evaluation_path),
                    Path(copied_metadata["dataset"]["path"]).parent / "val.lst",
                    self.device,
                )
                with open(os.path.join(self.parser.output_dir, f"fid.txt"), "a") as f:
                    f.write(
                        f"\n{self.generator.step} {fid} {ofid} {lpips_score} {psnr}"
                    )
                print(
                    f"Metrics at step {self.generator.step}: FID {fid} | oFID {ofid} | LPIPS {lpips_score} | PNSR {psnr}"
                )
            torch.cuda.empty_cache()
