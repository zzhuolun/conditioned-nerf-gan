# ----------------------------------------------------
# Inference the trained model on the specified dataset.
# The script can:
#   1) generate images on the perturbed train views: python inference.py MODEL_ABSOLUTE_PATH.tar --images --cam_fixed
#   2) generate images on spherical uniformally distributed novel views: python inference.py MODEL_ABSOLUTE_PATH.tar --images
#   3) rendering a video: python inference.py MODEL_ABSOLUTE_PATH.tar --video
#   4) extract voxel: python inference.py MODEL_ABSOLUTE_PATH.tar --voxel
# Of course those flags can be combined and do multiple tasks at once.
# ----------------------------------------------------
import argparse
import copy
import json
from json import encoder
import random
from datetime import datetime
from pathlib import Path
import os
import mrcfile
import numpy as np
import torch
from torchvision.utils import save_image

import configs
from configs import curriculums
from datasets import ShapeNet, readEXR
from extract_shapes import sample_generator
from generators.volumetric_rendering import (
    create_cam2world_matrix,
    sample_camera_positions,
)
from utils import generate_img, jsonKeys2int
import cv2 as cv
from generators import generators, siren, unet3d
from tqdm import tqdm
from typing import Union
import fid_evaluation
from metric_utils import compute_metrics_from_dirs
from feature_volume.voxel2img import voxel_surface_render


def load_voxel(voxel_path: str):
    return (
        torch.from_numpy(np.load(voxel_path)["voxel"])
        .permute(3, 2, 1, 0)
        .type(torch.float)
    )


def tensor_to_cv(img: torch.Tensor) -> np.ndarray:
    img = (
        img.squeeze() * 0.5 + 0.5
    )  # unnormalized, according to the normalization in dataset loading
    img = (
        img.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def perturb_extrinsics(P_batch: torch.Tensor, noise_weight: float) -> torch.Tensor:
    """Perturb the camera rotation and translation with noise using rodrigues formula."""
    P_batch = P_batch.numpy()
    new_P_batch = []
    for P in P_batch:
        assert P.shape[0] == 4
        assert P.shape[1] == 4
        R = P[:3, :3]
        t = P[:3, -1]
        rvec = cv.Rodrigues(R)[0]
        new_rvec = rvec + np.random.randn(*rvec.shape) * noise_weight
        new_R = cv.Rodrigues(new_rvec)[0]
        new_t = torch.from_numpy(t + np.random.randn(*t.shape) * noise_weight)
        new_t = clip_norm(new_t).numpy()

        # aim the camera to the origin
        new_t = -1 * new_R[:, -1] * np.linalg.norm(new_t)
        assert 0.69 <= np.linalg.norm(new_t) <= 1.51
        new_P = np.eye(4)
        new_P[:3, :3] = new_R
        new_P[:3, -1] = new_t
        new_P_batch.append(new_P)
    return torch.from_numpy(np.stack(new_P_batch)).type(torch.float32)


class Inferencer:
    """class for inference trained model"""

    def __init__(
        self,
        checkpoint_dir: str,
        colored_pcl: bool,
        sampling_mode: str,
        input_voxel_resolution,
        z_up: bool,
        num_imgs: int = 12,
        image_size: int = 128,
        ray_step_multiplier: int = 2,
        load_dataset: bool = True,
    ) -> None:
        """
        Args:
            checkpoint_dir (str): directory to the checkpoint.
            z_up (bool): If true, upward direction in the world space is z; else, upward direction is y.
            num_imgs (int, optional): number of latent encodes to evaluate on. Defaults to 12.
            image_size (int, optional): size of each single rendered image. Defaults to 128.
            ray_step_multiplier (int, optional): multiplier of the number of steps on the NeRF ray. Defaults to 2.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_imgs = num_imgs
        self.image_size = image_size
        self.ray_step_multiplier = ray_step_multiplier
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(checkpoint_dir)
        self.curriculum = self._get_curriculum(
            int(self.checkpoint_dir.stem), input_voxel_resolution
        )
        self.colored_pcl = colored_pcl
        print(self.curriculum)
        self._set_models()
        if load_dataset:
            self.valset = ShapeNet(
                self.curriculum["dataset"]["path"],
                img_size=image_size,
                sampling_mode=sampling_mode,
                metadata=self.curriculum,
                load_voxel=self.curriculum["dataset"]["load_voxel"],
                load_pcl=self.curriculum["dataset"]["load_pcl"],
                dataset_mode="test",
                shuffle=True,
                multifold=1,
                voxel_resolution=self.curriculum["voxel_resolution"],
            )
        if z_up:
            self.up_direction = "z"
        else:
            self.up_direction = "y"

    def _get_curriculum(self, step, input_voxel_resolution) -> dict:
        """Get the curriclum dict for inference

        Returns:
            dict: the curriculum dictionary
        """
        if (self.checkpoint_dir.parent.parent / "curriculum.json").exists():
            with open(self.checkpoint_dir.parent.parent / "curriculum.json", "r") as f:
                print(
                    "Loading curriculum from",
                    str(self.checkpoint_dir.parent.parent / "curriculum.json"),
                )
                curriculum = json.load(f, object_hook=jsonKeys2int)
        else:
            print("Creating curriculum from configs/default.py and configs/special.py")
            curriculum = curriculums.update_recursive(configs.default, configs.special)
        curriculum = curriculums.extract_metadata(curriculum, step)
        # input voxel resolution
        curriculum["num_workers"] = 4
        curriculum["voxel_resolution"] = input_voxel_resolution
        curriculum["num_steps"] = int(48 * self.ray_step_multiplier)
        curriculum["img_size"] = self.image_size
        curriculum["psi"] = 1
        curriculum["v_stddev"] = 0
        curriculum["h_stddev"] = 0
        curriculum["last_back"] = curriculum.get("eval_last_back", False)
        curriculum["nerf_noise"] = 0
        # curriculum["num_points_ratio"] = 1
        curriculum["noise_weight"] = 0
        curriculum = {
            key: value for key, value in curriculum.items() if type(key) is str
        }
        return curriculum

    def _set_models(self) -> None:
        """Load the generator, and possibly encoder and ema model, from the checkpoint"""
        generator = generators.ImplicitGenerator3d(**self.curriculum["generator"])
        generator.load_state_dict(self.checkpoint["generator_state_dict"])
        generator.step = self.checkpoint["step"]
        generator.to(self.device)
        generator.set_device(self.device)
        generator.eval()
        self.generator = generator
        # encoder = VoxelEncoder(c_dim=16)
        encoder = getattr(unet3d, self.curriculum["unet"]["type"])(
            **self.curriculum["unet"]
        )
        encoder.load_state_dict(self.checkpoint["encoder_state_dict"])
        encoder.to(self.device)
        self.encoder = encoder

    def render_imgs(self, is_fixed: bool) -> None:
        if is_fixed:
            print("Inference with fixed camera")
            self._inference_fixed_camera()
        else:
            print("Inference with uniform spherical camera")
            self.curriculum["sample_dist"] = "spherical_uniform"
            self._inference_random_camera()

    def _inference_fixed_camera(self) -> None:
        """Sample a few generated images conditioned on pcl and with fixed camera"""
        valset = self.valset
        assert len(valset) >= self.num_imgs
        img_ls = []
        cam_ls = []
        depth_ls = []
        pcl_ls = []
        for i in tqdm(random.sample(range(len(valset)), self.num_imgs)):
            sample = valset[i]
            img = sample["img"]
            cam = sample["cam2world"]
            depth = torch.from_numpy(sample["depth"])
            pcl = sample["voxel"]
            img_ls.append((sample["img"] * 0.5) + 0.5)
            cam_ls.append(cam)
            depth_ls.append(torch.stack([depth] * 3, 0))
            pcl_ls.append(pcl)
        img = torch.stack(img_ls)
        cam = torch.stack(cam_ls)
        depth = torch.stack(depth_ls)
        pcl = torch.stack(pcl_ls)
        # generated_imgs = [img, depth]
        generated_imgs = []
        max_imgs_per_batch = 1
        split_size = self.num_imgs // max_imgs_per_batch  # Note here num_imgs >=8
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for noise2cam in [0]:#[0, 0.05, 0.1, 0.5, 1, 2]:
                    cam_perturbed = perturb_extrinsics(cam, noise2cam)
                    rgbs = []
                    depths = []
                    for split in range(split_size):
                        subset_cam = cam_perturbed[
                            split
                            * max_imgs_per_batch : (split + 1)
                            * max_imgs_per_batch
                        ]
                        subset_pcl = pcl[
                            split
                            * max_imgs_per_batch : (split + 1)
                            * max_imgs_per_batch
                        ]
                        z = self.encoder(subset_pcl.to(self.device))
                        rgb, depth = generate_img(
                            self.generator,
                            z,
                            subset_cam.to(self.device),
                            self.curriculum,
                        )
                        rgbs.append((rgb * 0.5) + 0.5)
                        depths.append(depth / self.curriculum["ray_end"])
                    generated_imgs.append(torch.cat(rgbs, 0))
                    generated_imgs.append(torch.cat(depths, 0))
        img_name = (
            self.checkpoint_dir.parent.parent
            / "inference"
            / f"{self.checkpoint_dir.stem}_fixed_{datetime.now().strftime('%d--%H:%M:%S.%f')}.png"
        )
        img_name.parent.mkdir(exist_ok=True)
        save_image(
            torch.cat(generated_imgs),  # Bx3X128x128
            img_name,
            nrow=self.num_imgs,
            normalize=False,
            padding =0,
        )
        print("Save image to ", img_name)

    def _inference_random_camera(self) -> None:
        """Generate inferenced images with random camera poses."""
        with torch.no_grad():
            cam_origins = sample_camera_positions(
                self.device,
                self.up_direction,
                self.curriculum["cam_r_start"],
                self.curriculum["cam_r_end"],
                self.num_imgs,
            )
            cam = create_cam2world_matrix(cam_origins, self.up_direction, self.device)
            valset = self.valset
            assert len(valset) >= self.num_imgs
            img_ls = []
            depth_ls = []
            pcl_ls = []
            torch.manual_seed(0)
            # for i in random.sample(range(len(valset)), self.num_imgs):
            for i in range(self.num_imgs):
                sample = valset[i]
                pcl = sample["voxel"]
                img_ls.append((sample["img"] * 0.5) + 0.5)
                depth_ls.append(
                    torch.stack([torch.from_numpy(sample["depth"])] * 3, 0)
                    / self.curriculum["ray_end"]
                )
                pcl_ls.append(pcl)
            img = torch.stack(img_ls)
            depth = torch.stack(depth_ls)
            pcl = torch.stack(pcl_ls)
            generated_imgs = [img, depth]
            max_imgs_per_batch = 1
            split_size = self.num_imgs // max_imgs_per_batch  # Note here num_imgs >=8
            assert len(img_ls) >= max_imgs_per_batch
            with torch.cuda.amp.autocast():
                rgbs = []
                depths = []
                for split in range(split_size):
                    subset_cam = cam[
                        split * max_imgs_per_batch : (split + 1) * max_imgs_per_batch
                    ]
                    subset_pcl = pcl[
                        split * max_imgs_per_batch : (split + 1) * max_imgs_per_batch
                    ]
                    z = self.encoder(subset_pcl.to(self.device))
                    rgb, depth = generate_img(
                        self.generator,
                        z,
                        subset_cam.to(self.device),
                        self.curriculum,
                    )
                    rgbs.append((rgb * 0.5) + 0.5)
                    depths.append(depth / self.curriculum["ray_end"])
                generated_imgs.append(torch.cat(rgbs, 0))
                generated_imgs.append(torch.cat(depths, 0))
        img_name = (
            self.checkpoint_dir.parent.parent
            / "inference"
            / f"{self.checkpoint_dir.stem}_random_{datetime.now().strftime('%d--%H:%M:%S.%f')}.png"
        )
        img_name.parent.mkdir(exist_ok=True)
        save_image(
            torch.cat(generated_imgs),  # Bx3X128x128
            img_name,
            nrow=self.num_imgs,
            normalize=False,
            pad_value =0,
        )
        print("Save image to ", img_name)

    def save_voxel(self, pcl_name, cube_size, voxel_resolution) -> None:
        """Extract, save and plot the voxel from latent code"""
        # Might have bug

        if isinstance(pcl_name, int):
            pcl = self.valset[pcl_name]["voxel"].unsqueeze(0)
            car_name = Path(self.valset[pcl_name]["img_path"]).parent.parent.stem
        else:
            pcl = (
                torch.from_numpy(
                    np.load(
                        Path(self.curriculum["dataset"]["path"])
                        / pcl_name
                        / "voxel_32.npz"
                    )["voxel"]
                )
                .permute(3, 2, 1, 0)
                .type(torch.float32)
                .unsqueeze(0)
            )
            car_name = pcl_name

        # pcl = self.valset[pcl_idx]["voxel"].unsqueeze(0)
        # car_name = Path(self.valset[pcl_idx]["img_path"]).parent.parent.stem
        with torch.no_grad():
            z = self.encoder(pcl.to(self.device))
        voxel_grid = sample_generator(
            self.generator, z, cube_length=cube_size, voxel_resolution=voxel_resolution
        )
        voxel_name = (
           self.checkpoint_dir.parent.parent
            / "inference"
            / f"{car_name}.mrc"
        )
        voxel_name.parent.mkdir(exist_ok=True)
        with mrcfile.new_mmap(
            voxel_name,
            overwrite=True,
            shape=voxel_grid.shape,
            mrc_mode=2,
        ) as mrc:
            mrc.data[:] = voxel_grid
        print("Save voxel to ", voxel_name)
        # plot_mrc(voxel_name, thres=0)

    def render_video(
        self,
        pcl_name: Union[int, str],
        num_frames: int,
        fps: int,
    ) -> None:
        """Render video of 256*256 size

        Args:
            pcl_name : chooses which car's pcl to encode
            num_frames : number of frames to render. Note that fps is 24
            fps: frame per second
        """
        assert num_frames > fps * 4
        copied_metadata = copy.deepcopy(self.curriculum)
        # copied_metadata["img_size"] = 256
        assert num_frames % 4 == 0
        if isinstance(pcl_name, int):
            pcl = self.valset[pcl_name]["voxel"].unsqueeze(0)
            car_name = Path(self.valset[pcl_name]["img_path"]).parent.parent.stem
        else:
            pcl = (
                torch.from_numpy(
                    np.load(
                        Path(copied_metadata["dataset"]["path"])
                        / pcl_name
                        / "voxel.npz"
                    )["voxel"]
                )
                .permute(3, 2, 1, 0)
                .type(torch.float32)
                .unsqueeze(0)
            )
            car_name = pcl_name

        print("Rendering video for car ", car_name)
        print("Video Resolution:", copied_metadata["img_size"])
        frames = []
        output_name = (
            self.checkpoint_dir.parent.parent
            / "inference"
            / "video"
            / f"{str(self.checkpoint_dir.parent.parent.stem)}_{self.checkpoint_dir.stem}_{car_name}_{datetime.now().strftime('%d--%H:%M:%S')}.mp4"
        )
        output_name.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv.VideoWriter_fourcc(*"MPEG")
        video = cv.VideoWriter(
            str(output_name),
            fourcc,
            fps,
            (copied_metadata["img_size"], copied_metadata["img_size"]),
        )

        with torch.no_grad():
            z = self.encoder(pcl.to(self.device))
            theta0 = np.linspace(1e-5, np.pi / 2 - 1e-5, num_frames // 2)
            phi0 = np.linspace(0, np.pi * 2, num_frames // 2)

            theta1 = np.linspace(np.pi / 2 - 1e-5, 1e-5, num_frames // 4)
            phi11 = np.linspace(np.pi * 2, np.pi * 5 / 4, fps)
            phi12 = np.asarray([np.pi * 5 / 4] * (num_frames // 4 - fps))

            theta21 = np.linspace(1e-5, np.pi / 4 - 1e-5, fps)
            theta22 = np.asarray([np.pi / 4 - 1e-5] * (num_frames // 4 - fps))
            phi2 = np.linspace(np.pi * 5 / 4, 0, num_frames // 4)

            theta = np.concatenate([theta0, theta1, theta21, theta22], axis=0)
            phi = np.concatenate([phi0, phi11, phi12, phi2], axis=0)

            r = np.linspace(
                copied_metadata["cam_r_start"], copied_metadata["cam_r_end"], num_frames
            )
            fov = np.linspace(60, 30, num_frames)
            cam_origin = np.zeros((num_frames, 3))
            cam_origin[:, 0] = r * np.sin(theta) * np.cos(phi)
            if self.up_direction == "z":
                cam_origin[:, 1] = r * np.sin(theta) * np.sin(phi)
                cam_origin[:, 2] = r * np.cos(theta)
            elif self.up_direction == "y":
                cam_origin[:, 2] = r * np.sin(theta) * np.sin(phi)
                cam_origin[:, 1] = r * np.cos(theta)

            # View from below the cars
            # cam_origin[:,1] *= -1

            cam_origin = (
                torch.from_numpy(cam_origin).type(torch.float32).to(self.device)
            )
            cam2world = create_cam2world_matrix(
                cam_origin, self.up_direction, self.device
            )
            for idx in tqdm(range(num_frames)):
                copied_metadata["fov"] = fov[idx]
                imgs, depth_map = generate_img(
                    self.generator,
                    z,
                    cam2world[idx].unsqueeze(0),
                    copied_metadata,
                )
                frames.append(imgs)
            frames = torch.cat(frames, dim=0)
            # save_image(
            #     frames,  # Bx3X128x128
            #     output_name.parent
            #     / f"{self.checkpoint_dir.stem}_{car_name}_{datetime.now().strftime('%d--%H:%M:%S')}.jpg",
            #     nrow=int(np.sqrt(num_frames)),
            #     normalize=True,
            # )
            for frame in frames:
                video.write(tensor_to_cv(frame))
            cv.destroyAllWindows()
            video.release()
        print("Writing video to ", str(output_name))

    def render_pcl(self, pcl_idx: int, num_views: int) -> None:
        """Inference and render a colored point cloud from randomly sampled views.
        The point cloud is saved as .obj file and can be easily views by MeshLab.

        Args:
            pcl_idx : chooses which car's pcl to encode
            num_views (int): number of views.
            mask (bool): if true, mask the reprojected depth with grount truth depth map.
        """
        img_size = self.curriculum["img_size"]
        pcl = self.valset[pcl_idx]["voxel"].unsqueeze(0)
        car_name = Path(self.valset[pcl_idx]["img_path"]).parent.parent.stem
        print("Rendering point clouds for car ", car_name)
        pts_color = []
        pts_xyz = []
        rgbs = []
        depth_maps = []
        with torch.no_grad():
            z = self.encoder(pcl.to(self.device))
            cam_origins = sample_camera_positions(
                self.device,
                self.up_direction,
                self.curriculum["cam_r_start"],
                self.curriculum["cam_r_end"],
                num_views,
            )
            cam2worlds = create_cam2world_matrix(
                cam_origins, self.up_direction, self.device
            )
            for cam in cam2worlds:
                img, depth = self.generator(
                    z,
                    cam2worlds=cam.unsqueeze(0),
                    **self.curriculum,
                )
                img = img.cpu()
                depth_map = torch.stack([depth.cpu()] * 3, 1)
                rgbs.append(img)
                depth_maps.append(depth_map)

                img = img.squeeze().permute(1, 2, 0)
                depth = depth.cpu().squeeze()
                valid_idx = torch.logical_and(
                    depth > self.curriculum["ray_start"],
                    depth < self.curriculum["ray_end"],
                )
                pts_color.append(img[torch.where(valid_idx)])
                pts = torch.ones(valid_idx.sum(), 4)
                yx = (
                    (2 * torch.nonzero(valid_idx) - (img_size - 1))
                    / (img_size - 1)
                    / 2.1875
                    * depth[torch.where(valid_idx)].unsqueeze(1)
                )
                pts[:, 0] = yx[:, 1]
                pts[:, 1] = yx[:, 0]
                pts[:, 2] = depth[torch.where(valid_idx)]
                pts_world = (pts @ cam.cpu().T)[:, :3]
                pts_xyz.append(pts_world)
        pts_color = torch.cat(pts_color, dim=0)
        # unnormalize the point color
        pts_color = pts_color * 0.5 + 0.5
        pts_color = pts_color.mul(255).add_(0.5).clamp(0, 255)

        pts_xyz = torch.cat(pts_xyz, dim=0)
        points = torch.cat([pts_xyz, pts_color], dim=1).numpy()
        pcl_name = (
            self.checkpoint_dir.parent.parent
            / "inference"
            / f"{self.checkpoint_dir.stem}_{car_name}_{datetime.now().strftime('%d--%H:%M:%S')}.obj"
        )
        pcl_name.parent.mkdir(exist_ok=True)
        with open(pcl_name, "w+") as f:
            for i in range(points.shape[0]):
                color = points[i, 3:].astype("uint8")
                f.write(
                    "v "
                    + str(points[i, 0])
                    + " "
                    + str(points[i, 1])
                    + " "
                    + str(points[i, 2])
                    + " "
                    + str(color[0])
                    + " "
                    + str(color[1])
                    + " "
                    + str(color[2])
                    + "\n"
                )
        print("Saving pcl to ", str(pcl_name))

        img_name = (
            self.checkpoint_dir.parent.parent
            / "inference"
            / f"{self.checkpoint_dir.stem}_pcl_{datetime.now().strftime('%d--%H:%M:%S')}.png"
        )
        save_image(
            torch.cat([*rgbs, *depth_maps]), img_name, nrow=num_views, normalize=True
        )
        print("Rgb and depth image saved at ", img_name)

    def render_pcl_masked(self, num_views: int, mask: bool = True) -> None:
        """Inference and render a colored point cloud from existing multiple views.
        The point cloud is saved as .obj file and can be easily views by MeshLab.

        Args:
            num_views (int): number of views.
            mask (bool): if true, mask the reprojected depth with grount truth depth map.
        """
        self.curriculum["sample_dist"] = "fixed"
        img_size = self.curriculum["img_size"]

        fp = Path(
            "/usr/stud/zhouzh/stanford-shapenet-renderer/shapenet_test/baa1e44e6e086c233e320a6703976361"
        )
        cams = np.load(fp / "cameras.npz")
        views = random.sample(range(100), num_views)
        pts_color = []
        pts_xyz = []
        with torch.no_grad():
            for idx in views:
                depth_gt = readEXR(
                    str(fp / "depth" / (f"{idx:04d}" + "0001.exr")),
                    img_size,
                )
                depth_gt = torch.from_numpy(depth_gt)
                cam2world = torch.from_numpy(cams[f"world_mat_inv_{idx}"]).type(
                    torch.float32
                )
                # cam2world[:, 1] *= -1
                # cam2world[:, 2] *= -1
                img, depth = self.generator(
                    cam2worlds=cam2world.unsqueeze(0).to(self.device),
                    **self.curriculum,
                )
                assert depth.squeeze().shape == depth_gt.shape
                img = img.squeeze().permute(1, 2, 0).cpu()
                depth = depth.squeeze().cpu()
                valid_idx = torch.logical_and(
                    depth > self.curriculum["ray_start"],
                    depth < self.curriculum["ray_end"],
                )
                if mask:
                    valid_idx = torch.logical_and(
                        valid_idx,
                        depth_gt > 1e-4,
                    )
                pts_color.append(img[torch.where(valid_idx)])
                pts = torch.ones(valid_idx.sum(), 4)
                yx = (
                    (2 * torch.nonzero(valid_idx) - (img_size - 1))
                    / (img_size - 1)
                    / 2.1875
                    * depth[torch.where(valid_idx)].unsqueeze(1)
                )
                pts[:, 0] = yx[:, 1]
                pts[:, 1] = yx[:, 0]
                pts[:, 2] = depth[torch.where(valid_idx)]

                # cam2world[:, 1] *= -1
                # cam2world[:, 2] *= -1

                pts_world = (pts @ cam2world.T)[:, :3]
                pts_xyz.append(pts_world)
        pts_color = torch.cat(pts_color, dim=0)
        pts_xyz = torch.cat(pts_xyz, dim=0)
        points = torch.cat([pts_xyz, pts_color], dim=1).numpy()
        pcl_name = (
            self.checkpoint_dir.parent.parent
            / "inference"
            / f"{self.checkpoint_dir.stem}_{datetime.now().strftime('%d--%H:%M:%S')}.obj"
        )
        pcl_name.parent.mkdir(exist_ok=True)
        with open(pcl_name, "w+") as f:
            for i in range(points.shape[0]):
                color = (points[i, 3:] * 255.0).astype("uint8")
                f.write(
                    "v "
                    + str(points[i, 0])
                    + " "
                    + str(points[i, 1])
                    + " "
                    + str(points[i, 2])
                    + " "
                    + str(color[0])
                    + " "
                    + str(color[1])
                    + " "
                    + str(color[2])
                    + "\n"
                )
        print("Saving pcl to ", str(pcl_name))

    def compute_metrics(
        self,
    ):
        evaluation_path = self.checkpoint_dir.parent.parent / (
            f"evaluation_{self.curriculum['voxel_resolution']}_{self.curriculum['num_steps']}_{self.generator.step}_128"
        )
        evaluation_path.mkdir(exist_ok=True)
        real_dir = fid_evaluation.setup_evaluation(self.curriculum)
        fid_evaluation.output_images(
            self.generator,
            self.encoder,
            self.valset,
            self.curriculum,
            0,
            1,
            str(evaluation_path),
            self.device,
        )
        fid, ofid, lpips_score, psnr = compute_metrics_from_dirs(
            real_dir,
            str(evaluation_path),
            Path(self.curriculum["dataset"]["path"]).parent / "val.lst",
            self.device,
        )
        with open(
            self.checkpoint_dir.parent.parent / f"fid_inference.txt",
            "a",
        ) as f:
            f.write(
                f"{self.generator.step} {self.curriculum['voxel_resolution']} {self.curriculum['num_steps']} {fid} {ofid} {lpips_score} {psnr}\n"
            )
        print(
            f"Metrics at step {self.generator.step}: FID {fid} | oFID {ofid} | LPIPS {lpips_score} | PNSR {psnr}"
        )
        torch.cuda.empty_cache()

    def interpolate_latent_codes(
        self,
        voxel_src_id: str,
        voxel_tgt_id: str,
        save_dir: str,
        mode: str,
    ):
        voxel_src_path = (
            Path(self.curriculum["dataset"]["path"]) / voxel_src_id / "voxel.npz"
        )
        voxel_tgt_path = (
            Path(self.curriculum["dataset"]["path"]) / voxel_tgt_id / "voxel.npz"
        )
        voxel_src = load_voxel(voxel_src_path)
        voxel_tgt = load_voxel(voxel_tgt_path)
        generated_imgs = []
        voxel_render_imgs = []
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                fv_src, z_src = self.encoder(voxel_src.unsqueeze(0).to(self.device))
                fv_tgt, z_tgt = self.encoder(voxel_tgt.unsqueeze(0).to(self.device))

                # cam_origins = sample_camera_positions(
                #     self.device,
                #     self.up_direction,
                #     1.0,
                #     1.1,
                #     self.num_imgs,
                # )
                # cams = create_cam2world_matrix(
                #     cam_origins, self.up_direction, self.device
                # )
                cameras_npz = np.load('/storage/user/yenamand/one/differentiable_volumetric_rendering/data/ShapeNet/02958343/38ca3e0a0763b54ad98ce7833bb35f18/cameras.npz')
                cams = torch.from_numpy(
                    cameras_npz["world_mat_inv_20"]
                ).type(torch.float32).unsqueeze(0)
                
                interpolate_n = 8
                for i in range(interpolate_n + 1):
                    z_it = (1 - i / interpolate_n) * z_src + i / interpolate_n * z_tgt
                    # z_it = (
                    #     1 - i / interpolate_n
                    # ) * z_src + i / interpolate_n * torch.randn_like(z_src)
                    fv_it = (
                        1 - i / interpolate_n
                    ) * fv_src + i / interpolate_n * fv_tgt
                    rgbs = []
                    depths = []
                    voxel_imgs = []
                    # if mode == "fz":
                    #     fvz_it = (fv_it, z_it)
                    # elif mode == "f":
                    #     fvz_it = (fv_it, z_src)
                    # elif mode == "z":
                    #     fvz_it = (fv_src, z_it)

                    for cam in cams:
                        rgb, depth = generate_img(
                            self.generator,
                            (fv_it, z_it),
                            cam.unsqueeze(0).to(self.device),
                            self.curriculum,
                        )
                        rgbs.append((rgb * 0.5) + 0.5)
                        depths.append(depth / self.curriculum["ray_end"])
                    #     voxel_it = (
                    #         1 - i / interpolate_n
                    #     ) * voxel_src + i / interpolate_n * voxel_tgt
                    #     rgbs_voxelrender = voxel_render(
                    #         voxel_it.unsqueeze(0).to(self.device),
                    #         cam.unsqueeze(0),
                    #         128,
                    #         self.curriculum["fov"],
                    #         0.2,
                    #         2.0,
                    #         256,
                    #         self.device,
                    #     )
                    #     voxel_imgs.append(rgbs_voxelrender)
                    # voxel_render_imgs += voxel_imgs
                    generated_imgs += rgbs
                    # generated_imgs.append(torch.cat(depths, 0))
        output_imgs = []
        for i in range(self.num_imgs):
            for j in range(interpolate_n + 1):
                output_imgs.append(generated_imgs[j * self.num_imgs + i])
        # output_voxel_imgs = []
        # for i in range(self.num_imgs):
        #     for j in range(interpolate_n + 1):
        #         output_voxel_imgs.append(voxel_render_imgs[j * self.num_imgs + i].cpu())
        save_name = Path(save_dir) / (
            voxel_src_id + "_to_" + voxel_tgt_id + "_" + mode + ".png"
        )
        save_name.parent.mkdir(exist_ok=True)
        print("saving image to", str(save_name))
        save_image(
            torch.cat(output_imgs), save_name, nrow=interpolate_n + 1, normalize=False, padding=0
        )
        # save_image(
        #     torch.cat(output_voxel_imgs),
        #     Path(save_dir) / (voxel_src_id + "_to_" + voxel_tgt_id + "_voxel.png"),
        #     nrow=interpolate_n + 1,
        #     normalize=False,
        # )


def clip_norm(x: torch.Tensor, lower=0.7, upper=1.5):
    """clip the norm of input tensor x to the range of [lower, upper]"""
    denom = x.norm(dim=-1, keepdim=True)
    denom_clip = torch.clamp(denom, lower, upper)
    return x / denom * denom_clip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="absolute path to the checkpoint .tar",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="val",
        help="Which subset of shapnetcars to sample from, chosen from (single|twenty|twohunderd|thousand, etc",
    )
    # NeRF model settings:
    parser.add_argument(
        "--image_size", type=int, default=128, help="size of generated images"
    )
    parser.add_argument(
        "--ray_step_multiplier",
        type=int,
        default=1,
        help="multiplier for the ray steps of renderer",
    )
    parser.add_argument(
        "--z_up",
        action="store_true",
        help="If true, upward direction is z; else upword direction is y.",
    )
    parser.add_argument(
        "--colored_pcl",
        action="store_true",
        help="If true, the input dimension of encoder is 6; else the input dimension is 3.",
    )
    # Generating images setting:
    parser.add_argument(
        "--images",
        action="store_true",
        help="If true, inference images based on different camera sampling dist.",
    )
    parser.add_argument(
        "--num_imgs", type=int, default=8, help="number of images/cars to generate"
    )
    parser.add_argument(
        "--cam_fixed",
        action="store_true",
        help="If true, use fixed camera pose from dataset; else, use uniform spherical camera",
    )
    # Rendering video settings:
    parser.add_argument(
        "--video",
        action="store_true",
        help="If true, store video; else, generate images based on camera sampling mode.",
    )
    parser.add_argument(
        "--num_frames",
        default=240,
        type=int,
        help="number of frames to generate the video.",
    )
    # Generating pcl settings:
    parser.add_argument(
        "--pcl", action="store_true", help="If true, render pcl from different poses."
    )
    parser.add_argument(
        "--pcl_views", default=20, type=int, help="number of views used to render pcl"
    )
    # Generating voxel settings:
    parser.add_argument(
        "--voxel",
        action="store_true",
        help="if true, also generates extracted voxel",
    )
    parser.add_argument(
        "--cube_size",
        type=float,
        default=1.2,
        help="the size/length of the voxel cube.",
    )
    parser.add_argument(
        "--voxel_resolution", type=int, default=48, help="the output voxel resolution."
    )
    parser.add_argument(
        "--metrics", action="store_true", help="If true, compute metrics."
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="If true, interploate between voxel latent codes.",
    )
    parser.add_argument("--seeds", nargs="+", default=[0])
    opt = parser.parse_args()
    print(opt)

    input_voxel_resolution = 64
    load_dataset = False if opt.interpolate else True

    for seed in tqdm(opt.seeds):
        seed = int(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        inferencer = Inferencer(
            opt.path,
            opt.colored_pcl,
            opt.sampling_mode,
            input_voxel_resolution,
            opt.z_up,
            opt.num_imgs,
            opt.image_size,
            opt.ray_step_multiplier,
            load_dataset,
            )
        if opt.images:
            print("Starting generating images:")
            inferencer.render_imgs(is_fixed=opt.cam_fixed)

        pcl_idx = random.sample(range(len(inferencer.valset)), 1)[0]
       
        if opt.video:
            print("Starting rendering video:")
            inferencer.render_video(pcl_idx, opt.num_frames, fps=24)
        for pcl_id in pcl_idx:
            if opt.voxel:
                print("Starting generating voxel:")
                inferencer.save_voxel(pcl_id, opt.cube_size, opt.voxel_resolution)
        if opt.pcl:
            print("Starting generating colored point cloud:")
            inferencer.render_pcl(pcl_idx, opt.pcl_views)
    # if opt.metrics:
    #     print("Computing metrics.")
    #     assert opt.sampling_mode == "val"
    #     inferencer.compute_metrics()
    # if opt.interpolate:
    #     # val_dir = "/storage/user/zhouzh/data/ShapeNetChair/val.lst"
    #     # with open(val_dir, 'r') as f:
    #     #     objts=f.readlines()
    #     # objts = [i[:-1] for i in objts]
    #     # for i in range(50):
    #     # src, tgt = random.sample(objts, 2)
    #     srcs = ["85743f17e13271e0709eb7790ef48e0c", "b51032670accbd25d11b9a498c2f9ed5", "5a5e4590c54a70c6322c168d7a7c32f4"]
    #     tgts = ["1e40d41905a9be766ed8c57a1980bb26", "d70d648947c65b1eca8607f540cc62ba", "1954754c791e4571873ec74c119307b9"]
    #     for i in range(len(srcs)):
    #         seed = 1
    #         random.seed(seed)
    #         torch.manual_seed(seed)
    #         np.random.seed(seed)
    #         inferencer.interpolate_latent_codes(
    #             srcs[i], tgts[i], "/usr/stud/zhouzh/thesis/interp/plane/", mode=""
    #         )

        # random.seed(seed)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # inferencer.interpolate_latent_codes(
        #     src, tgt, "/usr/stud/zhouzh/thesis/interp/plane", mode="f"
        # )
        # random.seed(seed)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # inferencer.interpolate_latent_codes(
        #     src, tgt, "/usr/stud/zhouzh/thesis/interp/plane", mode="z"
        # )
