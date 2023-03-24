# ---------------------------------------------------------------------
# Some commonly changed hyperparameters, inherited from the default.py.
# See documentation at curriculums.py
# ---------------------------------------------------------------------

# Note: batch size are for training on 48GB VRAM GPUs, for smaller gpus,
# adjust the batch_size accordingly.
from .default import upsample_step

gpu_small = False
special = {
    upsample_step[0]: {
        "batch_size": 24,  # 32
        "batch_split": 2,
    },
    upsample_step[1]: {
        "batch_size": 12,  # 24
        "batch_split": 6,
    },
    upsample_step[2]: {
        "batch_size": 12,
        "batch_split": 6,
        # "img_size": 64,
    },
    upsample_step[3]: {
        "batch_size": 12,
        "batch_split": 6,
        "num_steps": 48,
        # "img_size": 256,
    },
    "num_workers": 8,
    "photo_loss": True,
    "depth_loss": False,
    "lock_view_dependence": True,
    "ray_start": 0.25,
    "ray_end": 1.95,
    "cam_r_start": 0.7,
    "cam_r_end": 1.5,
    "dropout_ratio": 0,
    "fov": 49.134342641202636,
    "white_back": True,
    "depth_loss_weight": 1,
    "noise_weight": 0,
    "enable_discriminator": True,
    "generator": {
        "siren_type": "SHORTSIREN",
        "z_dim": 512,  # dimention of latent codes
        "input_dim": 3,  # input dimension of siren (previously always 3)
        "output_dim": 4,  # output dimension of siren
        "hidden_dim": 256,  # hidden dimension of siren layers
    },
    "discriminator": "ProgressiveDiscriminator",
    "unet": {
        "type": "UNet3D",
        "in_channels": 4,
        "out_channels": 32,
        "f_maps": 32,
        "num_levels": 4,
        "is_segmentation": False,
        "final_sigmoid": False,
        "return_global": True,
    },
    "fid": {
        "subset": "val",
        # "num_real_imgs": 6000,
        "target_size": 128,
    },
    "dataset": {
        "path": "/storage/user/zhouzh/data/ShapeNetCar",
        "trainset": "thousand",
        "testset": "twenty",
        "load_voxel": False,
        "load_pcl": True,
        "load_depth": False,
        "cond_mode": False,
    },
    "random_gen_img": True,
}

# if special["dataset_path"] == "/usr/stud/zhouzh/data/ShapeNetCar":
#     assert special["white_back"]
if gpu_small:
    for i in [0, 1, 2, 3]:
        special[upsample_step[i]]["batch_size"] = int(
            special[upsample_step[i]]["batch_size"] / 2
        )
