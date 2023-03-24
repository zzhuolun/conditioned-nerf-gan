# ---------------------------------------------------------------------
# Default hyperparameters, inherited by special.py. See documentation at curriculums.py
# ---------------------------------------------------------------------
import math

num_gpus = 1
# upsample_step_twenty = [0, 500, 1500, 3500, 200e3]
upsample_step_thousand = [0, 5000, 15000, 25000]
upsample_step = upsample_step_thousand
# upsample_step = [int(i * 1000 / 20 / num_gpus) for i in upsample_step_twenty]
# upsample_step = [0, 3, 6, 9]
default = {
    upsample_step[0]: {
        "batch_size": 32,
        "num_steps": 48,
        "img_size": 32,
        "batch_split": 1,
        "gen_lr": 4e-5,
        "disc_lr": 4e-4,
        "enc_lr": 8e-5,
        "num_points_ratio": 0.0625,
        "voxel_resolution": 64,
        "z_reg_weight": 0.01,
    },
    upsample_step[1]: {
        "batch_size": 24,
        "num_steps": 48,
        "img_size": 64,
        "batch_split": 2,
        "gen_lr": 2e-5,
        "disc_lr": 2e-4,
        "enc_lr": 4e-5,
        "num_points_ratio": 0.25,
        "voxel_resolution": 64,
        "z_reg_weight": 0.001,
    },
    upsample_step[2]: {
        "batch_size": 4,
        "num_steps": 48,
        "img_size": 128,
        "batch_split": 4,
        "gen_lr": 10e-6,
        "disc_lr": 10e-5,
        "enc_lr": 2e-5,
        "num_points_ratio": 1.0,
        "voxel_resolution": 64,
        "z_reg_weight": 0.0005,
    },
    upsample_step[3]: {
        "batch_size": 4,
        "num_steps": 64,
        "img_size": 128,
        "batch_split": 4,
        "gen_lr": 10e-6,
        "disc_lr": 10e-5,
        "enc_lr": 2e-5,
        "num_points_ratio": 1.0,
        "voxel_resolution": 64,
        "z_reg_weight": 0.0001,
    },
    "white_back": True,
    # renderer
    "fov": 30,
    "ray_start": 0.7,
    "ray_end": 1.3,
    "cam_r_start": 0,
    "cam_r_end": 1,
    "fade_steps": 2000,
    "h_stddev": math.pi,
    "v_stddev": math.pi / 4 * 85 / 90,
    "h_mean": math.pi * 0.5,
    "v_mean": math.pi / 4 * 85 / 90,
    "topk_interval": 1000,
    "topk_v": 1,
    "betas": (0, 0.9),
    "unique_lr": False,  # disabled
    "weight_decay": 0,
    "r1_lambda": 10,
    "grad_clip": 1,
    "discriminator": "ProgressiveDiscriminator",
    # "dataset": "ShapeNet",
    "clamp_mode": "relu",
    "z_dist": "gaussian",
    "hierarchical_sample": True,
    "z_lambda": 0,
    "pos_lambda": 0,
    "learnable_dist": False,
    "lock_view_dependence": False,
    "geo_loss": False,
    "photo_loss": False,
}
