# ----------------------------------------------------------------
# Documentation of the meaning of each hyperparameter. 
# And the helper function to load and update the hyperparameters.
# ----------------------------------------------------------------
"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size (Note the batch size is for each gpu, i.e if n gpus are assigned for training, each step n*batch_size number of images are being used to update model parameters)
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.
        enc_lr: Encoder learning rate.
        num_points_ratio: the ratio to downsample pcl at dataset loading.
    
    NeRF paramters:
        fov: Camera field of view
        ray_start: Near clipping for camera rays.
        ray_end: Far clipping for camera rays.
        h_stddev: Stddev of camera yaw in radians.
        v_stddev: Stddev of camera pitch in radians.
        h_mean: Mean of camera yaw in radians.
        v_mean: Mean of camera pitch in radians.
        cam_r_start: The minimal camera distance to the world origin.
        cam_r_end: The maximal camera distance to the world origin.
        lock_view_dependence: if true, sigma of siren output will not be affected by the ray direction
        hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
        last_back: Flag to fill in background color with last sampled color on ray.

    Train parameters:
        fade_steps: Number of steps to fade in new layer on discriminator after upsample.
        topk_interval: Interval over which to fade the top k ratio.
        topk_v: Minimum fraction of a batch to keep during top k training.
        betas: Beta parameters for Adam.
        unique_lr: Whether to use reduced LRs for mapping network.
        weight_decay: Weight decay parameter.
        r1_lambda: R1 regularization parameter.
        latent_dim: Latent dim for Siren network  in generator.
        grad_clip: Grad clipping parameter.
        discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
        clamp_mode: Clamping function for Siren density output. (relu | softplus)
        num_workers: number of workers used to load dataset during training of each process
        noise_weight: weight of random gaussian noise added to the encoded pcl(output of piontnet)
        enable_discriminator: If true, discriminator is enabled; else, only generator is used and there will be no GAN loss.
        dropout_ratio: The dropout ratio of FiLM layers. 
    
    Generator parameters:
        siren_type: The name of the siren class to use.
        z_dim: The dimension of latent codes / global feature.
        input_dim: input dimension of siren (previously always 3).
        output_dim: output dimension of siren (always 4, i.e [r,g,b,sigma])
        hidden_dim: hidden dimension of siren layers
    
    Encoder parameters (currently unet):
        in_channels: the input volume feature dimension (C_in)
        out_channels: the output volume feature dimension (C_out)
        f_maps: number of feature maps at each level of the encoder; if it's an integer the number of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        num_levels: number of levels in the encoder/decoder path

    Loss parameters:
        geo_loss: if use geometry loss
        photo_loss: if use photometry loss
        depth_loss: if use depth loss
        z_labmda: Weight for experimental latent code positional consistency loss.
        pos_lambda: Weight parameter for experimental positional consistency loss.
        depth_loss_weight: weight for the depth loss

    Dataset parameters:
        path: The path to the trainset and valset.
        dataset: Training dataset. Default shapenet. (CelebA | Carla | Cats | ShapeNet)
        trainset: Determines which part of the shapnet cars should be used as trainset and valset, read from the data/shapnet_MODE.lst. (train | test | valsingle | single2 | twenty | thousand | mysingle....)
        testset: Testset mode, under test/ folder.
"""

import math


def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata["img_size"]
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if (
            curriculum_step > current_step
            and curriculum[curriculum_step].get("img_size", 512) > current_size
        ):
            return curriculum_step
    return float("Inf")


def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata["img_size"]
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if (
            curriculum_step <= current_step
            and curriculum[curriculum_step]["img_size"] == current_size
        ):
            return curriculum_step
    return 0


# This function is never referenced
def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum["update_epochs"]:
        if epoch >= update_epoch:
            step += 1
    return step


def extract_metadata(curriculum, current_step):
    return_dict = {}
    # read training settings at different steps
    for curriculum_step in sorted(
        [cs for cs in curriculum.keys() if type(cs) == int], reverse=True
    ):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break

    # reading other training settings
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


# CelebA = {
#     0: {
#         "batch_size": 28 * 2,
#         "num_steps": 12,
#         "img_size": 64,
#         "batch_split": 2,
#         "gen_lr": 6e-5,
#         "disc_lr": 2e-4,
#     },
#     int(200e3): {},
#     "dataset_path": "/home/ericryanchan/data/celeba/img_align_celeba/*.jpg",
#     "fov": 12,
#     "ray_start": 0.88,
#     "ray_end": 1.12,
#     "fade_steps": 10000,
#     "h_stddev": 0.3,
#     "v_stddev": 0.155,
#     "h_mean": math.pi * 0.5,
#     "v_mean": math.pi * 0.5,
#     "sample_dist": "gaussian",
#     "topk_interval": 2000,
#     "topk_v": 0.6,
#     "betas": (0, 0.9),
#     "unique_lr": False,
#     "weight_decay": 0,
#     "r1_lambda": 0.2,
#     "latent_dim": 256,
#     "grad_clip": 10,
#     "model": "SPATIALSIRENBASELINE",
#     "generator": "ImplicitGenerator3d",
#     "discriminator": "CCSEncoderDiscriminator",
#     "dataset": "CelebA",
#     "clamp_mode": "relu",
#     "z_dist": "gaussian",
#     "hierarchical_sample": True,
#     "z_lambda": 0,
#     "pos_lambda": 15,
#     "last_back": False,
#     "eval_last_back": True,
# }

# CATS = {
#     0: {
#         "batch_size": 28,
#         "num_steps": 24,
#         "img_size": 64,
#         "batch_split": 4,
#         "gen_lr": 6e-5,
#         "disc_lr": 2e-4,
#     },
#     int(200e3): {},
#     "dataset_path": "/home/ericryanchan/graf-beta/data/carla/carla/*.png",
#     "fov": 12,
#     "ray_start": 0.8,
#     "ray_end": 1.2,
#     "fade_steps": 10000,
#     "h_stddev": 0.5,
#     "v_stddev": 0.4,
#     "h_mean": math.pi * 0.5,
#     "v_mean": math.pi * 0.5,
#     "sample_dist": "uniform",
#     "topk_interval": 2000,
#     "topk_v": 0.6,
#     "betas": (0, 0.9),
#     "unique_lr": False,
#     "weight_decay": 0,
#     "r1_lambda": 0.2,
#     "latent_dim": 256,
#     "grad_clip": 10,
#     "model": "SPATIALSIRENBASELINE",
#     "generator": "ImplicitGenerator3d",
#     "discriminator": "StridedDiscriminator",
#     "dataset": "Cats",
#     "clamp_mode": "relu",
#     "z_dist": "gaussian",
#     "hierarchical_sample": True,
#     "z_lambda": 0,
#     "pos_lambda": 0,
#     "last_back": False,
#     "eval_last_back": True,
# }

# CARLA = {
#     0: {
#         "batch_size": 30,  # 30
#         "num_steps": 48,  # 48
#         "img_size": 32,  # 32
#         "batch_split": 1,
#         "gen_lr": 4e-5,
#         "disc_lr": 4e-4,
#     },
#     int(10e3): {
#         "batch_size": 14,
#         "num_steps": 48,
#         "img_size": 64,
#         "batch_split": 2,
#         "gen_lr": 2e-5,
#         "disc_lr": 2e-4,
#     },
#     int(55e3): {
#         "batch_size": 10,
#         "num_steps": 48,
#         "img_size": 128,
#         "batch_split": 5,
#         "gen_lr": 10e-6,
#         "disc_lr": 10e-5,
#     },
#     int(200e3): {},
#     "dataset_path": "/usr/stud/zhouzh/data/carla/*.png",
#     "fov": 30,
#     "ray_start": 0.7,
#     "ray_end": 1.3,
#     "fade_steps": 10000,
#     "sample_dist": "spherical_uniform",
#     "h_stddev": math.pi,
#     "v_stddev": math.pi / 4 * 85 / 90,
#     "h_mean": math.pi * 0.5,
#     "v_mean": math.pi / 4 * 85 / 90,
#     "topk_interval": 1000,
#     "topk_v": 1,
#     "betas": (0, 0.9),
#     "unique_lr": False,
#     "weight_decay": 0,
#     "r1_lambda": 10,
#     "latent_dim": 512, #256
#     "grad_clip": 1,
#     "model": "TALLSIREN",
#     "generator": "ImplicitGenerator3d",
#     "discriminator": "ProgressiveEncoderDiscriminator",
#     "dataset": "Carla",
#     "white_back": True,
#     "clamp_mode": "relu",
#     "z_dist": "gaussian",
#     "hierarchical_sample": True,
#     "z_lambda": 0,
#     "pos_lambda": 0,
#     "learnable_dist": False,
#     "num_workers": 4,
#     "num_fid_output_imgs": 2048,  # 2048
#     "load_all_images": True,
#     "img_subset_lst": "/usr/stud/zhouzh/data/1000_carla_idx.lst",
# }

# CO3D = {
#     0: {
#         "batch_size": 30,
#         "num_steps": 48,
#         "img_size": 32,
#         "batch_split": 1,
#         "gen_lr": 4e-5,
#         "disc_lr": 4e-4,
#     },
#     int(10e3): {
#         "batch_size": 14,
#         "num_steps": 48,
#         "img_size": 64,
#         "batch_split": 2,
#         "gen_lr": 2e-5,
#         "disc_lr": 2e-4,
#     },
#     int(55e3): {
#         "batch_size": 10,
#         "num_steps": 48,
#         "img_size": 128,
#         "batch_split": 5,
#         "gen_lr": 10e-6,
#         "disc_lr": 10e-5,
#     },
#     int(200e3): {},
#     "dataset_path": "/usr/stud/zhouzh/data/my_co3d/car/*.jpg",
#     "fov": 30,
#     "ray_start": 0.7,
#     "ray_end": 1.3,
#     "fade_steps": 10000,
#     "sample_dist": "spherical_uniform",
#     "h_stddev": math.pi,
#     "v_stddev": math.pi / 4 * 85 / 90,
#     "h_mean": math.pi * 0.5,
#     "v_mean": math.pi / 4 * 85 / 90,
#     "topk_interval": 1000,
#     "topk_v": 1,
#     "betas": (0, 0.9),
#     "unique_lr": False,
#     "weight_decay": 0,
#     "r1_lambda": 10,
#     "latent_dim": 256,
#     "grad_clip": 1,
#     "model": "TALLSIREN",
#     "generator": "ImplicitGenerator3d",
#     "discriminator": "ProgressiveEncoderDiscriminator",
#     "dataset": "CO3D",
#     "white_back": True,
#     "clamp_mode": "relu",
#     "z_dist": "gaussian",
#     "hierarchical_sample": True,
#     "z_lambda": 0,
#     "pos_lambda": 0,
#     "learnable_dist": False,
# }

# shapenet = {
#         0: {
#         "batch_size": 16,
#         "num_steps": 48,
#         "img_size": 32,
#         "batch_split": 1,
#         "gen_lr": 4e-5,
#         "disc_lr": 4e-4,
#         "enc_lr": 4e-5,
#         "num_points_ratio": 0.0625
#     },
#     int(10e3): {
#         "batch_size": 12,
#         "num_steps": 48,
#         "img_size": 64,
#         "batch_split": 3,
#         "gen_lr": 2e-5,
#         "disc_lr": 2e-4,
#         "enc_lr": 2e-5,
#         "num_points_ratio": 0.25
#     },
#     int(55e3): {
#         "batch_size": 4,
#         "num_steps": 48,
#         "img_size": 128,
#         "batch_split": 4,
#         "gen_lr": 10e-6,
#         "disc_lr": 10e-5,
#         "enc_lr": 1e-5,
#         "num_points_ratio": 1.0
#     },
#     int(200e3): {},
#     "dataset_path": "/usr/stud/zhouzh/data/ShapeNetCar/", #"/usr/stud/zhouzh/data/ShapeNetCar/*/image/**.png",
#     "fov": 30,
#     "ray_start": 0.7,
#     "ray_end": 1.3,
#     "fade_steps": 2000,
#     "sample_dist": "spherical_uniform",
#     "h_stddev": math.pi,
#     "v_stddev": math.pi / 4 * 85 / 90,
#     "h_mean": math.pi * 0.5,
#     "v_mean": math.pi / 4 * 85 / 90,
#     "topk_interval": 1000,
#     "topk_v": 1,
#     "betas": (0, 0.9),
#     "unique_lr": False, # disabled
#     "weight_decay": 0,
#     "r1_lambda": 10,
#     "latent_dim": 512,
#     "grad_clip": 1,
#     "model": "TALLSIREN",
#     "generator": "ImplicitGenerator3d",
#     "discriminator": "ProgressiveEncoderDiscriminator",
#     "dataset": "ShapeNet",
#     "white_back": True,
#     "clamp_mode": "relu",
#     "z_dist": "gaussian",
#     "hierarchical_sample": True,
#     "z_lambda": 0,
#     "pos_lambda": 0,
#     "learnable_dist": False,
#     "num_workers": 4,
#     "num_fid_output_imgs": 2048,  # 2048
#     'noise_weight': 0,
#     'encoding': True,
#     'geo_loss': True,
#     'kdtree_r': 0.05,
#     'rgb_siren_noise_ratio': 0.1,
#     'freeze_encoder_until': 5e3,
#     'lock_view_dependence': True
# "load_mode": 2,  # 0: load all images; 1: load all images of 42 cars; 2: load 250 cars, each car 4 images
# "subset_42_lst": "/usr/stud/zhouzh/data/42car_ShapeNet_idx.lst",
# "subset_250_lst": "/usr/stud/zhouzh/data/250car_ShapeNet_idx.lst",
# }
