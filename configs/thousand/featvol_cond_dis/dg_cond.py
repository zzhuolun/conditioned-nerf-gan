from ..default import upsample_step
from copy import deepcopy

cond_nearest = {
    upsample_step[0]: {"batch_size": 24},
    "enable_discriminator": True,
    "generator": {
        "siren_type": "TALLSIREN_dg",
        "z_dim": 256,  # dimention of latent codes
        "input_dim": 32,  # input dimension of siren (previously always 3)
        "output_dim": 4,  # output dimension of siren
        "hidden_dim": 256,  # hidden dimension of siren layers
    },
    "unet": {
        "in_channels": 4,
        "out_channels": 32,
        "f_maps": 32,
        "num_levels": 4,
        "is_segmentation": False,
        "final_sigmoid": False,
        "return_global": True,
    },
    "dataset":{"cond_mode": "nearest"},
    "discriminator": "ProgressiveDiscriminator_inputCat",
}
cond_random = deepcopy(cond_nearest)
cond_random["dataset"]["cond_mode"] = "random"

cond_furthest = deepcopy(cond_nearest)
cond_furthest["dataset"]["cond_mode"] = "furthest"

cond_nearest_fix = deepcopy(cond_nearest)
cond_nearest_fix["random_gen_img"] = False

cond_nearest_nophotoloss = deepcopy(cond_nearest)
cond_nearest_nophotoloss["photo_loss"] = False
cond_nearest_nophotoloss[
    "random_gen_img"
] = False  # When training discriminator, use fixed cam2world or random.

cond_random_32 = deepcopy(cond_random)
cond_random_32["dataset"]["voxel_resolution"] = 32

cond_random_nophotoloss_32 = deepcopy(cond_random_32)
cond_random_nophotoloss_32["photo_loss"] = False
cond_random_nophotoloss_32["random_gen_img"] = False


cond_random_nophotoloss = deepcopy(cond_nearest_nophotoloss)
cond_random_nophotoloss["dataset"]["cond_mode"] = "random"