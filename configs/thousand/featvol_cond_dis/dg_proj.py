from ..default import upsample_step
from copy import deepcopy

proj_random = {
    upsample_step[0]: {"batch_size": 24, "batch_split": 2,},
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
    "dataset":{"cond_mode": "random"},
    "discriminator": "ProgressiveDiscriminator_projection",
}