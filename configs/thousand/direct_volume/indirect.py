from ..default import upsample_step
from copy import deepcopy

indirect = {
    upsample_step[0]: {"batch_size": 24},
    upsample_step[2]: {"batch_split": 12},
    upsample_step[3]: {"batch_split": 12},
    "enable_discriminator": False,
    "generator": {
        "siren_type": "TALLSIREN",
        "z_dim": 32,  # dimention of latent codes
        "input_dim": 3,  # input dimension of siren (previously always 3)
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
        "return_global": False,
    },
}

indirect_dis = deepcopy(indirect)
indirect_dis["enable_discriminator"] = True
