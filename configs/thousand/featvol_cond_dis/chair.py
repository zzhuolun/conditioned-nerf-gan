from ..default import upsample_step
from copy import deepcopy

chair_doublesiren_nophotoloss = {
    upsample_step[0]: {"batch_size": 24},
    "enable_discriminator": True,
    "generator": {
        "siren_type": "DoubleSIREN_dg",
        "z_dim": 256,  # dimention of latent codes
        "input_dim": 32,  # input dimension of siren (previously always 3)
        "output_dim": 4,  # output dimension of siren
        "hidden_dim": 256,  # hidden dimension of siren layers
    },
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
    "dataset": {
        "path": "/usr/stud/zhouzh/pi-gan/data/ShapeNetChair/ShapeNetChair",
        "trainset": "thousand",
        "testset": "twenty",
        "cond_mode": False,
        "voxel_resolution": 64,
    },
    "photo_loss": False,
    "random_gen_img": False
}

chair_doublesiren_dis = deepcopy(chair_doublesiren_nophotoloss)
chair_doublesiren_dis["photo_loss"] = True
chair_doublesiren_dis["random_gen_img"] = True