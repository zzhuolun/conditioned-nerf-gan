from ..default import upsample_step
from copy import deepcopy

dg = {
    upsample_step[0]: {"batch_size": 24},
    "enable_discriminator": False,
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
}

dg_dis = deepcopy(dg)
dg_dis["enable_discriminator"] = True

dg_deepunet = {
    upsample_step[0]: {"batch_size": 24},
    "enable_discriminator": False,
    "generator": {
        "siren_type": "TALLSIREN_dg",
        "z_dim": 512,  # dimention of latent codes
        "input_dim": 32,  # input dimension of siren (previously always 3)
        "output_dim": 4,  # output dimension of siren
        "hidden_dim": 256,  # hidden dimension of siren layers
    },
    "unet": {
        "in_channels": 4,
        "out_channels": 32,
        "f_maps": 32,
        "num_levels": 5,
        "is_segmentation": False,
        "final_sigmoid": False,
        "return_global": True,
    },
}

# four layers of siren, w/o discriminator
dg_shortsiren = deepcopy(dg)
dg_shortsiren["generator"]["siren_type"] = "SHORTSIREN_dg"

# two layers of siren, w/o discriminator
dg_doublesiren = deepcopy(dg)
dg_doublesiren["generator"]["siren_type"] = "DoubleSIREN_dg"

# one layer of siren, w/o discriminator
dg_singlesiren = deepcopy(dg)
dg_singlesiren["generator"]["siren_type"] = "SingleSIREN_dg"

# four layers of siren with unconditioned discriminator
dg_shortsiren_dis = deepcopy(dg_shortsiren)
dg_shortsiren_dis["enable_discriminator"] = True

# four layers of siren, unconditioned discriminator, w/o photo loss 
dg_shortsiren_dis_nophotoloss = deepcopy(dg_shortsiren_dis)
dg_shortsiren_dis_nophotoloss["photo_loss"] = False
dg_shortsiren_dis_nophotoloss["random_gen_img"] = False

# four layers of siren, unconditioned discriminator, w/o photo loss, donot use same camera for generated image and real image
dg_shortsiren_dis_nophotoloss_randomgenimg = deepcopy(dg_shortsiren_dis_nophotoloss)
dg_shortsiren_dis_nophotoloss_randomgenimg["random_gen_img"] = True



# feature pyramid unet
doublesiren_dis_featurepyramid = deepcopy(dg_shortsiren_dis)
doublesiren_dis_featurepyramid["generator"]["siren_type"] = "DoubleSIREN_dg"
doublesiren_dis_featurepyramid["unet"]["type"] = "PyramidUNet3D"
doublesiren_dis_featurepyramid["generator"]["input_dim"] = 224


# feature pyramid unet
doublesiren_dis= deepcopy(dg_shortsiren_dis)
doublesiren_dis["generator"]["siren_type"] = "DoubleSIREN_dg"
