dRes = {
    "enable_discriminator": True,
    "generator": {
        "siren_type": "TALLSIREN_dRes",
        "z_dim": 32,  # dimention of latent codes
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
        "return_global": False,
    },
}
