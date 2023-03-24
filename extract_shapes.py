# ----------------------------------
# Extract shapes from latent code
# ---------------------------------
from datetime import datetime
from locale import currency
from pathlib import Path
from tabnanny import check

import mrcfile
import numpy as np
import plyfile
import torch


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_generator(
    generator,
    z,
    voxel_resolution=256,
    voxel_origin=[0, 0, 0],
    cube_length=1.2,
    psi=0.5,
):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(
        voxel_resolution, voxel_origin, cube_length
    )
    samples = samples.to(generator.device)
    sigmas = torch.zeros(
        (samples.shape[0], samples.shape[1], 1), device=generator.device
    )
    max_points = 64**3
    splits = voxel_resolution**3 // max_points
    head = 0
    with torch.no_grad():
        for i in range(splits):
            split_size = voxel_resolution**3 // splits
            assert voxel_resolution % splits == 0
            coarse_output = generator.siren(
                samples[:, head : head + split_size],
                z,
                voxel_resolution,
                voxel_resolution // splits,
            ).reshape(samples.shape[0], -1, 4)
            sigmas[:, head : head + split_size] = coarse_output[:, :, -1:]
            head += split_size

    sigmas = (
        sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution))
        .cpu()
        .numpy()
    )

    return sigmas
