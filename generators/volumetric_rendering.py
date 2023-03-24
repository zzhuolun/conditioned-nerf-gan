"""
Differentiable volumetric implementation used by pi-GAN generator.
"""

import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

from .math_utils_torch import *


def fancy_integration(
    rgb_sigma,
    z_vals,
    device,
    noise_std=0.5,
    last_back=False,
    white_back=False,
    clamp_mode=None,
    fill_mode=None,
):
    """Performs NeRF volumetric rendering."""

    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(
        deltas[:, :, :1]
    )  # set background as infinite far.
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == "softplus":
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == "relu":
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2
    )
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += 1 - weights_sum

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum

    if fill_mode == "debug":
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor(
            [1.0, 0, 0], device=rgb_final.device
        )
    elif fill_mode == "weight":
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, weights


def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""
    # n: batch_size
    
    W, H = resolution
    # Create full screen cooridinates
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, W, device=device), torch.linspace(-1, 1, H, device=device)
    )  # 32x32
    x = x.T.flatten()  # 1024
    y = y.T.flatten()
    z = torch.ones_like(x, device=device) / np.tan(
        (2 * math.pi * fov / 360) / 2
    )  # -1/tan(fov/2) # 1024

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))  # ray directions 1024x3

    z_vals = (
        torch.linspace(ray_start, ray_end, num_steps, device=device)
        .reshape(1, num_steps, 1)
        .repeat(W * H, 1, 1)
    )  # 1024x48x1
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals  # t*d

    points = torch.stack(n * [points])  # Bx1024x48x3
    z_vals = torch.stack(n * [z_vals])  # Bx1024x48x1
    rays_d_cam = torch.stack(n * [rays_d_cam]).to(device)  # 1x1024x3
    return points, z_vals, rays_d_cam


def perturb_points(points, z_vals, ray_directions, device):
    # randomized the distance between sampled points along a ray
    distance_between_points = z_vals[:, :, 1:2, :] - z_vals[:, :, 0:1, :]
    offset = (torch.rand(z_vals.shape, device=device) - 0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def transform_sampled_points(
    points,
    z_vals,
    ray_directions,
    device,
    cam2worlds,
):
    """Samples a camera position and maps points in camera space to world space.
    Args:
        points (torch.Tensor: batchsize x (W*H) x num_steps x 3): points in camera space
        z_vals (torch.Tensor: batchsize x (W*H) x num_steps x 1): depth/z values of the points on rays
        ray_directions (torch.Tensor: batchsize x (W*H) x 3): ray directions of each pixel
        device (torch.device):
        cam2worlds (torch.Tensor: batchsize x 3, optional): The origin of camera in world space, passed from the ShapeNet dataset. Defaults to None.
        cam_r_start (int, optional): The minimal camera distance to the world origin. Defaults to 0.
        cam_r_end (int, optional): The maximal camera distance to the world origin. Defaults to 1.
        h_stddev (int, optional): std of camera horizontal postion . Defaults to 1.
        v_stddev (int, optional): std of camera vertical positon. Defaults to 1.
        h_mean (_type_, optional): mean of camera horizontal position. Defaults to math.pi*0.5.
        v_mean (_type_, optional): mean of camera vertical position. Defaults to math.pi*0.5.
        mode (str, optional): mode of camera position distribution. Defaults to "normal".
    Returns:
        transformed_points[..., :3] (torch.Tensor: batchsize x (W*H) x num_steps x 3): points in world space
        z_vals (torch.Tensor: batchsize x (W*H) x num_steps x 1): perturbed depth/z values
        transformed_ray_directions (torch.Tensor: batchsize x (W*H) x 3): ray directions in world space
        transformed_ray_origins (torch.Tensor: batchsize x (W*H) x 3): ray_origins in world space. Actually transformed_ray_origins[:,i,:]=camera_origin
        pitch (torch.Tensor: batchsize x 1): yaw angles of camera
        yaw (torch.Tensor: batchsize x 1): pitch angles of camera
    """

    n, num_rays, num_steps, channels = points.shape

    points, z_vals = perturb_points(points, z_vals, ray_directions, device)
    # if mode == "fixed":
    cam2world_matrix = cam2worlds
    # else:
    #     camera_origin = sample_camera_positions(
    #         device=device,
    #         cam_r_start=cam_r_start,
    #         cam_r_end=cam_r_end,
    #         n=points.shape[0],
    #         mode=mode,
    #     )
    #     # batchsize x 4 x 4
    #     cam2world_matrix = create_cam2world_matrix(camera_origin, device=device)

    # torch.save(cam2world_matrix, f"checkpos/data/cam2world_128_{step}")
    # batchsize x (W*H) x num_steps x 4
    points_homogeneous = torch.ones(
        (points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1),
        device=device,
    )
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = (
        torch.bmm(
            cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0, 2, 1)
        )
        .permute(0, 2, 1)
        .reshape(n, num_rays, num_steps, 4)
    )  # batchsize x (W*H) x num_steps x 4

    # batchsize x (W*H) x 3
    transformed_ray_directions = (
        torch.bmm(
            cam2world_matrix[..., :3, :3],
            ray_directions.reshape(n, -1, 3).permute(0, 2, 1),
        )
        .permute(0, 2, 1)
        .reshape(n, num_rays, 3)
    )

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = (
        torch.bmm(cam2world_matrix, homogeneous_origins)
        .permute(0, 2, 1)
        .reshape(n, num_rays, 4)[..., :3]
    )

    return (
        transformed_points[..., :3],
        z_vals,
        transformed_ray_directions,
        transformed_ray_origins,
    )


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def sample_camera_positions(
    device,
    up_direction,
    cam_r_start=0,
    cam_r_end=1,
    n=1,
):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    up denotes the upward direction is y axis or z axis.
    """
    assert up_direction in ["y", "z"]
    theta = np.arccos(1 - np.random.rand(n))
    theta = np.clip(theta, 1e-5, np.pi - 1e-5)
    phi = np.random.rand(n) * np.pi * 2
    r = np.random.rand(n) * (cam_r_end - cam_r_start) + cam_r_start
    cam_origin = np.zeros((n, 3))
    cam_origin[:, 0] = r * np.sin(theta) * np.cos(phi)
    if up_direction == "z":
        cam_origin[:, 1] = r * np.sin(theta) * np.sin(phi)
        cam_origin[:, 2] = r * np.cos(theta)
    elif up_direction == "y":
        cam_origin[:, 2] = r * np.sin(theta) * np.sin(phi)
        cam_origin[:, 1] = r * np.cos(theta)
    return torch.from_numpy(cam_origin).type(torch.float32).to(device)


def xyz2angles(xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the phi a,d theta angels from the xyz coordinates of a points

    Args:
        cam_origin (torch.Tensor): shape batchsize x 3
    """
    xyz_normalized = normalize_vecs(xyz)
    phi = torch.acos(xyz_normalized[:, -1])  # arccos(z)
    theta = torch.acos(
        xyz[:, 0] / (torch.sqrt(1 - xyz[:, -1] * 2))
    )  # arccos(x/sqrt(1-z^2))
    return phi.unsqueeze(1), theta.unsqueeze(1)


def create_cam2world_matrix(origin, up_direction, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    # Note:
    forward_vector = normalize_vecs(-origin)
    # forward_vector = normalize_vecs(forward_vector)
    assert up_direction in ["y", "z"]
    if up_direction == "y":
        up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(
            forward_vector
        )
    elif up_direction == "z":
        up_vector = torch.tensor([0, 0, 1], dtype=torch.float, device=device).expand_as(
            forward_vector
        )
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = (
        torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    )
    rotation_matrix[:, :3, :3] = torch.stack(
        (-left_vector, -up_vector, forward_vector), axis=-1
    )

    translation_matrix = (
        torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    )
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def create_world2cam_matrix(forward_vector, origin, device):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[
        denom < eps
    ] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
        bins_g[..., 1] - bins_g[..., 0]
    )
    return samples


def distance2depth(distance: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
    """Projects the distance between 3d point to camera origin to the z value in the camera coord.

    Args:
        distance (torch.Tensor): batch_size*(W*H)*1, the distances
        ray (torch.Tensor): batch_size*(W*H)*3, the direction of the rays

    Returns:
        torch.Tensor: batch_size*(W*H)*1, The depth map
    """
    # ray[...,-1] = ray dotproduct (0,0,-1)
    return ray[..., -1:] * distance
