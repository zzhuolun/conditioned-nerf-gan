"""Implicit generator for 3D volumes"""

import torch.nn as nn
import torch
from .volumetric_rendering import *
from . import siren


class ImplicitGenerator3d(nn.Module):
    def __init__(
        self, siren_type, z_dim, input_dim, output_dim, hidden_dim, drop_out=0
    ):
        super().__init__()
        self.z_dim = z_dim
        SIREN = getattr(siren, siren_type)
        self.siren = SIREN(
            z_dim=z_dim,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            drop_out=drop_out,
            device=None,
        )
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.siren.device = device

        # self.generate_avg_frequencies()

    def forward(
        self,
        z,
        cam2worlds,
        img_size,
        fov,
        ray_start,
        ray_end,
        num_steps,
        hierarchical_sample,
        # lock_view_dependence=False,
        **kwargs
    ):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """
        # if isinstance(z,tuple): # if the global features of unet is also returned
            
        #     batch_size = z[0].shape[0]
        # else:
        #     batch_size = z.shape[0]
        batch_size = cam2worlds.shape[0]
        # Generate initial camera rays and sample points.
        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
            )  # batch_size, pixels, num_steps, 1
            (
                transformed_points,
                z_vals,
                transformed_ray_directions,
                transformed_ray_origins,
            ) = transform_sampled_points(
                points_cam,
                z_vals,
                rays_d_cam,
                device=self.device,
                cam2worlds=cam2worlds,
            )
            # torch.save(transformed_points,f'checkpos/data/ray_128_{self.step}')
            # transformed_ray_directions_expanded = torch.unsqueeze(
            #     transformed_ray_directions, -2
            # )
            # transformed_ray_directions_expanded = (
            #     transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            # )
            # transformed_ray_directions_expanded = (
            #     transformed_ray_directions_expanded.reshape(
            #         batch_size, img_size * img_size * num_steps, 3
            #     )
            # )
            transformed_points = transformed_points.reshape(
                batch_size, img_size * img_size * num_steps, 3
            )

            # if lock_view_dependence:
            #     transformed_ray_directions_expanded = torch.zeros_like(
            #         transformed_ray_directions_expanded
            #     )
            #     transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points
        coarse_output = self.siren(
            transformed_points,
            z,
            img_size,
            num_steps,
        ).reshape(batch_size, img_size * img_size, num_steps, 4)

        # Re-sample fine points along camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(
                    batch_size, img_size * img_size, num_steps, 3
                )
                _, _, weights = fancy_integration(
                    coarse_output,
                    z_vals,
                    device=self.device,
                    clamp_mode=kwargs["clamp_mode"],
                    noise_std=kwargs["nerf_noise"],
                )

                weights = (
                    weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                )

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(
                    z_vals_mid, weights[:, 1:-1], num_steps, det=False
                ).detach()
                fine_z_vals = fine_z_vals.reshape(
                    batch_size, img_size * img_size, num_steps, 1
                )

                fine_points = (
                    transformed_ray_origins.unsqueeze(2).contiguous()
                    + transformed_ray_directions.unsqueeze(2).contiguous()
                    * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                )
                fine_points = fine_points.reshape(
                    batch_size, img_size * img_size * num_steps, 3
                )

                # if lock_view_dependence:
                #     transformed_ray_directions_expanded = torch.zeros_like(
                #         transformed_ray_directions_expanded
                #     )
                #     transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(
                fine_points,
                z,
                img_size,
                num_steps,
            ).reshape(batch_size, img_size * img_size, -1, 4)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals
        # Create images with NeRF
        pixels, distances, weights = fancy_integration(
            all_outputs,
            all_z_vals,
            device=self.device,
            white_back=kwargs.get("white_back", False),
            last_back=kwargs.get("last_back", False),
            clamp_mode=kwargs["clamp_mode"],
            noise_std=kwargs["nerf_noise"],
        )

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        depth = distance2depth(distances, rays_d_cam)
        depth_map = depth.reshape(batch_size, img_size, img_size).contiguous()
        return pixels, depth_map

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts
