from tempfile import gettempdir
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30.0 * x)


def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.kaiming_normal_(
            m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_output_dim),
        )

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]

        return frequencies, phase_shifts


class PointFeaturesMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_output_dim),
        )
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        # z: B, N, c
        # output:
        # frequencies: B, N, map_output_dim/2
        # phase_shifts: B, N, map_output_dim/2
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]
        return frequencies, phase_shifts


class LongMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(map_hidden_dim, map_output_dim),
        )

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]

        return frequencies, phase_shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(
                    -np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq
                )

    return init


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_out_prob=0):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(drop_out_prob)
        self.drop_out_prob = drop_out_prob

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)  # B, num_points, 256
        freq = freq.unsqueeze(1).expand_as(x)  # B, num_points, 256
        phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        y = torch.sin(freq * x + phase_shift)
        if self.drop_out_prob > 0:
            y = self.dropout_layer(y)
        return y


class PointwiseFiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_out_prob=0):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(drop_out_prob)
        self.drop_out_prob = drop_out_prob

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)  # B, num_points, 256

        # No need to expand freq and phase_shift since feature vector for each point is different
        y = torch.sin(freq * x + phase_shift)
        if self.drop_out_prob > 0:
            y = self.dropout_layer(y)
        return y


class SirenLayer(nn.Module):
    """Original siren layer. No freq and phase_shift as in FiLMLayer."""

    def __init__(self, input_dim, hidden_dim, drop_out_prob=0):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.dropout_layer = nn.Dropout(drop_out_prob)
        self.drop_out_prob = drop_out_prob

    def forward(self, x):
        x = self.layer(x)  # B, num_points, 256

        # No need to expand freq and phase_shift since feature vector for each point is different
        # freq = freq.unsqueeze(1).expand_as(x)  # B, num_points, 256
        # phase_shift = phase_shift.unsqueeze(1).expand_as(x)

        y = torch.sin(x)
        if self.drop_out_prob > 0:
            y = self.dropout_layer(y)
        return y


# class ResSirenBlock(nn.Module):
#     """Original siren layer. No freq and phase_shift as in FiLMLayer."""

#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self, x):
#         net = torch.sin(self.fc1(x))  # B, num_points, 256
#         net = torch.sin(self.fc2(net))
#         net = self.fc3(net)
#         y = torch.sin(x + net)
#         return y

class ResSirenBlock(nn.Module):
    """Original siren layer. No freq and phase_shift as in FiLMLayer. Two layers."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        net = torch.sin(self.fc1(x))  # B, num_points, 256
        net = self.fc2(net)
        y = torch.sin(x + net)
        return y

class TALLSIREN(nn.Module):
    """Primary SIREN architecture used in pi-GAN generators."""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                PointwiseFiLMLayer(input_dim, hidden_dim, drop_out),
                PointwiseFiLMLayer(hidden_dim, hidden_dim, drop_out),
                PointwiseFiLMLayer(hidden_dim, hidden_dim, drop_out),
                PointwiseFiLMLayer(hidden_dim, hidden_dim, drop_out),
                PointwiseFiLMLayer(hidden_dim, hidden_dim, drop_out),
                PointwiseFiLMLayer(hidden_dim, hidden_dim, drop_out),
                PointwiseFiLMLayer(hidden_dim, hidden_dim, drop_out),
                PointwiseFiLMLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        # For color c output
        # self.color_layer_sine = FiLMLayer(
        #     hidden_dim + 3, hidden_dim
        # )  # the extra 3 dim at input is from the 3d ray direction
        # self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = PointFeaturesMappingNetwork(
            z_dim, 256, len(self.network) * hidden_dim * 2
        )
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        # self.color_layer_sine.apply(frequency_init(25))
        # self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, feature_volume, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            feature_volume (torch.Tensor: B, c, 32, 32, 32): output of the voxel encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        frequencies, phase_shifts = self.mapping_network(points_feature)
        frequencies = frequencies * 15 + 30

        x = points
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = self.final_layer(x)
        # freq = frequencies[..., -self.hidden_dim :]
        # phase = phase_shifts[..., -self.hidden_dim :]
        # rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), freq, phase)
        # rbg = self.color_layer_linear(rbg)
        return rgb_sigma

class TALLSIREN_dRes(nn.Module):
    """Direct feature volume approach with 2 residual blocks."""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        input_dim = z_dim
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                SirenLayer(input_dim, hidden_dim, drop_out),
                ResSirenBlock(hidden_dim),
                ResSirenBlock(hidden_dim),
                SirenLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, feature_volume, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            feature_volume (torch.Tensor: B, c, 32, 32, 32): output of the voxel encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for index, layer in enumerate(self.network):
            x = layer(x)

        rgb_sigma = self.final_layer(x)
        return rgb_sigma


class TALLSIREN_dResLong(nn.Module):
    """Direct feature volume approach with 4 residual blocks."""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        input_dim = z_dim
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                SirenLayer(input_dim, hidden_dim, drop_out),
                ResSirenBlock(hidden_dim),
                ResSirenBlock(hidden_dim),
                ResSirenBlock(hidden_dim),
                ResSirenBlock(hidden_dim),
                SirenLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, feature_volume, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            feature_volume (torch.Tensor: B, c, 32, 32, 32): output of the voxel encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for index, layer in enumerate(self.network):
            x = layer(x)

        rgb_sigma = self.final_layer(x)
        return rgb_sigma


class TALLSIREN_FG(nn.Module):
    """Direct feature volume with global features"""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.mapping_network = nn.Linear(z_dim, len(self.network) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, z, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        feature_volume, global_feature = z

        frequencies_offsets = self.mapping_network(global_feature)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]
        frequencies = frequencies * 15 + 30

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = _sigmoid_rgb(self.final_layer(x))
        return rgb_sigma


class SHORTSIREN_FG(nn.Module):
    """Direct feature volume with global features"""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.mapping_network = nn.Linear(z_dim, len(self.network) * hidden_dim * 2)
        self.network.apply(frequency_init(12))
        self.final_layer.apply(frequency_init(12))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, z, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        feature_volume, global_feature = z

        frequencies_offsets = self.mapping_network(global_feature)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]
        frequencies = frequencies * 15 + 30

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = _sigmoid_rgb(self.final_layer(x))
        return rgb_sigma


class SHORTSIREN_FG_Pyrmd(nn.Module):
    """Direct feature volume with global features"""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.mapping_network = nn.Linear(z_dim, len(self.network) * hidden_dim * 2)
        self.network.apply(frequency_init(12))
        self.final_layer.apply(frequency_init(12))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, z, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        feature_pyramid, global_feature = z

        frequencies_offsets = self.mapping_network(global_feature)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]
        frequencies = frequencies * 15 + 30

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        x = feature_pyramid_interpolation(
            points, feature_pyramid, img_size, num_steps, voxel_length=voxel_length
        )
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = _sigmoid_rgb(self.final_layer(x))
        return rgb_sigma


class DOUBLESIREN_FG(nn.Module):
    """Direct feature volume with global features"""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.mapping_network = nn.Linear(z_dim, len(self.network) * hidden_dim * 2)
        self.network.apply(frequency_init(12))
        self.final_layer.apply(frequency_init(12))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, z, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        feature_volume, global_feature = z

        frequencies_offsets = self.mapping_network(global_feature)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]
        frequencies = frequencies * 15 + 30

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = _sigmoid_rgb(self.final_layer(x))
        return rgb_sigma


class SHORTSIREN_F(nn.Module):
    """Direct feature volume with global features"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
        **kwargs,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.            
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                SirenLayer(input_dim, hidden_dim, drop_out),
                SirenLayer(hidden_dim, hidden_dim, drop_out),
                SirenLayer(hidden_dim, hidden_dim, drop_out),
                SirenLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.network.apply(frequency_init(12))
        self.final_layer.apply(frequency_init(12))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, feature_volume, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for layer in self.network:
            x = layer(x)

        rgb_sigma = _sigmoid_rgb(self.final_layer(x))
        return rgb_sigma

class SHORTSIREN_FRes(nn.Module):
    """Direct feature volume with global features"""

    def __init__(
        self,
        input_dim=3,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
        **kwargs,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.            
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                SirenLayer(input_dim, hidden_dim, drop_out),
                ResSirenBlock(hidden_dim),
                SirenLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.network.apply(frequency_init(12))
        self.final_layer.apply(frequency_init(12))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, feature_volume, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for layer in self.network:
            x = layer(x)

        rgb_sigma = _sigmoid_rgb(self.final_layer(x))
        return rgb_sigma



class SingleSIREN_dg(nn.Module):
    """Direct feature volume with global features"""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.mapping_network = nn.Linear(z_dim, len(self.network) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, z, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        feature_volume, global_feature = z

        frequencies_offsets = self.mapping_network(global_feature)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]
        frequencies = frequencies * 15 + 30

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = points_feature
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = self.final_layer(x)
        return rgb_sigma


class TALLSIREN_dgx(nn.Module):
    """Direct feature volume with global features, the features are concatenated with position x"""

    def __init__(
        self,
        input_dim=3,
        z_dim=100,
        hidden_dim=256,
        output_dim=4,
        drop_out=0,
        device=None,
    ):
        """
        Args:
            input_dim (int, optional): dimension of input points in world space. Defaults to 3.
            z_dim (int, optional): dimension of the features in feature volume (must equal to out_channels in unet3d). Defaults to 100.
            hidden_dim (int, optional): dimension of intermediate layers of the FiLM-SIREN layers. Defaults to 256.
            output_dim (int, optional): output dimensions of siren (x,y,z,sigma). Defaults to 4.
            drop_out (int, optional): Drop out probability of siren layers. Defaults to 0.
            device (torch.device, optional): Defaults to None.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )

        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        # For color c output
        # self.color_layer_sine = FiLMLayer(
        #     hidden_dim + 3, hidden_dim
        # )  # the extra 3 dim at input is from the 3d ray direction
        # self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = nn.Linear(z_dim, len(self.network) * hidden_dim * 2)
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        # self.color_layer_sine.apply(frequency_init(25))
        # self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, points, z, img_size, num_steps):
        """
        Args:
            points (torch.Tensor: B, N(img_size**2*num_steps), 3): 3d points in world space
            z (tuple): output of the encoder
            img_size (int):
            num_steps (int): the number of steps to sample on each ray
        """
        feature_volume, global_feature = z

        frequencies_offsets = self.mapping_network(global_feature)
        frequencies = frequencies_offsets[..., : frequencies_offsets.shape[-1] // 2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1] // 2 :]
        frequencies = frequencies * 15 + 30

        voxel_length = 1.2  # use 1.2 because the voxel.npz is sampled within the 1.2^3 cube, and the grid in F.grid_sample has to be normalized
        batch_size = points.shape[0]
        grid = points / (voxel_length / 2)  # normalize the sampling point location
        grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature = points_feature.reshape(
            batch_size, feature_volume.shape[1], img_size**2 * num_steps
        ).permute(0, 2, 1)

        x = torch.cat([points_feature, points], -1)
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = self.final_layer(x)
        # freq = frequencies[..., -self.hidden_dim :]
        # phase = phase_shifts[..., -self.hidden_dim :]
        # rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), freq, phase)
        # rbg = self.color_layer_linear(rbg)
        return rgb_sigma


class SHORTSIREN(nn.Module):
    """Used together with pointnet encoder"""

    def __init__(
        self,
        input_dim=2,
        z_dim=100,
        hidden_dim=256,
        output_dim=1,
        drop_out=0,
        mapping_network="CustomMappingNetwork",
        device=None,
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(input_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
                FiLMLayer(hidden_dim, hidden_dim, drop_out),
            ]
        )
        # For density and color sigma output
        self.final_layer = nn.Linear(hidden_dim, 4)

        self.mapping_network = globals()[mapping_network](
            z_dim, 256, len(self.network) * hidden_dim * 2
        )
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, *args):
        # input: B, num_points, 3
        # z: B, latent_dim
        frequencies, phase_shifts = self.mapping_network(
            z
        )  # frequencies: B, 1280(256*(4+1)); phase_shifts:B, 1280
        frequencies = frequencies * 15 + 30
        x = input
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        rgb_sigma = _sigmoid_rgb(self.final_layer(x))
        return rgb_sigma


def _sigmoid_rgb(rgb_sigma):
    """
    Add sigmoid activation only for rgb component.
    """
    rgb = rgb_sigma[..., :3]
    sigma = rgb_sigma[..., -1:]
    rgb = F.sigmoid(rgb)
    return torch.cat([rgb, sigma], dim=-1)


def perturb(input, ratio, device):
    "perturb the input with a gaussian noise, the std of which is ratio to the magnitude of the input"
    noise_std = ratio * (input.max().item() - input.min().item())
    noise = torch.randn(input.shape, device=device) * noise_std
    # simply add the noise to the input
    return input + noise


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2 / sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(
        self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(3, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
            ]
        )
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(
            z_dim, 256, (len(self.network) + 1) * hidden_dim * 2
        )

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

        self.gridwarper = UniformBoxWarp(
            0.24
        )  # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(
            input, frequencies, phase_shifts, ray_directions, **kwargs
        )

    def forward_with_frequencies_phase_shifts(
        self, input, frequencies, phase_shifts, ray_directions, **kwargs
    ):
        frequencies = frequencies * 15 + 30

        input = self.gridwarper(input)
        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(
            torch.cat([ray_directions, x], dim=-1),
            frequencies[..., -self.hidden_dim :],
            phase_shifts[..., -self.hidden_dim :],
        )
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)


def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()

    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(
        grid.expand(batch_size, -1, -1, -1, -1),
        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H * W * D, C)
    return sampled_features


def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""

    def __init__(
        self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList(
            [
                FiLMLayer(32 + 3, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
                FiLMLayer(hidden_dim, hidden_dim),
            ]
        )
        print(self.network)

        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(
            z_dim, 256, (len(self.network) + 1) * hidden_dim * 2
        )

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96) * 0.01)

        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(
            input, frequencies, phase_shifts, ray_directions, **kwargs
        )

    def forward_with_frequencies_phase_shifts(
        self, input, frequencies, phase_shifts, ray_directions, **kwargs
    ):
        frequencies = frequencies * 15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(
            torch.cat([ray_directions, x], dim=-1),
            frequencies[..., -self.hidden_dim :],
            phase_shifts[..., -self.hidden_dim :],
        )
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)


class EmbeddingPiGAN256(EmbeddingPiGAN128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64) * 0.1)


def feature_pyramid_interpolation(
    points, feature_pyramid, img_size, num_steps, voxel_length=1.2
):
    """Intepolate at different levels of the PyramidUNet"""

    batch_size = points.shape[0]
    grid = points / (voxel_length / 2)  # normalize the sampling point location
    grid = grid.reshape(batch_size, img_size, img_size, num_steps, 3)

    if not isinstance(feature_pyramid, list):
        feature_pyramid = [feature_pyramid]

    points_feature_ls = []
    for feature_volume in feature_pyramid:
        # tri-linear interpolation
        points_feature = F.grid_sample(
            feature_volume,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        # reshape points_feature to: B, N, c
        points_feature_ls.append(
            points_feature.reshape(
                batch_size, feature_volume.shape[1], img_size**2 * num_steps
            ).permute(0, 2, 1)
        )
    # stack the features from different levels of the feature pyramid
    return torch.cat(points_feature_ls, dim=2)
