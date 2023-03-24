import torch
import torch.nn as nn
from glob import glob
import numpy as np
from .unet3d import UNet3D

"Model adapted from OccupancyNetworks"
"https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/im2mesh/encoder/pointnet.py"
# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def maxpool(x, dim=-1, keepdim=False):
    out, idx = x.max(dim=dim, keepdim=keepdim)
    return out, idx
    
class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=512, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True)[0].expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True)[0].expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True)[0].expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled= self.pool(net, dim=1, keepdim=True)[0].expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Reduce to  B x F
        net, _ = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c

    def forward_with_idx(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled, idx0 = self.pool(net, dim=1, keepdim=True)
        net = torch.cat([net, pooled.expand(net.size())], dim=2)

        net = self.block_1(net)
        pooled, idx1 = self.pool(net, dim=1, keepdim=True)
        net = torch.cat([net, pooled.expand(net.size())], dim=2)

        net = self.block_2(net)
        pooled, idx2 = self.pool(net, dim=1, keepdim=True)
        net = torch.cat([net, pooled.expand(net.size())], dim=2)

        net = self.block_3(net)
        pooled, idx3 = self.pool(net, dim=1, keepdim=True)
        net = torch.cat([net, pooled.expand(net.size())], dim=2)

        net = self.block_4(net)

        # Reduce to  B x F
        net, idx4 = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))
        
        return c, (idx0, idx1, idx2, idx3, idx4)

# unet_params = {
#     "out_channels": 32,
#     "f_maps": 32,
#     "num_levels": 3,
#     "is_segmentation": False,
#     "final_sigmoid": False,
# }

# class VoxelEncoder(nn.Module):
#     def __init__(self, dim=4, c_dim=32, kernel_size=2):
#         super().__init__()
#         self.conv_in = nn.Conv3d(dim, c_dim, kernel_size, padding=0, stride=2)
#         self.unet3d = UNet3D(in_channels=dim, **unet_params)
#         self.actvn = nn.ReLU()

#     def forward(self, x):
#         # x: B, 4, 64, 64, 64
#         c = self.actvn(self.conv_in(x)) # c: B, c_dim, 32, 32, 32
#         c = self.unet3d(c) # c: B, out_channels, 32, 32, 32
#         return c

def encode_pcl(
    encoder: ResnetPointnet,
    pcl: torch.Tensor,
    device: torch.device,
    noise_weight: float = 0,
) -> torch.Tensor:
    """Encode the point cloud with encoder and add noise the the latent codes"""
    # note .to(device) is already done at encoder
    pcl = pcl.to(device)
    encodes = encoder(pcl)
    l_reg = encodes.norm(dim=1).mean()
    encodes -= encodes.mean(dim=1, keepdim=True)
    # might need to add an epsilon to avoid divsion by 0
    encodes = encodes / encodes.std(dim=1, keepdim=True)
    noise = torch.randn(encodes.shape, device=device)
    z = noise_weight * noise + (1 - noise_weight) * encodes
    return z, l_reg
