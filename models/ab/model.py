"""
Network of models A and B: a 3D residual U-Net predicts a stationary velocity field; the field is
smoothed on a B-spline control-point grid and integrated with forward Euler steps to move the
template vertices.

The residual U-Net is partly taken from Vox2Cortex (vox2organ/models/u_net.py,
https://github.com/ai-med/Vox2Cortex, GPL-3.0), Copyright (C) Fabian Bongratz; residual blocks after
nnU-Net (https://github.com/MIC-DKFZ/nnUNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bspline import BsplineTransformation


class ResidualBlock(nn.Module):
    """Residual block of https://arxiv.org/abs/1908.02182: conv, norm, leaky ReLU, dropout, conv, norm, + skip."""

    def __init__(self, c_in, c_out, p_dropout):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(c_out)
        self.dropout = nn.Dropout(p_dropout)
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(c_out)
        # 1x1x1 conv to adapt the channels of the residual
        if c_in != c_out:
            self.adapt_skip = nn.Sequential(nn.Conv3d(c_in, c_out, 1, bias=False), nn.InstanceNorm3d(c_out))
        else:
            self.adapt_skip = nn.Identity()

    def forward(self, x):
        x_out = F.leaky_relu(self.norm1(self.conv1(x)))
        x_out = self.dropout(x_out)
        x_out = self.norm2(self.conv2(x_out))
        x_out += self.adapt_skip(x)
        return F.leaky_relu(x_out)


class ResidualUNet(nn.Module):
    """Encoder: conv + residual block, then (2x2x2 strided conv + residual block) per level.
    Decoder: transposed conv, concatenation with the encoder skip, residual block.
    Output: 3 channels (velocity), last layer initialised to zero (identity deformation at start)."""

    def __init__(self, in_channels, encoder_channels, decoder_channels, p_dropout):
        super().__init__()
        enc = encoder_channels
        down = [nn.Sequential(nn.Conv3d(in_channels, enc[0], 3, padding=1), ResidualBlock(enc[0], enc[0], p_dropout))]
        for i in range(1, len(enc)):
            down.append(nn.Sequential(nn.Conv3d(enc[i - 1], enc[i], 2, stride=2),
                                      ResidualBlock(enc[i], enc[i], p_dropout)))
        self.encoder = nn.ModuleList(down)

        dec, skips = decoder_channels, enc[::-1]
        up = []
        for i in range(len(enc) - 1):
            c_prev = enc[-1] if i == 0 else dec[i - 1]
            up.append(nn.Sequential(nn.ConvTranspose3d(c_prev, dec[i], kernel_size=2, stride=2),
                                    ResidualBlock(dec[i] + skips[i + 1], dec[i], p_dropout)))
        self.decoder = nn.ModuleList(up)
        self.final_layer = nn.Conv3d(dec[-1], 3, kernel_size=1, bias=False)
        nn.init.constant_(self.final_layer.weight, 0.0)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x = layer(x)
            skips.append(x)
        x = skips[-1]
        for i, layer in enumerate(self.decoder):
            x = torch.cat([layer[0](x), skips[-2 - i]], dim=1)
            x = layer[1](x)
        return self.final_layer(x)


def sample_field(field, points):
    """Trilinear interpolation of field (B, C, D, H, W) at points (B, N, 3) in voxel coordinates."""
    size = torch.tensor([[[s - 1 for s in field.shape[2:]]]], device=points.device).float()
    grid = (2 * (points / size - 0.5))[:, :, None, None].flip(dims=(-1,))
    out = F.grid_sample(field, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out.squeeze().view(field.shape[0], field.shape[1], points.shape[1])


class TemplateDeformation(nn.Module):
    """Velocity field (B, 3, D, H, W) -> control points (spacing cp_spacing voxels) -> dense B-spline
    field -> integration_steps forward Euler steps of the template points (B, N, 3)."""

    def __init__(self, patch_size, cp_spacing, bspline_order, integration_steps, device):
        super().__init__()
        self.bspline = BsplineTransformation(patch_size, sigma=cp_spacing, order=bspline_order, device=device)
        self.steps = integration_steps

    def forward(self, velocity, points):
        b = velocity.shape[0]
        cp_idx = F.affine_grid(torch.eye(4, device=velocity.device).repeat([b, 1, 1])[:, 0:3, :],
                               self.bspline.cp_grid_shape, align_corners=True)
        flow = self.bspline(F.grid_sample(velocity, cp_idx, align_corners=True)) * (1.0 / self.steps)
        for _ in range(self.steps):
            points = points + sample_field(flow, points).transpose(2, 1)
        return points
