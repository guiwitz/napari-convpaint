import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum


# Convolutions
class ResBlock(nn.Module):
    """Basic Residual Block, adapted from magvit1"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super(ResBlock, self).__init__()
        self.use_conv_shortcut = use_conv_shortcut
        self.norm1 = norm_fn(num_groups, in_channels) if norm_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.norm2 = norm_fn(num_groups, out_channels) if norm_fn is not None else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=pad_mode,
            bias=False,
        )
        self.activation_fn = activation_fn()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                padding_mode=pad_mode,
                bias=False,
            )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        if self.use_conv_shortcut or residual.shape != x.shape:
            residual = self.shortcut(residual)
        return x + residual
