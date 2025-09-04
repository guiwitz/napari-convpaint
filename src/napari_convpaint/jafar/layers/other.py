import torch.nn as nn


class SFTModulation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.gamma = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.beta = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels, affine=False)

    def forward(self, image, features):
        gamma = self.gamma(features)
        beta = self.beta(features)
        return gamma * self.norm(image) + beta  # Spatial modulation
