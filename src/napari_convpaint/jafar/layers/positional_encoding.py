import torch
import torch.nn as nn
from einops import rearrange

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        # Create freqs in 1d
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 4)
        # duplicate freqs for rotation pairs of channels
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        # First half of channels do x, second half do y
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        # it's an angular freq here
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()
