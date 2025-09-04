#import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch

from napari_convpaint.jafar.layers import CrossAttentionBlock, ResBlock, SFTModulation
from napari_convpaint.jafar.layers.positional_encoding import RoPE

def create_coordinate(h, w, start=0, end=1, device="cpu", dtype=torch.float32):
    # Create a grid of coordinates
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    # Create a 2D map using meshgrid
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    # Stack the x and y coordinates to create the final map
    coord_map = torch.stack([xx, yy], axis=-1)[None, ...]
    coords = rearrange(coord_map, "b h w c -> b (h w) c", h=h, w=w)
    return coords


class JAFAR(nn.Module):

    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        v_dim=384,
        feature_dim=None,
        kernel_size=1,
        num_heads=4,
        **kwargs,
    ):
        super().__init__()

        def make_encoder(in_dim, kernel_size, num_layers=2):
            return nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    qk_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode="reflect",
                    bias=False,
                ),
                *[
                    ResBlock(
                        qk_dim,
                        qk_dim,
                        kernel_size=1,
                        num_groups=8,
                        pad_mode="reflect",
                        norm_fn=nn.GroupNorm,
                        activation_fn=nn.SiLU,
                        use_conv_shortcut=False,
                    )
                    for _ in range(num_layers)
                ],
            )

        # Image encoder uses kernel_size=3 for spatial context
        self.image_encoder = make_encoder(input_dim, kernel_size=kernel_size)
        self.key_encoder = make_encoder(qk_dim, kernel_size=1)
        self.query_encoder = make_encoder(qk_dim, kernel_size=1)

        # Create Query features encoder
        self.norm = nn.GroupNorm(num_groups=8, num_channels=qk_dim, affine=False)

        # Create Key features encoder
        self.key_features_encoder = make_encoder(v_dim, kernel_size=1)
        self.cross_decode = CrossAttentionBlock(qk_dim, qk_dim, v_dim, num_heads)

        # SFT modulation for keys
        self.sft_key = SFTModulation(qk_dim, qk_dim)

        self.rope = RoPE(qk_dim)
        self.rope._device_weight_init()

    def upsample(self, encoded_image, features, output_size):
        _, _, h, w = features.shape

        # Process Queries
        queries = self.query_encoder(encoded_image)
        queries = F.adaptive_avg_pool2d(queries, output_size=output_size)
        queries = self.norm(queries)

        # Process Keys and Values.
        keys = self.key_encoder(encoded_image)
        keys = F.adaptive_avg_pool2d(keys, output_size=(h, w))
        keys = self.sft_key(keys, self.key_features_encoder(F.normalize(features, dim=1)))

        # Values
        values = features

        # Attention layer
        out = self.cross_decode(queries, keys, values)

        return out

    def forward(self, image, features, output_size):
        # 1) Extract high-level features
        encoded_image = self.image_encoder(image)         # (B,C,H,W)
        device        = encoded_image.device              # ← MPS

        # 2) Build / move the coordinate grid on the **same device**
        H, W = encoded_image.shape[-2:]
        coords = create_coordinate(H, W).to(device, encoded_image.dtype)

        # 3) Apply RoPE
        encoded_image = rearrange(encoded_image, "b c h w -> b (h w) c")
        encoded_image = self.rope(encoded_image, coords)  # coords now on MPS
        encoded_image = rearrange(encoded_image, "b (h w) c -> b c h w", h=H)

        # 4) Upsample & reshape
        features = self.upsample(encoded_image, features, output_size)
        features = rearrange(features, "b (h w) c -> b c h w", h=output_size[0])
        return features