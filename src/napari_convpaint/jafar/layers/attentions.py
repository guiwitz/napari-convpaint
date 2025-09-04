import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super().__init__()
        self.norm_q = nn.RMSNorm(query_dim)
        self.norm_k = nn.RMSNorm(key_dim)
        self.norm_v = nn.RMSNorm(value_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            vdim=value_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(self, query, key, value):
        query = self.norm_q(query)
        key = self.norm_k(key)

        _, attn_scores = self.attention(query, key, self.norm_v(value), average_attn_weights=True)
        attn_output = einsum("b i j, b j d -> b i d", attn_scores, value)

        return attn_output, attn_scores


class CrossAttentionBlock(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim, num_heads, **kwargs):
        super().__init__()

        self.cross_attn = CrossAttention(
            query_dim,
            key_dim,
            value_dim,
            num_heads,
        )
        self.conv2d = nn.Conv2d(query_dim, query_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, **kwargs):
        q = self.conv2d(q)
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b (h w) c")
        v = rearrange(v, "b c h w -> b (h w) c")
        features, _ = self.cross_attn(q, k, v)

        return features
