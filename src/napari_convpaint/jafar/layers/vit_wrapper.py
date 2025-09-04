import re
import types
from typing import List, Tuple, Union

import timm
import timm.data
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torchvision import transforms 

# https://github.com/Jiawei-Yang/Denoising-ViT/blob/82704df9ba253c9696dcf3a8239434e3cbacf19d/dvt/models/vit_wrapper.py#L59

MODEL_LIST = [
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch16_224.dino",
    "vit_small_patch14_reg4_dinov2",
    "vit_small_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch16_clip_384",
    "vit_base_patch16_siglip_512.v2_webli",
    "vit_base_patch16_224",
]


def get_patch_size_channels(backbone_name):
    if "vit_small" in backbone_name:
        feats = 384
    elif "vit_base" in backbone_name:
        feats = 768
        if backbone_name == "vit_base_patch16_clip_384":
            feats = 384
    else:
        raise ValueError(f"Backbone name {backbone_name} not supported")

    if "patch14" in backbone_name:
        patch_size = 14
    elif "patch16" in backbone_name:
        patch_size = 16
    else:
        raise ValueError(f"Backbone name {backbone_name} not supported")

    return patch_size, feats


class PretrainedViTWrapper(nn.Module):

    def __init__(
        self,
        name,
        norm: bool = True,
        stride = None,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs,
    ):
        super().__init__()
        # comment out the following line to test the models not in the list
        assert name in MODEL_LIST, f"Model type {name} not tested yet."
        self.name = name
        self.patch_size = int(re.search(r"patch(\d+)", name).group(1))
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.model, self.config = self.create_model(name, **kwargs)
        self.embed_dim = self.model.embed_dim
        self.norm = norm

        if not stride:
            self.stride = self.model.patch_embed.proj.stride[0]

        # overwrite the stride size
        if stride and stride != self.model.patch_embed.proj.stride[0]:
            self.set_stride(stride)

    def set_stride(self, stride: int):
        """
        Sets the stride for the patch embedding layer and updates related properties.

        Args:
            stride (int): The new stride value to set.
        """
        # Update the stride in the patch embedding layer
        self.model.patch_embed.proj.stride = [stride, stride]
        self.stride = stride

        # Dynamically update the feature size calculation
        def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
            """
            Dynamically calculates the feature map size based on the input image size and stride.

            Args:
                img_size (Tuple[int, int]): The input image size (height, width).

            Returns:
                Tuple[int, int]: The feature map size (height, width).
            """
            return (
                (img_size[0] - self.patch_size[0]) // self.proj.stride[0] + 1,
                (img_size[1] - self.patch_size[1]) // self.proj.stride[1] + 1,
            )

        # Bind the dynamic_feat_size method to the patch embedding layer
        self.model.patch_embed.dynamic_feat_size = types.MethodType(dynamic_feat_size, self.model.patch_embed)

    @property
    def n_output_dims(self) -> int:
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(self, name: str, **kwargs) -> Tuple[VisionTransformer, transforms.Compose]:
        model = timm.create_model(
            name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=self.dynamic_img_size,
            dynamic_img_pad=self.dynamic_img_pad,
            **kwargs,
        )
        model = model.eval()
        # Different models have different data configurations
        # e.g., their training resolution, normalization, etc, are different
        data_config = timm.data.resolve_model_data_config(model=model)
        return model, data_config

    def forward(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Intermediate layer accessor inspired by DINO / DINOv2 interface.
        Args:
            x: Input tensor.
            n: Take last n blocks if int, all if None, select matching indices if sequence
            reshape: Whether to reshape the output.
        """
        if self.name in ["vit_base_patch16_siglip_512.v2_webli"]:
            feats = self.model.forward_intermediates(
                x,
                n,
                return_prefix_tokens=True,
                norm=self.norm,
                output_fmt="NCHW",
                intermediates_only=True,
            )[0]
            cls_token = self.model.pool(feats.permute(0, 2, 3, 1).flatten(1, 2))
        else:
            feats, cls_token = self.model.forward_intermediates(
                x,
                n,
                return_prefix_tokens=True,
                norm=self.norm,
                output_fmt="NCHW",
                intermediates_only=True,
            )[0]
        return feats, cls_token
