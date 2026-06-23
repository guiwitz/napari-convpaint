import torch
import timm
import numpy as np
from ..utils import get_device_from_torch_model, guided_model_download

# Weights are pulled from the open timm-mirrored HuggingFace repos. To add a
# variant, register it here and in AVAILABLE_MODELS. Requires timm >= 1.0.20.
def _hf_url(timm_name):
    return f"https://huggingface.co/timm/{timm_name}/resolve/main/model.safetensors"

DINOV3_MODELS = {
    'dinov3_small-plus': {
        'timm_name': 'vit_small_plus_patch16_dinov3.lvd1689m',
        'patch_size': 16,
        'embed_dim': 384,
    },
    # 'dinov3_vits16':     {'timm_name': 'vit_small_patch16_dinov3.lvd1689m',       'patch_size': 16, 'embed_dim': 384},
    # 'dinov3_vitb16':     {'timm_name': 'vit_base_patch16_dinov3.lvd1689m',        'patch_size': 16, 'embed_dim': 768},
    # 'dinov3_vitl16':     {'timm_name': 'vit_large_patch16_dinov3.lvd1689m',       'patch_size': 16, 'embed_dim': 1024},
    # 'dinov3_vith16plus': {'timm_name': 'vit_huge_plus_patch16_dinov3.lvd1689m',   'patch_size': 16, 'embed_dim': 1280},
}

AVAILABLE_MODELS = ['dinov3_small-plus']

STD_MODELS = {
    "dinov3": {"fe_name": "dinov3_small-plus"},
}

from ..feature_extractor import FeatureExtractor


class Dinov3Features(FeatureExtractor):
    """Feature extractor using DINOv3, a self-supervised vision transformer model from Meta AI Research.

    Loaded via the timm library (timm-mirrored HuggingFace weights, no gating).
    """

    def __init__(self, model_name='dinov3_small-plus', **kwargs):

        if model_name not in DINOV3_MODELS:
            raise ValueError(
                f"Unknown DINOv3 model '{model_name}'. Available: {list(DINOV3_MODELS)}"
            )
        spec = DINOV3_MODELS[model_name]

        super().__init__(model_name=model_name)

        self.patch_size = spec['patch_size']
        self.padding = 0  # final padding is automatically 1/2 patch size
        self.num_input_channels = [3]
        self.norm_mode = "imagenet"
        self.rgb_input = True
        self.proposed_scalings = [[1]]

        # CLS + register tokens prefix the patch tokens in forward_features output
        self.num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 5)

        self.device = get_device_from_torch_model(self.model)

    @staticmethod
    def create_model(model_name):
        spec = DINOV3_MODELS[model_name]
        # Pre-fetch with guided progress (mirrors DINOv2 UX), then point timm at the
        # local file instead of letting it download silently via huggingface_hub.
        weights_filename = f"{spec['timm_name'].replace('.', '_')}.safetensors"
        local_path = guided_model_download(weights_filename, _hf_url(spec['timm_name']))
        model = timm.create_model(
            spec['timm_name'],
            pretrained=False,
            num_classes=0,  # feature-extraction mode, no classifier head
            checkpoint_path=local_path,
        )
        model.eval()
        return model

    def get_description(self):
        desc = "Foundational ViT model (DINOv3). Extracts long range, semantic features."
        desc += "\nGood for: natural images (living beings, objects), histology etc."
        desc += "\n(The ViT-S+ version is used, with 4 register tokens and patch size 16x16.)"
        return desc

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_layers = None
        param.fe_scalings = [1]
        param.fe_order = 0
        param.tile_image = False
        param.tile_annotations = False
        return param

    def get_enforced_params(self, param=None):
        param = super().get_enforced_params(param=param)
        param.fe_scalings = [1]
        return param

    def extract_features_from_stack(self, image, device=torch.device("cpu"), **kwargs):
        # Stack passed as a single batched tensor.
        self.move_model_to_device(device)

        image_tensor = self.prep_img(image)

        with torch.no_grad():
            features = self.model.forward_features(image_tensor)
        # timm returns [B, num_prefix_tokens + N_patches, D] — drop the prefix tokens.
        features = features[:, self.num_prefix_tokens:, :]

        features = features.detach().cpu().numpy()
        features = np.moveaxis(features, -1, 0)

        patch_size = self.get_patch_size()
        features_shape = (features.shape[0],
                          image.shape[1],
                          int(image.shape[2] / patch_size),
                          int(image.shape[3] / patch_size))

        features = np.reshape(features, features_shape)

        assert features.shape[-2] == image.shape[-2] / patch_size
        assert features.shape[-1] == image.shape[-1] / patch_size

        return [features]

    def prep_img(self, image):

        patch_size = self.get_patch_size()
        assert len(image.shape) == 4
        assert image.shape[0] == 3
        assert image.shape[-2] % patch_size == 0
        assert image.shape[-1] % patch_size == 0

        h, w = image.shape[-2:]
        new_h = h - (h % patch_size)
        new_w = w - (w % patch_size)
        if new_h != h or new_w != w:
            image = image[:, :new_h, :new_w]

        # Treat z as batch dimension (temporarily).
        image = np.moveaxis(image, 1, 0)

        image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)

        return image_tensor
