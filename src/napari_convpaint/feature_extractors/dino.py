import torch
import numpy as np
from ..utils import get_device_from_torch_model, scale_img, guided_model_download

DINOV2_MODELS = {
    'dinov2_small-reg': {
        'internal_name': 'dinov2_vits14_reg',
        'patch_size': 14,
        'embed_dim': 384,
    }
}

AVAILABLE_MODELS = ['dinov2_small-reg']

STD_MODELS = {
    "dinov2": {"fe_name": "dinov2_small-reg"},
}

from ..feature_extractor import FeatureExtractor

class Dinov2Features(FeatureExtractor):
    """Feature extractor using DINOv2, a self-supervised vision transformer model from Facebook AI Research (Meta)."""
    
    def __init__(self, model_name='dinov2_small-reg', **kwargs):

        if model_name not in DINOV2_MODELS:
            raise ValueError(
                f"Unknown DINOv2 model '{model_name}'. Available: {list(DINOV2_MODELS)}"
            )
        spec = DINOV2_MODELS[model_name]
        
        super().__init__(model_name=model_name)
        
        self.patch_size = spec['patch_size']
        self.padding = 0 # Note: final padding is automatically 1/2 patch size
        self.num_input_channels = [3] # RGB
        self.norm_mode = "imagenet"  # DINOv2 expects ImageNet normalization
        self.rgb_input = True  # DINOv2 expects RGB input
        self.proposed_scalings = [[1]]

        # Register the device of the created model
        self.device = get_device_from_torch_model(self.model)

    @staticmethod
    def create_model(model_name):
        spec = DINOV2_MODELS[model_name]
        internal_model_name = spec['internal_name']
        # Validate for forks to prevent rate limit error on GitHub Actions: https://github.com/pytorch/pytorch/issues/61755
        torch.hub._validate_not_a_forked_repo=lambda a, b, c: True

        # Extract model backbone name
        model_backbone = internal_model_name.split('_')[0] + "_" + internal_model_name.split('_')[1]

        model_file = f"{internal_model_name}4_pretrain.pth"
        model_url = f"https://dl.fbaipublicfiles.com/dinov2/{model_backbone}/{model_file}"

        # Ensure weights are downloaded
        _ = guided_model_download(model_file, model_url)

        model = torch.hub.load('facebookresearch/dinov2', internal_model_name, pretrained=True, verbose=False)

        # Set model to evaluation mode. Device is selected at feature extraction time.
        model.eval()

        return model

    def get_description(self):
        desc = "Foundational ViT model. Extracts long range, semantic features."
        desc += "\nGood for: natural images (living beings, objects), histology etc."
        desc += "\n(The small version is used, with registers and patch size 14x14.)"
        return desc

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_layers = None
        param.fe_scalings = [1]
        param.fe_order = 0
        # param.image_downsample = 1
        param.tile_image = False
        param.tile_annotations = False
        # param.normalize = 1
        return param
    
    def get_enforced_params(self, param=None):
        param = super().get_enforced_params(param=param)
        param.fe_scalings = [1]
        return param

    def extract_features_from_stack(self, image, device=torch.device("cpu"), **kwargs):
        # NOTE: Use this method, as it can pass a stack as a tensor, processing it as a batch.
        self.move_model_to_device(device)
        
        # Prepare the image (normalize etc.)
        image_tensor = self.prep_img(image)

        with torch.no_grad():
            features_dict = self.model.forward_features(image_tensor)
        features = features_dict['x_norm_patchtokens']

        # Move features first, and reshape to spatial dimensions
        features = features.detach().cpu().numpy()
        features = np.moveaxis(features, -1, 0)

        patch_size = self.get_patch_size()
        features_shape = (features.shape[0],
                          image.shape[1],
                          int(image.shape[2] / patch_size),
                          int(image.shape[3] / patch_size))
        
        features = np.reshape(features, features_shape)

        # Test if the features are the right size
        assert features.shape[-2] == image.shape[-2] / patch_size
        assert features.shape[-1] == image.shape[-1] / patch_size

        return [features]

    def prep_img(self, image):
    
        patch_size = self.get_patch_size()
        assert len(image.shape) == 4
        assert image.shape[0] == 3
        assert image.shape[-2] % patch_size == 0
        assert image.shape[-1] % patch_size == 0

        # Crop image to make sure it is divisible by patch size
        # NOTE: This is not necessary (it's old code), but it does not hurt either
        h, w = image.shape[-2:]
        new_h = h - (h % patch_size)
        new_w = w - (w % patch_size)
        if new_h != h or new_w != w:
            image = image[:, :new_h, :new_w]

        # Treat z as batch dimension (temporarily)
        image = np.moveaxis(image, 1, 0)

        # Convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32, device=self.device)

        return image_tensor