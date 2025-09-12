import torch
import numpy as np
from .conv_paint_utils import get_device, get_device_from_torch_model, scale_img, guided_model_download
from .conv_paint_feature_extractor import FeatureExtractor

AVAILABLE_MODELS = ['dinov2_vits14_reg']

class DinoFeatures(FeatureExtractor):
    def __init__(self, model_name='dinov2_vits14_reg', use_gpu=False):
        
        # Sets self.model_name and self.use_gpu and creates the model
        super().__init__(model_name=model_name, use_gpu=use_gpu)
        
        self.patch_size = 14
        self.padding = 0 # Note: final padding is automatically 1/2 patch size
        self.num_input_channels = [3]
        self.norm_imagenet = True

        # Register the device of the created model
        self.device = get_device_from_torch_model(self.model)

    @staticmethod
    def create_model(model_name, use_gpu=False):
        # Validate for forks to prevent rate limit error on GitHub Actions: https://github.com/pytorch/pytorch/issues/61755
        torch.hub._validate_not_a_forked_repo=lambda a, b, c: True

        # Extract model backbone name
        model_backbone = model_name.split('_')[0] + "_" + model_name.split('_')[1]

        model_file = f"{model_name}4_pretrain.pth"
        model_url = f"https://dl.fbaipublicfiles.com/dinov2/{model_backbone}/{model_file}"

        # Ensure weights are downloaded
        _ = guided_model_download(model_file, model_url)

        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True, verbose=False)

        # Set model to evaluation mode and move it to the correct device
        model.eval()
        model = model.to(get_device(use_gpu))

        return model

    def get_description(self):
        desc = "Foundational ViT model. Extracts long range, semantic features."
        desc += "\nGood for: natural images (living beings, objects), histology etc."
        desc += "\n(The small version is used, with registers and patch size 14x14.)"
        return desc

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_use_gpu = self.use_gpu
        param.fe_layers = []
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

    def get_features(self, image, **kwargs):
        # NOTE: Use this method, as it can pass a stack as a tensor, processing it as a batch.
        
        # Prepare the image (normalize etc.)
        image_tensor = self.prep_img(image)

        with torch.no_grad():
            features_dict = self.model.forward_features(image_tensor)
        features = features_dict['x_norm_patchtokens']

        if self.use_gpu:
            features = features.cpu()

        # Move features first, and reshape to spatial dimensions
        features = features.numpy()
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