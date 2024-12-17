import numpy as np
import pandas as pd
import skimage
from .conv_paint_param import Param
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kernel_size = None;
        self.patch_size = None;

    @abstractmethod
    def extract_features_single_channel(self, image, param, **kwargs):
        pass

    def extract_features_rgb(self, image, param, **kwargs):
        # Ensure the image dimensions are compatible (put Channels first)
        if image.shape[0] != 3:
            raise ValueError("RGB image must have 3 channels (and dimensions must be [C, H, W]).")
        # DEFAULT: Extract features for each channel and concatenate them
        return self.extract_features_from_channels(image, param, **kwargs)
    
    def extract_features_from_channels(self, image, param, **kwargs):
        all_features = []
        for channel in image:
            features = self.extract_features_single_channel(channel, param, **kwargs)
            all_features.append(features)
        return np.concatenate(all_features, axis=0)

    def get_features(self, image, rgb=False, **kwargs):
        """
        Gets the features of an image.

        Parameters:
        - image: The input image.
            Dimensions are [C, H, W] or [C, Z, H, W] (for Feature Extractors that support 3D extraction)
        - rgb: Whether the image shall be treated as RGB format.

        Returns:
        - features: The extracted features of the image. [nb_features, width, height]
        """
        # Check that the image is 3D (or 4D)
        if image.ndim != 3 and image.ndim != 4:
            raise ValueError("Image must have 3 dimensions (or 4 for 3D Feature Extractors).")

        # For RGB images, extract features with the RGB method
        if rgb:
            return self.extract_features_rgb(image, self.param, **kwargs)

        # Otherwise, extract features for each channel and concatenate them
        return self.extract_features_from_channels(image, self.param, **kwargs)

    def get_default_param(self, param=None):
        """
        Get the defaul parameters that the FE shall set/enforce when the FE model is set.
        """

        # TAKE FROM OLD FE IMPLEMENTATION

        return param
    
    def get_enforced_param(self, param=None):
        """
        Define which parameters need to be absolutley enforced for this feature extractor.
        """
        if param is None:
            param = Param()
        return param

    def get_effective_kernel_padding(self):
        """
        Get the effective kernel padding for the feature extractor, that should be added on each side
        taking into account the scaling factor for later pyramid scaling.
        This assures that at the edge of the image, the kernel is still centered.
        """
        if self.kernel_size is None:
            return None
        kernel_size = self.kernel_size
        scale_factor = np.max(self.param.fe_scalings)
        # Compute padding for each dimension
        effective_z_pad = (kernel_size[0] // 2) if kernel_size[0] is not None else 0 # No scaling for z
        effective_h_pad = (kernel_size[1] // 2) * scale_factor
        effective_w_pad = (kernel_size[2] // 2) * scale_factor
        return (effective_z_pad, effective_h_pad, effective_w_pad)

    def get_effective_patch_size(self):
        """
        Get the effective patch size for the feature extractor,
        taking into account the scaling factor for later pyramid scaling.
        This assures that the scaled image is a multiple of the patch size.
        """
        if self.patch_size is None:
            return None
        patch_size = self.patch_size
        scale_factor = np.max(self.param.fe_scalings)
        # Compute padding for each dimension
        effective_patch_z = patch_size[0] if patch_size[0] is not None else None # No scaling for z
        effective_patch_h = (patch_size[1]*scale_factor)
        effective_patch_w = (patch_size[2]*scale_factor)
        return (effective_patch_z, effective_patch_h, effective_patch_w)

