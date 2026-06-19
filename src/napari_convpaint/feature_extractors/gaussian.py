import math
import skimage
import numpy as np

AVAILABLE_MODELS = ['gaussian_features']

STD_MODELS = {
    "gaussian": {"fe_name": "gaussian_features"},
}

from ..feature_extractor import FeatureExtractor

class GaussianFeatures(FeatureExtractor):
    """A simple feature extractor that applies a Gaussian filter to the input image."""
    def __init__(self, model_name='gaussian_features', sigma=3, **kwargs):
        super().__init__(model_name=model_name)
        self.sigma = sigma
        # skimage.filters.gaussian uses truncate=4.0 by default, so each
        # output pixel reads `ceil(4 * sigma)` input pixels on each side.
        self.padding = math.ceil(4 * sigma)

    def get_description(self):
        return "Minimal model to easily and rapidly extract basic image features."

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_layers = None
        return param

    def extract_features_from_plane(self, image, device=None):
        
        # Given that we get single-channel images as input:
        features = skimage.filters.gaussian(image, sigma=self.sigma, channel_axis=0)

        return features