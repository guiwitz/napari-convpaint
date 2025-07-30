import skimage
import numpy as np
from .conv_paint_feature_extractor import FeatureExtractor

AVAILABLE_MODELS = ['gaussian_features']

class GaussianFeatures(FeatureExtractor):
    def __init__(self, model_name='gaussian_features', use_gpu=False, sigma=3, **kwargs):
        super().__init__(model_name=model_name, use_gpu=use_gpu)
        self.sigma = sigma
        self.padding = 2*sigma # Padding established empirically to avoid edge effects

    def get_description(self):
        return "Minimal model to easily and rapidly extract basic image features."

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_layers = []
        return param

    def get_features_from_plane(self, image):
        
        # Given that we get single-channel images as input:
        features = skimage.filters.gaussian(image, sigma=self.sigma, channel_axis=0)

        return features