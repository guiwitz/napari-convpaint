import skimage
import numpy as np
from .conv_paint_feature_extractor import FeatureExtractor

AVAILABLE_MODELS = ['gaussian_features']

class GaussianFeatures(FeatureExtractor):
    def __init__(self, model_name='gaussian_features', use_cuda=False, sigma=3, **kwargs):
        super().__init__(model_name=model_name, use_cuda=use_cuda)
        self.sigma = sigma
        self.padding = sigma

    def get_description(self):
        return "Minimal model to easily and rapidly extract basic image features."

    def get_features(self, image, **kwargs):
        im_filter = skimage.filters.gaussian(image, sigma=self.sigma, channel_axis=0)
        return im_filter
    
    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_layers = []
        return param
