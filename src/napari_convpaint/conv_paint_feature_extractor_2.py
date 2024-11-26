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
    def get_features(self, image):
        """
        Gets the features of an image.

        Parameters:
        - image: The input image. Dimensions are [nb_channels, width, height]

        Returns:
        - features: The extracted features of the image. [nb_features, width, height]
        """
        pass

    def get_default_param(self):
        """
        Get the defaul parameters that the FE shall set/enforce when the FE model is set.
        """
        param = Param()
        # FE params (encouraged to set)
        param.fe_name: str = self.model_name
        param.fe_layers: list[str] = None
        param.fe_scalings: list[int] = [1]
        param.fe_order: int = 0
        param.fe_use_min_features: bool = False
        param.fe_use_cuda: bool = False
        # General settings (NOTE: only set if shall be enforced by FE !)
        param.multi_channel_img: bool = None # use multichannel if image dimensions allow
        param.normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image
        param.image_downsample: int = None
        param.tile_annotations: bool = None
        param.tile_image: bool = None
        # Classifier
        param.classifier: str = None
        # Classifier parameters
        param.clf_iterations: int = None
        param.clf_learning_rate: float = None
        param.clf_depth: int = None

        return param
    
    def get_enforced_param(self, param):
        """
        Define which parameters need to be absolutley enforced for this feature extractor.
        """
        return param
    
    def get_features_scaled(self, image, param, **kwargs):
        """
        Given a filter model and an image, extract features at different scales.

        Parameters
        ----------
        image: 2d array
            image to segment
        param: Param
            object containing the parameters for the feature extraction

        Returns
        -------
        features: [nb_features x width x height]
            return extracted features

        """

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if param.image_downsample > 1:
            image = image[:, ::param.image_downsample, ::param.image_downsample]

        padding = param.fe_padding
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')

        features_all_scales = []
        for s in param.fe_scalings:
            image_scaled = image[:, ::s, ::s]
            features = self.get_features(image_scaled, order=param.fe_order, **kwargs) #features have shape [nb_features, width, height]
            nb_features = features.shape[0]
            features = skimage.transform.resize(
                                image=features,
                                output_shape=(nb_features, image.shape[-2], image.shape[-1]),
                                preserve_range=True,
                                order=param.fe_order)
            
            features_all_scales.append(features)
        features_all_scales = np.concatenate(features_all_scales, axis=0)
        if padding > 0:
            features_all_scales = features_all_scales[:, padding:-padding, padding:-padding]
        return features_all_scales

