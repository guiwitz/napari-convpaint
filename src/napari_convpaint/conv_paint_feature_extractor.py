import numpy as np
import pandas as pd
import skimage
from .conv_paint_param import Param
from .conv_paint_utils import scale_img

class FeatureExtractor:
    def __init__(self, model_name="vgg16", model=None, use_cuda=None, **kwargs):
        """
        Initializing a feature extractor. This is a superclass for all feature extractors.

        Parameters:
        ----------
        model_name : str
            The name of the model to use. If model is not None, this parameter is ignored.
        model : object
            The model to use. If not None, this model is used instead of loading a new model.
        use_cuda : bool
            Whether to use CUDA or not. If not provided, the default is False.
        """

        self.model_name = model_name
        self.use_cuda = use_cuda
        self.padding = 0
        self.patch_size = 1

        # USE PROVIDED MODEL IF AVAILABLE
        if model is not None:
            assert model_name is None, 'model_name must be None if model is not None'
            self.model = model

        # ELSE CREATE MODEL
        else:
            self.model = self.create_model(model_name)

    @staticmethod
    def create_model(model_name):
        """
        This method is intended to be overridden by subclasses to load the specific model.
        """
        return None
    
    def get_description(self):
        """
        Returns a description of the feature extractor.
        """
        return "This is a generic feature extractor. Subclasses should override this method."

    def get_features(self, image):
        """
        Gets the features of an image.

        Parameters:
        ----------
        image : np.ndarray
            The input image. Dimensions are [nb_channels, width, height]

        Returns:
        features : np.ndarray
            The extracted features of the image. [nb_features, width, height]
        """
        raise NotImplementedError("Subclasses must implement get_features method.")

    def get_default_params(self, param=None):
        """
        Get the default parameters that the FE shall set/enforce when chosen by the user.
        These can still be changed by the user afterwards.
        """
        if param is None:
            def_param = Param()
        else:
            def_param = param.copy()
            
        # FE params (encouraged to set)
        def_param.fe_name= self.model_name
        # param.fe_layers: list[str] = None
        def_param.fe_use_cuda = False
        def_param.fe_scalings = [1]
        def_param.fe_order = 0
        def_param.fe_use_min_features = False

        # # General settings (NOTE: only set if shall be enforced by FE !)
        # param.multi_channel_img: bool = None # use multichannel if image dimensions allow
        # param.rgb_img: bool = None # use RGB image (note: should not be set, as the input defines this mode)
        # param.normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image
        # param.image_downsample: int = None
        # param.seg_smoothening: int = None
        # param.tile_annotations: bool = None
        # param.tile_image: bool = None
        # # Classifier
        # param.classifier: str = None
        # # Classifier parameters
        # param.clf_iterations: int = None
        # param.clf_learning_rate: float = None
        # param.clf_depth: int = None

        return def_param
    
    def get_layer_keys(self):
        """
        If the model has layers, return the keys of the layers.
        """
        return None
    
    def get_enforced_params(self, param=None):
        """
        Define which parameters need to be absolutley enforced for feature extraction.
        These are set in the course of the feature extraction process and cannot be changed by the user.
        """
        if param is None:
            enf_param = Param()
        else:
            enf_param = param.copy()
        return enf_param
    
    def get_padding(self):
        """
        Get the padding that shall be applied around the image before feature extraction.
        Important: For some FEs this might have to be calculated based on certain parameters (e.g. layers).
        """
        return self.padding
    
    def get_patch_size(self):
        """
        Get the patch size that shall be used for feature extraction (images will be padded to multiples of patch-size).
        """
        return self.patch_size

    def get_features_scaled(self, image, param, **kwargs):
        """
        Given a filter model and an image, extract features at different scales.

        Parameters
        ----------
        image : 2d array
            image to segment
        param: Param
            object containing the parameters for the feature extraction

        Returns
        -------
        features : [nb_features x width x height]
            return extracted features

        """

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if param.image_downsample > 1:
            # image = image[:, ::param.image_downsample, ::param.image_downsample]
            image = scale_img(image, param.image_downsample)

        features_all_scales = []
        for s in param.fe_scalings:
            # image_scaled = image[:, ::s, ::s]
            image_scaled = scale_img(image, s)
            features = self.get_features(image_scaled, order=param.fe_order, **kwargs) #features have shape [nb_features, width, height]
            nb_features = features.shape[0]
            features = skimage.transform.resize(
                                image=features,
                                output_shape=(nb_features, image.shape[-2], image.shape[-1]),
                                preserve_range=True,
                                order=param.fe_order)
            
            features_all_scales.append(features)
        features_all_scales = np.concatenate(features_all_scales, axis=0)

        return features_all_scales
