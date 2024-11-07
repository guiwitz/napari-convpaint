import numpy as np
import pandas as pd
import skimage
from .conv_parameters import Param

class FeatureExtractor:
    def __init__(self, model_name,**kwargs):
        self.model_name = model_name

    def get_features(self, image):
        """
        Gets the features of an image.

        Parameters:
        - image: The input image. Dimensions are [nb_channels, width, height]

        Returns:
        - features: The extracted features of the image. [nb_features, width, height]
        """
        raise NotImplementedError("Subclasses must implement get_features method.")

    def get_default_params(self):
        param = Param()
        # FE params (encouraged to set)
        param.model_name: str = self.model_name
        param.model_layers: list[str] = None
        param.scalings: list[int] = [1]
        param.order: int = 0
        param.use_min_features: bool = False
        param.use_cuda: bool = False
        param.padding : int = 0
        # Classifier
        param.classifier: str = None
        # General settings (NOTE: only set if shall be enforced by FE !)
        param.multi_channel_img: bool = None # use multichannel if image dimensions allow
        param.normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image
        param.image_downsample: int = None
        param.tile_annotations: bool = None
        param.tile_image: bool = None

        return param


    def get_features_scaled(self, image, param, **kwargs):
        """
        Given a filter model and an image, extract features at different scales.

        Parameters
        ----------
        image: 2d array
            image to segment
        scalings: list of ints
            downsampling factors
        order: int
            interpolation order for low scale resizing
        image_downsample: int, optional
            downsample image by this factor before extracting features, by default 1

        Returns
        -------
        features: [nb_features x width x height]
            return extracted features

        """

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if param.image_downsample > 1:
            image = image[:, ::param.image_downsample, ::param.image_downsample]

        padding = param.padding
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')

        features_all_scales = []
        for s in param.scalings:
            image_scaled = image[:, ::s, ::s]
            features = self.get_features(image_scaled, order=param.order, **kwargs) #features have shape [nb_features, width, height]
            nb_features = features.shape[0]
            features = skimage.transform.resize(
                                image=features,
                                output_shape=(nb_features, image.shape[-2], image.shape[-1]),
                                preserve_range=True,
                                order=param.order)
            
            features_all_scales.append(features)
        features_all_scales = np.concatenate(features_all_scales, axis=0)
        if padding > 0:
            features_all_scales = features_all_scales[:, padding:-padding, padding:-padding]
        return features_all_scales
    
    def predict_image(self, image, classifier, param, **kwargs):
        features = self.get_features_scaled(image=image,param=param)
        nb_features = features.shape[0] #[nb_features, width, height]

        #move features to last dimension
        features = np.moveaxis(features, 0, -1)
        features = np.reshape(features, (-1, nb_features))

        rows = np.ceil(image.shape[-2] / param.image_downsample).astype(int)
        cols = np.ceil(image.shape[-1] / param.image_downsample).astype(int)

        all_pixels = pd.DataFrame(features)
        predictions = classifier.predict(all_pixels)

        predicted_image = np.reshape(predictions, [rows, cols])
        if param.image_downsample > 1:
            predicted_image = skimage.transform.resize(
                image=predicted_image,
                output_shape=(image.shape[-2], image.shape[-1]),
                preserve_range=True, order=param.order).astype(np.uint8)
        return predicted_image
