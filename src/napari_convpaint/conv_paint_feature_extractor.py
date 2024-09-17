import numpy as np
import pandas as pd
import skimage

class FeatureExtractor:
    def __init__(self):
        padding = 0

    def get_features(self, image):
        """
        Gets the features of an image.

        Parameters:
        - image: The input image. Dimensions are [nb_channels, width, height]

        Returns:
        - features: The extracted features of the image. [nb_features, width, height]
        """
        raise NotImplementedError("Subclasses must implement get_features method.")

    def get_padding(self):
        """
        Gets the padding required for the feature extraction.

        Returns:
        - padding: The padding required for the feature extraction.
        """
        raise NotImplementedError("Subclasses must implement get_padding method.")

    def get_features_scaled(self, image, scalings = [1], order=0, image_downsample=1, **kwargs):
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

        if image_downsample > 1:
            image = image[:, ::image_downsample, ::image_downsample]

        padding = self.get_padding()
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')

        features_all_scales = []
        for s in scalings:
            image_scaled = image[:, ::s, ::s]
            features = self.get_features(image_scaled, order=order, **kwargs) #features have shape [nb_features, width, height]
            nb_features = features.shape[0]
            features = skimage.transform.resize(
                                image=features,
                                output_shape=(nb_features, image.shape[-2], image.shape[-1]),
                                preserve_range=True,
                                order=order)
            
            features_all_scales.append(features)
        features_all_scales = np.concatenate(features_all_scales, axis=0)
        if padding > 0:
            features_all_scales = features_all_scales[:, padding:-padding, padding:-padding]
        return features_all_scales
    
    def predict_image(self, image, classifier, scalings = [1], order=0, image_downsample=1, **kwargs):
        features = self.get_features_scaled(image=image,
                                            scalings=scalings,
                                            order=order,
                                            image_downsample=image_downsample)
        nb_features = features.shape[0] #[nb_features, width, height]

        #move features to last dimension
        features = np.moveaxis(features, 0, -1)
        features = np.reshape(features, (-1, nb_features))

        rows = np.ceil(image.shape[-2] / image_downsample).astype(int)
        cols = np.ceil(image.shape[-1] / image_downsample).astype(int)

        all_pixels = pd.DataFrame(features)
        predictions = classifier.predict(all_pixels)

        predicted_image = np.reshape(predictions, [rows, cols])
        if image_downsample > 1:
            predicted_image = skimage.transform.resize(
                image=predicted_image,
                output_shape=(image.shape[-2], image.shape[-1]),
                preserve_range=True, order=order).astype(np.uint8)
        return predicted_image
