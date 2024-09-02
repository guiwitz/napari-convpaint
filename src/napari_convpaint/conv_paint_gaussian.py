import skimage
import numpy as np
AVAILABLE_MODELS = ['gaussian_features']

class GaussianFeatures:
    def __init__(self, model_name='gaussian_features', use_cuda=False, sigma=3, **kwargs):
        
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.sigma = 3


    def get_features(self, image, annotations, **kwargs):
        """Given an image and a set of annotations, extract multiscale features

        Parameters
        ----------
        image : np.ndarray
            Image to extract features from, either HxW or CxHxW
        annotations : np.ndarray
            Annotations for training, HxW

        Returns
        -------
        extracted_features : np.ndarray
            Extracted features. Dimensions npixels x [nfeatures * nbscales]
        """
        
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        im_filter = skimage.filters.gaussian(image, sigma=self.sigma, channel_axis=0)
        mask = annotations > 0
        features = np.hstack([image[:, mask].T, im_filter[:, mask].T])
        
        return features


    def predict_image(self, image, classifier, **kwargs):
        """Given an image, predict the annotations

        Parameters
        ----------
        image : np.ndarray
            Image to predict annotations from (currently 2D only or 2D multichannel (C,H,W))
        classifier : randomforestclassifier
            Classifier to use for prediction

        Returns
        -------
        predicted_annotations : np.ndarray
            Predicted annotations. Dimensions (1,2)
        """
        
        image_XY_dims = image.shape[-2::]
        annotations = np.ones(image_XY_dims, dtype=np.uint8)
        features = self.get_features(image, annotations)
        predictions = classifier.predict(features)
        predictions = np.reshape(predictions, image_XY_dims)

        return predictions
    
    def get_padding(self):
        return self.sigma