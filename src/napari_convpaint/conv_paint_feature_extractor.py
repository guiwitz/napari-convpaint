import numpy as np
import pandas as pd
import skimage
import torch
from .conv_paint_param import Param
from .conv_paint_utils import scale_img, rescale_features, reduce_to_patch_multiple

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
        self.num_input_channels = [1]

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

    def get_features(self, image): # NOTE: can be deleted once new F extraction is implemented
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
            
        # FE params
        def_param.fe_name= self.model_name
        def_param.fe_use_cuda = False
        def_param.fe_scalings = [1]
        def_param.fe_order = 0
        def_param.fe_use_min_features = False

        # NOTE: Non-FE params should be set as default with caution

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

    def get_num_input_channels(self):
        """
        Get the number of input channels that the model expects.
        """
        return self.num_input_channels

    def gives_patched_features(self):
        return self.get_patch_size() > 1


### FEATURE EXTRACTION METHODS

    def get_feature_pyramid(self, data, param, patched=True):

        features_all_scales = []

        # Iterate over the scales and extract features for each scale
        for s in param.fe_scalings:

            # Downscale the image
            image_scaled = scale_img(data, s)

            # Make sure the downscaled part is a multiple of the patch size
            patch_size = self.get_patch_size()
            image_scaled = reduce_to_patch_multiple(image_scaled, patch_size)

            # Extract features as list for different channel_series and layers
            # Each element is [nb_features, z, w, h]
            features = self.get_features_from_channels(image_scaled)

            # Resize the features from this downscaling to the size of the (possibly patched) input
            # NOTE: this shouldn't do anything for scaling of 1, unless we want to "unpatch"
            if patched:
                target_shape = (data.shape[0],
                                data.shape[1],
                                int(data.shape[2]/self.get_patch_size()),
                                int(data.shape[3]/self.get_patch_size()))
            else:
                target_shape = data.shape

            features = [rescale_features(
                                feature_img=f,
                                target_shape=target_shape,
                                order=param.fe_order)
                        for f in features]

            # If torch tensor is returned, convert to numpy array
            if isinstance(features[0], torch.Tensor):
                # Detach, move to cpu, make np array
                features = [feature.detach().cpu().numpy() for feature in features]
            
            # Put together features for each input_channels procession and layers (if applicable)
            features = np.concatenate(features, axis=0)

            # If use_min_features is True, shorten features
            if param.fe_use_min_features:
                max_features = np.min(self.features_per_layer)
                features = features[:max_features] # NOTE: This would probably be better with PCA

            # Add the features to the list of features
            features_all_scales.append(features)
        
        # Concatenate the features from all scales
        features_all_scales = np.concatenate(features_all_scales, axis=0)

        return features_all_scales

    def get_features_from_channels(self, image):

        # Get the number of channels the model expects (e.g., RGB = 3)
        input_channels = self.get_num_input_channels()

        # If the image has one the number of channels the model expects, use it directly
        if image.shape[0] in input_channels:
            return self.extract_features_new(image)

        # For each channel, create a replicate with the needed number of input channels
        input_channels = min(input_channels)
        channel_series = [np.tile(ch, (input_channels, 1, 1, 1)) for ch in image]
        
        # Get outputs for each channel_series
        outputs = []
        for channel in channel_series:
            # output is a list of features, possibly from different layers (and thus with different sizes)
            output = self.extract_features_new(channel)
            # Make one list of all outputs (aligning different channel_series and layers)
            if isinstance(output, list):
                # If the output is a list of features, add the elements to the list
                outputs += output
            else:
                # If the output is a single feature, add it to the list
                outputs.append(output)

        return outputs
    
    def extract_features_new(self, image): # NOTE: can be renamed to get_features later
        """
        Gets the features of an image.

        Parameters:
        ----------
        image : np.ndarray [C, Z, Y, X]
            The input image. Dimensions are [channels, z, y, x].

        Returns:
        features : np.ndarray or torch.Tensor [F, Z, Y, X]
            The extracted features of the image. [nb_features, z, y, x]
        """
        # Extract features
        all_features = []
        for z in range(image.shape[1]):
            # Given that we get single-channel images as input:
            features = self.extract_features_from_plane(image[:,z])
            all_features.append(features)

        # Create a 4D array with the features first
        all_features = np.stack(all_features, axis=0)
        all_features = np.moveaxis(all_features, 1, 0)

        return [all_features]

    def extract_features_from_plane(self, image):
        """
        Given a single plane of the image, extract features.
        This method should be overridden by subclasses to implement the specific feature extraction logic.
        Important: Output needs to be 3D: [nb_features, h, w]

        Parameters:
        ----------
        image : np.ndarray [C, H, W]
            The input image. Dimensions are [channels, h, w].

        Returns:
        features : np.ndarray [F, H, W]
            The extracted features of the image. [nb_features, h, w]
        """
        raise NotImplementedError("Subclasses must implement get_features method.")