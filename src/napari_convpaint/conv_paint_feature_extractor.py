import numpy as np
import torch
from .conv_paint_param import Param
from .conv_paint_utils import scale_img, rescale_features, reduce_to_patch_multiple, pad_to_shape

class FeatureExtractor:
    def __init__(self, model_name="vgg16", model=None, use_gpu=None, **kwargs):
        """
        Initializing a feature extractor. This is a superclass for all feature extractors.

        Parameters:
        ----------
        model_name : str
            The name of the model to use. If model is not None, this parameter is ignored.
        model : object
            The model to use. If not None, this model is used instead of creating a new model.
        use_gpu : bool
            Whether to use GPU or not. If not provided, the default is False.
        """

        self.model_name = model_name
        self.use_gpu = use_gpu

        # Define specifications for the feature extractor model
        self.padding = 0
        self.patch_size = 1
        self.num_input_channels = [1]
        self.norm_imagenet = False

        # Make sure only one method of providing the model is used
        if model is not None and model_name is not None:
            raise ValueError('Please provide either a model or a model_name, not both.')
        
        # USE PROVIDED MODEL IF AVAILABLE
        if model is not None:
            self.model = model
        # ELSE CREATE MODEL
        else:
            self.model = self.create_model(model_name, use_gpu=use_gpu)

    @staticmethod
    def create_model(model_name, use_gpu=None):
        """
        This method is intended to be overridden by subclasses to load the specific model.

        Parameters:
        ----------
        model_name : str
            The name of the model to create (can be used e.g. in torch functions).
        
        Returns:
        ----------
        model : object
            The created FE model.
        """
        model = None
        return model
    
    def get_description(self):
        """
        Returns a description of the feature extractor as a string.

        Returns:
        ----------
        description : str
            A string describing the feature extractor.
        """
        return "This is a generic feature extractor. Subclasses should override this method."

    def get_default_params(self, param=None):
        """
        Get the default parameters that the FE shall set/enforce when chosen by the user.
        These can still be changed by the user afterwards.

        Parameters:
        ----------
        param : Param, optional
            The parameters to use as a basis. If not provided, a new Param object is created.
            Allows to use a parameter set and only adjust the parameters defined as defaults.

        Returns:
        ----------
        def_param : Param
            param object with the default parameters for the feature extractor.
        """
        if param is None:
            def_param = Param()
        else:
            def_param = param.copy()
            
        # FE params
        def_param.fe_name= self.model_name
        def_param.fe_use_gpu = False
        def_param.fe_scalings = [1]
        def_param.fe_order = 0
        def_param.fe_use_min_features = False

        # NOTE: Non-FE params should be set as default with caution

        return def_param
    
    def get_layer_keys(self):
        """
        If the model has layers, return the keys of the layers.

        Returns:
        ----------
        layer_keys : list of str
            The keys of the layers in the model.
        """
        return None
    
    def get_enforced_params(self, param=None):
        """
        Define which parameters need to be absolutley enforced for feature extraction.
        These are set in the course of the feature extraction process and cannot be changed by the user.

        Parameters:
        ----------
        param : Param, optional
            The parameters to use as a basis. If not provided, a new Param object is created.
            Allows to use a parameter set and only adjust the parameters defined as enforced.

        Returns:
        ----------
        enf_param : Param
            param object with the enforced parameters for the feature extractor.
        """
        if param is None:
            enf_param = Param()
        else:
            enf_param = param.copy()
        return enf_param
    
    def get_name(self):
        """
        Get the name of the feature extractor.

        Returns:
        ----------
        name : str
            The name of the feature extractor.
        """
        return self.model_name
    
    def get_padding(self):
        """
        Get the padding that shall be applied around the image before feature extraction.
        Important: For some FEs this might have to be calculated based on certain parameters (e.g. layers).

        Returns:
        ----------
        padding : int
            The padding that shall be applied around the image before feature extraction.
        """
        return self.padding

    def get_patch_size(self):
        """
        Get the patch size that shall be used for feature extraction
        (images will be padded to multiples of patch-size).
        Important: For some FEs this might have to be calculated based on certain parameters (e.g. layers).

        Returns:
        ----------
        patch_size : int
            The patch size that shall be used for feature extraction.
        """
        return self.patch_size

    def get_num_input_channels(self):
        """
        Get the number of input channels that the model expects.

        Returns:
        ----------
        num_input_channels : list of int
            The number of input channels that the model expects.
            A list of values for the different numbers of channels the model can handle.
            Ex.: [1, 3] for grayscale and RGB images.
        """
        return self.num_input_channels

    def gives_patched_features(self):
        """
        Define if the model gives patched features or not. This is important for later rescaling of
        the features and/or predictions.

        Returns:
        ----------
        patched : bool
            Whether the model gives patched features or not.
            If True, the features are in patch-size resolution.
            If False, the features are in image/pixel resolution.
        """
        patched = self.get_patch_size() > 1
        return patched


### FEATURE EXTRACTION METHODS

    def get_feature_pyramid(self, data, param, patched=True):
        """
        Gets the feature pyramid of an image with an arbitrary number of channels.
        Assumes that the image is a 4D array with dimensions [C, Z, H, W].
        Scales and prepares the image for feature extraction as required by the model/parameters.
        Then extracts the features from the image, and rescales them to an image shape.
        The features are concatenated along the first axis.

        Parameters:
        ----------
        data : np.ndarray [C, Z, H, W]
            The input image. Dimensions are [C, Z, H, W].
        param : Param
            The parameters for the feature extraction.
        patched : bool
            Whether the features shall be kept in patch-size resolution or not.
            If False, the features are resized to the size of the input image.
        
        Returns:
        ----------
        features_all_scales : np.ndarray [F, Z, H, W]
            The extracted features of the image as a single array with [nb_features, Z, H, W]
        """

        features_all_scales = []

        # Iterate over the scales and extract features for each scale
        for s in param.fe_scalings:

            # Downscale the image
            image_scaled = scale_img(data, s)

            # Make sure the downscaled part is a multiple of the patch size
            patch_size = self.get_patch_size()
            pre_reduction_shape  = image_scaled.shape
            image_scaled = reduce_to_patch_multiple(image_scaled, patch_size)
            reduced_shape = image_scaled.shape

            # Extract features as list for different channel_series (and layers if applicable)
            # Each element is [nb_features, z, w, h]
            features = self.get_features_from_channels(image_scaled)
            # In case the features are not a list, but a single array, make it a list
            if not isinstance(features, list):
                features = [features]

            # Resize the features from this downscaling to the size of the (possibly patched) input
            # NOTE: this shouldn't do anything for scaling of 1, unless we want to "unpatch"
            if patched:
                target_shape = (data.shape[0],
                                data.shape[1],
                                data.shape[2]//self.get_patch_size(),
                                data.shape[3]//self.get_patch_size())
            else:
                # When not patched, but patch_size > 1, we handle possible cropping due to reduce_to_patch_multiple
                # NOTE: this should not be necessary if the inputs are already multiples of the patch size at all scales
                if patch_size > 1 and reduced_shape[2:] != pre_reduction_shape[2:] :
                    # Step 1: rescale to the reduced (cropped to patch multiple) shape
                    features = [rescale_features(
                                    feature_img=f,
                                    target_shape=reduced_shape,
                                    order=param.fe_order)
                                for f in features]

                    # Step 2: pad back to original pre_reduction_shape (but still downscaled)
                    features = [pad_to_shape(f, pre_reduction_shape[2:] ) for f in features]

                # Rescale to the full original shape
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
            
            # Put together features for each input_channels procession (and layers if applicable)
            features = np.concatenate(features, axis=0)

            # If use_min_features is True, shorten features
            if param.fe_use_min_features:
                max_features = np.min(self.features_per_layer)
                features = features[:max_features] # NOTE: This would probably be better with PCA

            # Add the features to the list of features
            features_all_scales.append(features)
        
        # Concatenate the features from all scales along the first axis
        features_all_scales = np.concatenate(features_all_scales, axis=0)

        return features_all_scales

    def get_features_from_channels(self, image):
        """
        Gets the features of an image with an arbitrary number of channels.
        Assumes that the image is a 4D array with dimensions [C, Z, H, W],
        and compatible spatial dimensions (e.g. multiple of patch size)

        Parameters:
        ----------
        image : np.ndarray [C, Z, H, W]
            The input image. Dimensions are [C, Z, H, W].
        
        Returns:
        ----------
        features : list of np.ndarrays or torch.Tensors [F, Z, H, W]
            The extracted features of the image. Each is [nb_features, Z, H, W]
        """

        # Get the number of channels the model expects (e.g., RGB = 3)
        input_channels = self.get_num_input_channels()

        # If the image has one the number of channels the model expects, use it directly
        if image.shape[0] in input_channels:
            # return [self.get_features(image)]
            return self.get_features(image)

        # For each channel, create a replicate with the needed number of input channels
        input_channels = min(input_channels)
        channel_series = [np.tile(ch, (input_channels, 1, 1, 1)) for ch in image]
        
        # Get outputs for each channel_series
        all_outputs = []
        for channel in channel_series:
            # Output is either a single array or a list of features,
            # possibly from different layers (and thus with different sizes)
            output = self.get_features(channel)
            # Make one list of all outputs (aligning different channel_series and layers)
            if isinstance(output, list):
                # If the output is a list of features, add the elements to the list
                all_outputs += output
            else:
                # If the output is a single feature, add it to the list
                all_outputs.append(output)

        return all_outputs

    def get_features(self, image):
        """
        Gets the features of an image.
        Assumes that the image is a 4D array with dimensions [C, Z, H, W],
        with C being the number of channels, complying with the model's input channels,
        and compatible spatial dimensions (e.g. multiple of patch size).

        Parameters:
        ----------
        image : np.ndarray [C, Z, H, W]
            The input image. Dimensions are [C, Z, H, W].

        Returns:
        ----------
        features : list of np.ndarrays or torch.Tensors [F, Z, H, W]
            The extracted features of the image. Each is [nb_features, Z, H, W]
        """
        # Extract features
        all_features = []
        # Go through the stack, and get features for each plane
        for z in range(image.shape[1]):
            features = self.get_features_from_plane(image[:,z])
            all_features.append(features)

        # If the features are a list of arrays, stack them separately along z-axis
        if isinstance(all_features[0], list):
            all_features = [np.stack([f[i] for f in all_features], axis=1)
                            for i in range(len(all_features[0]))]
        # Create a 4D array with the features first, then z, h, w
        else:
            all_features = [np.stack(all_features, axis=1)]

        return all_features

    def get_features_from_plane(self, image):
        """
        Given a single plane of the image with [C, H, W], extract features.
        Assumes that the image is a 3D array with dimensions [C, H, W],
        with C being the number of channels, complying with the model's input channels,
        and compatible spatial dimensions (e.g. multiple of patch size).
        This method should be overridden by subclasses to implement the specific feature extraction logic.
        Important: Output needs to be (a list of) 3D: [nb_features, H, W]

        Parameters:
        ----------
        image : np.ndarray [C, H, W]
            The input image. Dimensions are [C, H, W].

        Returns:
        ----------
        features : np.ndarray [F, H, W]
            The extracted features of the image. [nb_features, H, W]
        """
        raise NotImplementedError("Subclasses must implement get_features method.")