import numpy as np
import torch
import warnings
from .param import Param
from .utils import scale_img, rescale_features, reduce_to_patch_multiple, pad_to_shape, get_device_from_torch_model, concat_features

class FeatureExtractor:
    def __init__(self, model_name="vgg16", model=None, **kwargs):
        """
        Initializing a feature extractor. This is a superclass for all feature extractors.

        Parameters:
        ----------
        model_name : str
            The name of the model to use. If model is not None, this parameter is ignored.
        model : object
            The model to use. If not None, this model is used instead of creating a new model.
        """
        # Make sure a modelname is provided
        if model_name is None:
            raise ValueError('Please provide a model_name.')

        self.model_name = model_name

        # Define specifications for the feature extractor model
        self.padding = 0
        self.patch_size = 1
        self.has_global_context = False # Whether the FE contains operators that inject global (whole-input) context into per-pixel features 
        # e.g. AdaptiveAvgPool2d inside SE blocks (EfficientNet), attention across all patches (ViT-like), etc
        # For such FEs, tile_annotations / tile_image cannot match whole-image features at any finite padding
        self.has_3d_context = False # Whether the features of a plane depend on neighbouring planes (true 3D FE); such features are reused per stack, not per plane
        self.tile_block_size = None # If not None, this block size is used for tiling the image at segmentation
        self.num_input_channels = [1]
        self.features_per_layer = None # Set by Hookmodel; if None, fe_use_min_features warns and uses all features
        self.norm_mode = "default"  # or "imagenet" or "percentile"
        self.rgb_input = False # Whether the model takes RGB input or not
        self.proposed_scalings = [[1],
                                  [1,2],
                                  [1,2,4],
                                  [1,2,4,8]] # The scalings that are proposed to the user for this FE. Can be adjusted by the user.
        self.device = torch.device("cpu")
        self._warned_gpu_unavailable = False

        
        # USE PROVIDED MODEL IF AVAILABLE
        if model is not None:
            self.model = model
        # ELSE CREATE MODEL
        else:
            self.model = self.create_model(model_name)

    @staticmethod
    def create_model(model_name):
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
        return f"No specific description added for this Feature Extractor ({self.model_name}). Note that subclasses should override the get_description method."

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
        def_param.fe_scalings = [1]
        def_param.fe_order = 0
        def_param.fe_use_min_features = False

        # NOTE: Non-FE params should be set as default with caution
        def_param.tile_annotations = not self.get_has_global_context()

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
    
    def get_proposed_scalings(self):
        """
        Get the proposed scalings for this feature extractor. These are the scalings that are proposed to the user
        when this feature extractor is chosen. The user can still adjust the scalings afterwards.

        Returns:
        ----------
        proposed_scalings : list of list of int
            The proposed scalings for this feature extractor. Each element is a list of scalings for one option.
            Ex.: [[1], [1,2], [1,2,4], [1,2,4,8]]
        """
        # In case we have only one proposed scaling, we put it in a list to be consistent with the expected format
        if isinstance(self.proposed_scalings[0], int):
            return [self.proposed_scalings]
        return self.proposed_scalings
    
    def get_tile_block_size(self):
        """
        Get the tile block size that the feature extractor gives, if applicable. This is important for later tiling of the image at segmentation.

        Returns:
        ----------
        tile_block_size : int or None
            The tile block size that the feature extractor gives. If None, the default tile block size is used.
        """
        return self.tile_block_size
    
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
        Important: For some FEs this might have to be calculated based on certain parameters
        (e.g. layers and the according stride given e.g. max_pool).

        Returns:
        ----------
        patch_size : int
            The patch size that shall be used for feature extraction.
        """
        return self.patch_size

    def get_has_global_context(self):
        """
        True if the FE contains operators that inject global (whole-input)
        context into per-pixel features — e.g. AdaptiveAvgPool2d inside SE
        blocks (EfficientNet), attention across all patches (ViT-like), etc.
        For such FEs, tile_annotations / tile_image cannot match whole-image
        features at any finite padding because each output pixel depends on
        the entire input. Default False.
        """
        return self.has_global_context

    def get_has_3d_context(self):
        """
        True if the features of a plane depend on neighbouring planes (a true 3D FE,
        e.g. 3x3x3 kernels). Features of such FEs are reused (cache/store) per stack
        as passed in, never per plane. Default False.
        """
        return self.has_3d_context

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

    def move_model_to_device(self, device=torch.device("cpu")):
        """
        Move torch-based feature extractor model to the runtime device.

        For non-torch feature extractors (or models without a .to method), only stores
        the device on the extractor.
        """
        if isinstance(device, torch.device):
            device = device
        elif device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            warnings.warn(f"Resolving device from string '{device}' is deprecated. Please provide a torch.device object instead.")
            # Legacy fallback for direct FE usage outside ConvpaintModel.
            if device == "cpu":
                device = torch.device("cpu")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            raise ValueError(f"Invalid device: {device}. Please provide a torch.device object or a string ('cpu', 'cuda', 'mps').")

        if self.model is None or not hasattr(self.model, "to"):
            self.device = device
            return device

        current_device = get_device_from_torch_model(self.model)
        if current_device != device:
            self.model = self.model.to(device)
            if hasattr(self.model, "eval"):
                self.model.eval()

        self.device = device
        return device
    
    def supported_devices(self):
        """
        Get the list of devices that the feature extractor supports. This is by default determined by whether the model is a torch model or not.

        Returns:
        ----------
        supported_devices : list of torch.device
            The list of devices that the feature extractor supports.
        """
        if self.model is not None and hasattr(self.model, "to"):
            return [torch.device("cuda"), torch.device("mps"), torch.device("cpu")]
        else:
            return [torch.device("cpu")]



### FEATURE EXTRACTION METHODS

    def extract_features(self, data, param, device=torch.device("cpu")):
        """
        Extracts the features of an image (stack) with an arbitrary number of channels.
        This is the main method to call for feature extraction, which will handle scaling and rescaling of the features as needed.
        It calls the extract_features_pyramid method, which handles the actual feature extraction for each scale.

        Parameters:
        ----------
        data : np.ndarray [C, Z, H, W]
            The input image. Dimensions are [C, Z, H, W].
        param : Param
            The parameters for the feature extraction.
        device : torch.device, optional
            The device on which to perform feature extraction.

        Returns:
        ----------
        features : np.ndarray [F, Z, H, W]
            The extracted features of the image as a single array with [nb_features, Z, H, W]
        """
        # Move model to device if needed; for now, we this along the downstream process
        # self.move_model_to_device(device)

        # Extract features with scaling and rescaling as needed
        features = self.extract_features_pyramid(data=data, param=param, patched=self.gives_patched_features(), device=device)

        return features

    def extract_features_pyramid(self, data, param, patched=True, device=torch.device("cpu")):
        """
        Extracts the feature pyramid of an image (stack) with an arbitrary number of channels.
        Assumes that the image is a 4D array with dimensions [C, Z, H, W].
        Scales and prepares the image for feature extraction as required by the model/parameters.
        Then extracts the features from the image, and rescales them to image shape.
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
        rgb_data : bool
            Whether the input data is RGB or not. If True, the input data must also have 3 channels.
            Relevant only if the model can take RGB input, to distinguish between 3-channel non-RGB and RGB input.
        device : torch.device, optional
            The device on which to perform feature extraction.

        Returns:
        ----------
        features_all_scales : np.ndarray [F, Z, H, W]
            The extracted features of the image as a single array with [nb_features, Z, H, W]
        """

        native = self.extract_native(data, param, device)
        return self.reconstruct_from_native(native, data.shape, param, patched)

    def extract_native(self, data, param, device=torch.device("cpu")):
        """Expensive half of the pyramid: extract the *native* (pre-rescale) features per level.
        Returns one level per scaling: (features, scaled_shape, cropped_shape), where the shapes
        are those of the scaled input before and after cropping it to a multiple of the patch size.
        The features stay in the form the extractor produced them (for NN FEs, torch tensors
        on the extraction device), so that reconstruct_from_native can rescale on-device.
        They are also what is reused (independent of the requested output resolution;
        for patched FEs the small patch-grid features, not the full-resolution stack)."""
        # Check if the given selection of scalings is in the proposed scalings, and if not, give a warning
        if not param.fe_scalings in self.get_proposed_scalings():
            warnings.warn(f"The selected scalings {param.fe_scalings} are not in the proposed scalings {self.proposed_scalings}. Please check if this is intentional.")

        # Iterate over the scalings and extract the features for each (one level per scaling)
        native = []
        for s in param.fe_scalings:

            # Downscale the image
            image_scaled = scale_img(data, s)

            # Make sure the downscaled part is a multiple of the patch size
            patch_size = self.get_patch_size()
            scaled_shape = image_scaled.shape
            # NOTE: reduce_to_patch_multiple should not do anything if the inputs are already multiples of the patch size at all scales
            image_scaled = reduce_to_patch_multiple(image_scaled, patch_size)
            cropped_shape = image_scaled.shape

            # Extract features as list for different channel_series (and layers if applicable)
            # Each element is [nb_features, z, w, h]
            rgb_data = param.channel_mode == 'rgb'
            features = self.extract_features_from_multichannel_stack(image_scaled, rgb_data=rgb_data, device=device)
            # In case the features are not a list, but a single array, make it a list
            if not isinstance(features, list):
                features = [features]

            native.append((features, scaled_shape, cropped_shape))

        return native

    def reconstruct_from_native(self, native, data_shape, param, patched=True):
        """Cheap half of the pyramid: rescale the native features of each level (from extract_native)
        to the requested resolution and concatenate across channel-series and scales.
        data_shape is the original (pre-scaling) input shape [C, Z, H, W]."""
        patch_size = self.get_patch_size()
        features_all_scales = []
        for features, scaled_shape, cropped_shape in native:
            # Resize the features from this downscaling to the size of the (possibly patched) input
            # NOTE: this shouldn't do anything for scaling of 1, unless we want to "unpatch"
            if patched:
                target_shape = (data_shape[0],
                                data_shape[1],
                                data_shape[2]//patch_size,
                                data_shape[3]//patch_size)
            else:
                # When not patched, but patch_size > 1, we handle possible cropping due to reduce_to_patch_multiple
                # NOTE: this should not be necessary if the inputs are already multiples of the patch size at all scales
                if patch_size > 1 and cropped_shape[2:] != scaled_shape[2:] :
                    # Step 1: rescale to the reduced (cropped to patch multiple) shape
                    features = [rescale_features(
                                    feature_img=f,
                                    target_shape=cropped_shape,
                                    order=param.fe_order)
                                for f in features]

                    # Step 2: pad back to the (still downscaled) shape before the cropping
                    features = [pad_to_shape(f, scaled_shape[2:] ) for f in features]

                # Rescale to the full original shape
                target_shape = data_shape

            features = [rescale_features(
                                feature_img=f,
                                target_shape=target_shape,
                                order=param.fe_order)
                        for f in features]

            # Put together features for each input_channels procession (and layers if applicable)
            # (staying on the extraction device for torch tensors; the single host transfer happens at the very end)
            features = concat_features(features)

            # If use_min_features is True, shorten features
            if param.fe_use_min_features:
                if self.features_per_layer is None:
                    warnings.warn("Cannot use min_features because features_per_layer is not defined. Using all features instead.")
                else:
                    max_features = np.min(self.features_per_layer)
                    features = features[:max_features] # NOTE: This would probably be better with PCA

            # Add the features to the list of features
            features_all_scales.append(features)
        
        # Concatenate the features from all scales along the first axis (then move to CPU numpy in a single host transfer)
        features_all_scales = concat_features(features_all_scales)
        if isinstance(features_all_scales, torch.Tensor):
            features_all_scales = features_all_scales.detach().cpu().numpy()

        return features_all_scales

    def extract_features_from_multichannel_stack(self, image, rgb_data=False, device=torch.device("cpu")):
        """
        Extracts the features of an image (stack) with an arbitrary number of channels.
        Assumes that the image is a 4D array with dimensions [C, Z, H, W],
        and compatible spatial dimensions (e.g. multiple of patch size).
        If the number of channels is not compatible with the model,
        each channel is replicated to the number of input channels the model expects,
        and features are extracted for each channel separately.

        Parameters:
        ----------
        image : np.ndarray [C, Z, H, W]
            The input image. Dimensions are [C, Z, H, W].
        rgb_data : bool
            Whether the input data is RGB or not. If True, the input data must also have 3 channels.
            Relevant only if the model can take RGB input, to distinguish between 3-channel non-RGB and RGB input.
        device : torch.device, optional
            The device on which to perform feature extraction.

        Returns:
        ----------
        features : list of np.ndarrays or torch.Tensors [F, Z, H, W]
            The extracted features of the image. Each is [nb_features, Z, H, W]
        """

        # Get the number of channels the model expects
        fe_input_channels = self.get_num_input_channels()
        fe_rgb_input = self.rgb_input

        # Check if the FE model expects RGB input, but does not support 3 channels
        if fe_rgb_input and 3 not in fe_input_channels:
            raise ValueError(f"The feature extractor model is defined as taking RGB input, but does not take 3 channels (it takes {fe_input_channels}).")

        # Check if the input data is supposed to be RGB, but does not have 3 channels
        img_channels = image.shape[0]
        if rgb_data and img_channels != 3:
            raise ValueError(f"The input data is defined as RGB but has {img_channels} instead of 3 channels.")

        # If the image has a number of channels that the model expects, use it directly
        # Exception: If the model takes RGB input but the image is not RGB, repeat channels even if there are 3
        non_rgb_triple_with_rgb_fe = not rgb_data and img_channels == 3 and fe_rgb_input
        if img_channels in fe_input_channels and not non_rgb_triple_with_rgb_fe:
            # return [self.extract_features_from_stack(image)]
            return self.extract_features_from_stack(image, device=device)

        # For each channel, create a replicate with the needed number of input channels
        fe_input_channels = min(fe_input_channels)
        channel_series = [np.tile(ch, (fe_input_channels, 1, 1, 1)) for ch in image]
        
        # Get outputs for each channel_series
        all_outputs = []
        for channel in channel_series:
            # Output is either a single array or a list of features,
            # possibly from different layers (and thus with different sizes)
            output = self.extract_features_from_stack(channel, device=device)
            # Make one list of all outputs (aligning different channel_series and layers)
            if isinstance(output, list):
                # If the output is a list of features, add the elements to the list
                all_outputs += output
            else:
                # If the output is a single feature, add it to the list
                all_outputs.append(output)

        return all_outputs

    def extract_features_from_stack(self, image, device=torch.device("cpu")):
        """
        Extracts the features of an image given as a stack of planes.
        Assumes that the image is a 4D array with dimensions [C, Z, H, W],
        with C being the number of channels, complying with the model's input channels,
        and compatible spatial dimensions (e.g. multiple of patch size).

        Parameters:
        ----------
        image : np.ndarray [C, Z, H, W]
            The input image. Dimensions are [C, Z, H, W].
        device : torch.device, optional
            The device on which to perform feature extraction.

        Returns:
        ----------
        features : list of np.ndarrays or torch.Tensors [F, Z, H, W]
            The extracted features of the image. Each is [nb_features, Z, H, W]
        """
        # Extract features
        all_features = []
        # Go through the stack, and get features for each plane
        for z in range(image.shape[1]):
            features = self.extract_features_from_plane(image[:,z], device=device)
            all_features.append(features)

        # If the features are a list of arrays, stack them separately along z-axis
        if isinstance(all_features[0], list):
            all_features = [np.stack([f[i] for f in all_features], axis=1)
                            for i in range(len(all_features[0]))]
        # Create a 4D array with the features first, then z, h, w
        else:
            all_features = [np.stack(all_features, axis=1)]

        return all_features

    def extract_features_from_plane(self, image, device=torch.device("cpu")):
        """
        Extracts the features of a single plane of the image with [C, H, W].
        Assumes that the image is a 3D array with dimensions [C, H, W],
        with C being the number of channels, complying with the model's input channels,
        and compatible spatial dimensions (e.g. multiple of patch size).
        This method should be overridden by subclasses to implement the specific feature extraction logic.
        Important: Output needs to be (a list of) 3D: [nb_features, H, W]

        Parameters:
        ----------
        image : np.ndarray [C, H, W]
            The input image. Dimensions are [C, H, W].
        device : torch.device, optional
            The device on which to perform feature extraction.

        Returns:
        ----------
        features : np.ndarray [F, H, W]
            The extracted features of the image. [nb_features, H, W]
        """
        raise NotImplementedError("Subclasses must implement extract_features_from_plane method (or any method upstream of it).")


### FEATURE REUSE PROTOCOL (cache/store; see feature_cache.py and feature_store.py)
# NOTE: Nothing here needs to be implemented by a new FE (subclass): the defaults cover any FE
# that implements one of the hierarchical extraction methods above. Only an FE that overrides
# extract_features_pyramid itself is automatically excluded from feature reuse.

    def supports_feature_reuse(self, param):
        """Whether the features of this FE can be reused (cached/stored). The default excludes
        FEs that override extract_features_pyramid, since reuse relies on its native/reconstruct
        split; FEs that are cheap to recompute can also return False."""
        return type(self).extract_features_pyramid is FeatureExtractor.extract_features_pyramid

    @staticmethod
    def native_to_numpy(native):
        """Cast the native features (for NN FEs on-device torch tensors) to CPU numpy arrays,
        the device-independent form that is kept in the cache and the store: one level per scaling
        (as returned by extract_native), each with its shapes. `was_torch` records the native
        form, so native_from_numpy can lift them back and reconstruct with the same
        rescale backend as a fresh extraction (reused results must be identical to fresh ones)."""
        was_torch = any(isinstance(f, torch.Tensor)
                        for features, _, _ in native for f in features)
        levels = [([f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else f
                    for f in features], scaled_shape, cropped_shape)
                  for features, scaled_shape, cropped_shape in native]
        return {"levels": levels, "was_torch": was_torch}

    @staticmethod
    def native_from_numpy(entry, device=None):
        """Lift the native features of every level from their numpy form (see native_to_numpy)
        back onto `device` if the FE produced them as torch tensors. Pass the same resolved
        device the fresh extraction would use; device=None reconstructs on the CPU, which
        matches a CPU extraction but not bit-exactly a GPU one (interpolation kernels differ)."""
        native = entry["levels"]
        if entry.get("was_torch"):
            lift_device = device if device is not None else "cpu"
            native = [([torch.from_numpy(f).to(lift_device)
                        for f in features], scaled_shape, cropped_shape)
                      for features, scaled_shape, cropped_shape in native]
        return native

    @staticmethod
    def split_native_planes(native):
        """Split the native features of a stack (numpy form, one level per scaling) into one per plane (along Z)."""
        num_planes = native["levels"][0][0][0].shape[1]
        planes = []
        for z in range(num_planes):
            # Create a separate native with Z=1 for each plane
            levels = [([np.ascontiguousarray(f[:, z:z+1]) for f in features],
                       (scaled_shape[0], 1) + tuple(scaled_shape[2:]),
                       (cropped_shape[0], 1) + tuple(cropped_shape[2:]))
                      for features, scaled_shape, cropped_shape in native["levels"]]
            planes.append({"levels": levels, "was_torch": native["was_torch"]})
        return planes

    @staticmethod
    def join_native_planes(planes):
        """Join the per-plane native features (see split_native_planes) into those of the stack."""
        num_planes = len(planes)
        levels = []
        # Build the levels one by one; the planes are joined inside (along Z)
        for i, (features, scaled_shape, cropped_shape) in enumerate(planes[0]["levels"]):
            joined = [np.concatenate([p["levels"][i][0][j] for p in planes], axis=1) # Join planes
                      for j in range(len(features))] # ... for each array of the level
            levels.append((joined, # Level tuple: joined arrays, shapes with Z back to num_planes
                           (scaled_shape[0], num_planes) + tuple(scaled_shape[2:]),
                           (cropped_shape[0], num_planes) + tuple(cropped_shape[2:])))
        return {"levels": levels, "was_torch": planes[0]["was_torch"]}
