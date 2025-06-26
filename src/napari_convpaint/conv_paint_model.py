
import pickle
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
import skimage
import torch
import pandas as pd

from napari_convpaint.conv_paint_nnlayers import AVAILABLE_MODELS as NN_MODELS
from napari_convpaint.conv_paint_gaussian import AVAILABLE_MODELS as GAUSSIAN_MODELS
from napari_convpaint.conv_paint_dino import AVAILABLE_MODELS as DINO_MODELS
from napari_convpaint.conv_paint_combo_fe import AVAILABLE_MODELS as COMBO_MODELS
from napari_convpaint.conv_paint_nnlayers import Hookmodel
from napari_convpaint.conv_paint_gaussian import GaussianFeatures
from napari_convpaint.conv_paint_dino import DinoFeatures
from napari_convpaint.conv_paint_combo_fe import ComboFeatures
from napari_convpaint.conv_paint_param import Param
from . import conv_paint_utils

class ConvpaintModel:
    """
    Base Convpaint model class that combines a feature extraction and a classifier.
    Consists of a feature extractor model, a classifier and a Param object,
    which defines the details of the model procedures.
    Model can be initialized with an alias, a model path, a param object, or a feature extractor name.
    If initialized by FE name, also other parameters can be given to override the defaults of the FE model.
    If neither an alias, a model path, a param object, nor a feature extractor name is given,
    a default Conpaint model is created (defined in the get_default_params() method).

    Parameters:
    ----------
    alias : str, optional
        Alias of a predefined model, by default None
    model_path : str, optional
        Path to a saved model, by default None
    param : Param, optional
        Param object with the model parameters, by default None
    fe_name : str, optional
        Name of the feature extractor model, by default None
    fe_use_cuda : bool, optional
        Whether to use CUDA for the feature extractor (if initialized by name), by default None
    fe_layers : list[str], optional
        List of layer names to extract features from (if initialized by name), by default None
    **kwargs : additional parameters
        Additional parameters to override defaults for the model or feature extractor
    """

    FE_MODELS_TYPES_DICT = {}

    # Define the standard models that are available to be loaded by alias
    std_models = {"vgg": Param(fe_name="vgg16"),
                  "vgg-m": Param(fe_name="vgg16",
                                 fe_layers=['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
                                            'features.2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
                                            'features.5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'],
                                 fe_scalings=[1, 2, 4]),
                  "vgg-l": Param(fe_name="vgg16",
                                 fe_layers=['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
                                            'features.2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
                                            'features.5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
                                            'features.7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
                                            'features.10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'],
                                 fe_scalings=[1, 2, 4, 8]),
                  "dino": Param(fe_name="dinov2_vits14_reg"),
                  "gaussian": Param(fe_name="gaussian_features"),
                  "cellpose": Param(fe_name="cellpose_backbone")
                  }

    def __init__(self, alias=None, model_path=None, param=None, fe_name=None, fe_use_cuda=None, fe_layers=None, **kwargs):
        """
        Initializes the Convpaint model by loading the specified model, parameters, or feature extractor.

        The constructor can be initialized in four ways:
        1. By providing an alias (e.g., "vgg-s", "dino", "gaussian", etc.), in which case a corresponding model
        configuration will be loaded.
        2. By providing a saved model path (model_path) to load a pre-trained model.
        This can be a .pkl file (holding the clf and FE models) or .yml file (defining the model parameters).
        3. By providing a Param object, which contains model parameters.
        4. By providing the name of the feature extractor, CUDA usage, and feature extraction layers, in which case
        the additional kwargs will be used to override the defaults of the feature extractor model.
        
        If none of the options are provided, a default Convpaint model will be created.

        Parameters:
        ----------
        alias : str, optional
            The alias of a predefined model, by default None.
        model_path : str, optional
            Path to a saved model file, by default None.
        param : Param, optional
            Param object containing model parameters, by default None.
        fe_name : str, optional
            Name of the feature extractor model, by default None.
        fe_use_cuda : bool, optional
            Whether to use CUDA for the feature extractor, by default None.
        fe_layers : list[str], optional
            List of layers to extract features from, by default None.
        **kwargs : additional parameters, used when initialzing the model by name
            Additional parameters to override defaults for the model or feature extractor.

        Raises:
        ------
        ValueError
            If more than one of alias, model_path, param, or fe_name is provided.
        """
        # Initialize parameters and features
        self._param = Param()
        self.reset_training()
        self.num_trainings = 0

        # Initialize the dictionary of all available models
        if not ConvpaintModel.FE_MODELS_TYPES_DICT:
            ConvpaintModel._init_fe_models_dict()

        if (alias is not None) + (model_path is not None) + (param is not None) + (fe_name is not None) > 1:
            raise ValueError('Please provide a model path, a param object, or a model name but not multiples.')

        # If an alias is given, create an according model
        if alias is not None:
            if alias in ConvpaintModel.std_models:
                param = ConvpaintModel.std_models[alias]
            else:
                raise ValueError(f'Alias "{alias}" not found in the standard models.')

        # Initialize the model
        if model_path is not None:
            self._load(model_path)
        elif param is not None:
            self._load_param(param)
        elif fe_name is not None:
            self._set_fe(fe_name, fe_use_cuda, fe_layers)
            self._param = self.get_fe_defaults()
            self.set_params(no_warning=True, # Here at initiation, it is intended to set FE parameters...
                            fe_layers = fe_layers, # Overwrite the layers with the given layers
                            fe_use_cuda = fe_use_cuda, # Overwrite the cuda usage with the given cuda usage
                            **kwargs) # Overwrite the parameters with the given parameters
        else:
            cpm_defaults = ConvpaintModel.get_default_params()
            self._load_param(cpm_defaults)

    @staticmethod
    def _init_fe_models_dict():
        """
        Initializes the dictionary of all available feature extractor models.
        """
        # Initialize the MODELS TO TYPES dictionary with the models that are always available
        ConvpaintModel.FE_MODELS_TYPES_DICT = {x: Hookmodel for x in NN_MODELS}
        ConvpaintModel.FE_MODELS_TYPES_DICT.update({x: GaussianFeatures for x in GAUSSIAN_MODELS})
        ConvpaintModel.FE_MODELS_TYPES_DICT.update({x: DinoFeatures for x in DINO_MODELS})
        ConvpaintModel.FE_MODELS_TYPES_DICT.update({x: ComboFeatures for x in COMBO_MODELS})

        # Try to import CellposeFeatures and update the MODELS TO TYPES  dictionary if successful
        # Cellpose is only installed with pip install napari-convpaint[cellpose]
        try:
            from napari_convpaint.conv_paint_cellpose import AVAILABLE_MODELS as CELLPOSE_MODELS
            from napari_convpaint.conv_paint_cellpose import CellposeFeatures
            ConvpaintModel.FE_MODELS_TYPES_DICT.update({x: CellposeFeatures for x in CELLPOSE_MODELS})
        except ImportError:
            # Handle the case where CellposeFeatures or its dependencies are not available
            print("Info: Cellpose is not installed and is not available as feature extractor.\n"
                "Run 'pip install napari-convpaint[cellpose]' to install it.")
        # Same for ilastik
        try:
            from napari_convpaint.conv_paint_ilastik import AVAILABLE_MODELS as ILASTIK_MODELS
            from napari_convpaint.conv_paint_ilastik import IlastikFeatures
            ConvpaintModel.FE_MODELS_TYPES_DICT.update({x: IlastikFeatures for x in ILASTIK_MODELS})
        except ImportError:
            # Handle the case where IlastikFeatures or its dependencies are not available
            print("Info: Ilastik is not installed and is not available as feature extractor.\n"
                "Run 'pip install napari-convpaint[ilastik]' to install it.\n"
                "Make sure to also have fastfilters installed ('conda install -c ilastik-forge fastfilters').")

    @staticmethod
    def get_fe_models_types():
        """
        Returns a dictionary of all available feature extractors; names as keys and types as values.

        Parameters:
        ----------
        None

        Returns:
        ----------
        models_dict : dict
            Dictionary of all available feature extractors; names as keys and types as values.
        """
        models_dict = ConvpaintModel.FE_MODELS_TYPES_DICT.copy()
        return models_dict
    
    @staticmethod
    def get_default_params():
        """
        Returns a param object, which defines the default Convpaint model.

        Parameters:
        ----------
        None

        Returns:
        ----------
        def_param : Param
            Param object with the default Convpaint model parameters.
        """

        def_param = Param()

        # Image processing parameters
        def_param.multi_channel_img = False # Interpret the first dimension as channels
        def_param.rgb_img = False # Used to signal to the model that the image is RGB
        def_param.normalize = 2  # 1: no normalization, 2: normalize stack, 3: normalize each image

        # Acceleration parameters
        def_param.image_downsample = 1
        def_param.seg_smoothening = 1
        def_param.tile_annotations = True
        def_param.tile_image = False
        def_param.use_dask = False

        # FE parameters
        def_param.fe_name = "vgg16"
        def_param.fe_layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))']
        # def_param.fe_name = "convnext"
        # def_param.fe_layers = [0,1]
        def_param.fe_use_cuda = False
        def_param.fe_scalings = [1, 2, 4]
        def_param.fe_order = 0
        def_param.fe_use_min_features = False

        # Classifier parameters
        def_param.classifier = None
        def_param.clf_iterations = 50
        def_param.clf_learning_rate = 0.1
        def_param.clf_depth = 5

        return def_param
    
    def get_param(self, key):
        """
        Returns the value of the given parameter key.

        Parameters:
        ----------
        key : str
            Key of the parameter to get
        
        Returns:
        ----------
        param_value : any
            Value of the parameter with the given key
        """
        param_value = self._param.get(key)
        return param_value
    
    def get_params(self):
        """
        Returns all parameters of the model (= a copy of the Param object).

        Parameters:
        ----------
        None

        Returns:
        ----------
        param : Param
            Copy of the Param object with all parameters of the model
        """
        param = self._param.copy()
        return param
    
    def set_param(self, key, val, no_warning=False):
        """
        Sets the value of the given parameter key.
        Raises a warning if key FE parameters are set (intended only for initiation),
        unless no_warning is True.

        Parameters:
        ----------
        key : str
            Key of the parameter to set
        val : any
            Value to set the parameter to
        no_warning : bool, optional
            Whether to suppress the warning for setting FE parameters, by default False
        """
        if key in Param.get_keys():
            self._param.set_single(key, val)
            if (key == 'fe_name' or key == 'fe_use_cuda' or key == 'fe_layers') and not no_warning:
                warnings.warn("Setting the parameters fe_name, fe_use_cuda, or fe_layers is not intended. " +
                    "You should create a new ConvpaintModel instead.")
        else:
            warnings.warn(f'Parameter "{key}" not found in the model parameters.')

    def set_params(self, param=None, no_warning=False, **kwargs):
        """
        Sets the parameters, given either as a Param object or as keyword arguments.
        Note that the model is not reset and no new FE model is created.
        If fe_name, fe_use_cuda, and fe_layers change, you should create a new ConvpaintModel.

        Parameters:
        ----------
        param : Param, optional
            Param object with the parameters to set, by default None
        no_warning : bool, optional
            Whether to suppress the warning for setting FE parameters, by default False
        **kwargs : parameters as keyword arguments
            Parameters to set as keyword arguments (instead of a Param object)
        
        Raises
        ----------
        ValueError
            If both a Param object and keyword arguments are provided.
        """
        if param is not None and kwargs:
            raise ValueError('Please provide either a Param object or keyword arguments, not both.')
        if param is not None:
            kwargs = param.__dict__
        for key, val in kwargs.items():
            if val is not None:
                self.set_param(key, val, no_warning=no_warning)

    def save(self, model_path, create_pkl=True, create_yml=True):
        """
        Saves the model to a pkl an/or yml file.
        The pkl file includes the classifier and the param object, as well as the
        features and targets (from continuous training).
        The yml file includes only the parameters of the model.

        Parameters:
        ----------
        model_path : str
            Path to save the model to.
        create_pkl : bool, optional
            Whether to create a pkl file, by default True
        create_yml : bool, optional
            Whether to create a yml file, by default True
        """
        if model_path[-4:] == ".pkl" or model_path[-4:] == ".yml":
            model_path = model_path[:-4]
        if create_pkl:
            pkl_path = model_path + ".pkl"
            if self.classifier is None:
                warnings.warn('No trained classifier found.')
            with open(pkl_path, 'wb') as f:
                data = {
                    'classifier': self.classifier,
                    'param': self._param,
                    'table': self.table,
                    'annnotations': self.annot_dict
                }
                pickle.dump(data, f)

        if create_yml:
            yml_path = model_path + ".yml"
            self._param.save(yml_path)

    def _load(self, model_path):
        """
        Loads the model from a file. Guesses the file type based on the file ending.
        Only intended for internal use at model initiation.
        """
        if model_path[-4:] == ".pkl":
            self._load_pkl(model_path)
        elif model_path[-4:] == ".yml":
            self._load_yml(model_path)
        else:
            raise ValueError('Model path must end with ".pkl" or ".yml".')

    def _load_pkl(self, pkl_path):
        """
        Loads the model from a pickle file.
        Only intended for internal use at model initiation.
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        new_param = data['param']
        self._set_fe(new_param.fe_name, new_param.fe_use_cuda, new_param.fe_layers)
        self._param = new_param.copy()
        self.classifier = data['classifier']
        self.table = data['table']
        self.annot_dict = data['annnotations']

    def _load_yml(self, yml_path):
        """
        Loads the model from a yml file.
        Only intended for internal use at model initiation.
        """
        new_param = Param.load(yml_path)
        self._set_fe(new_param.fe_name, new_param.fe_use_cuda, new_param.fe_layers)
        self._param = new_param

    def _load_param(self, param: Param):
        """
        Loads the given param object into the model and sets the model accordingly.
        Only intended for internal use at model initiation.
        """
        self._set_fe(param.fe_name, param.fe_use_cuda, param.fe_layers)
        self._param = self.get_fe_defaults()
        self.set_params(no_warning=True, **param.__dict__) # Overwrite the parameters with the given parameters


### FE METHODS

    def _set_fe(self, fe_name=None, fe_use_cuda=None, fe_layers=None):
        """
        Sets the model based on the given FE parameters.
        Creates new feature extracture, and resets the classifier.
        Only intended for internal use at model initiation.
        """

        # Reset the model and classifier
        self.reset_classifier()

        # Check if we need to create a new FE model
        fe_name_changed = fe_name is not None and fe_name != self._param.get("fe_name")
        fe_use_cuda_changed = fe_use_cuda is not None and fe_use_cuda != self._param.get("fe_use_cuda")
        fe_layers_changed = fe_layers is not None and fe_layers != self._param.get("fe_layers")

        # Create the feature extractor model
        if fe_name_changed or fe_use_cuda_changed or fe_layers_changed:
            self.fe_model = ConvpaintModel.create_fe(
                name=fe_name,
                use_cuda=fe_use_cuda,
                layers=fe_layers
            )
        # Set the parameters
        self._param.set(fe_name=fe_name, fe_use_cuda=fe_use_cuda, fe_layers=fe_layers)

    @staticmethod
    def create_fe(name, use_cuda=None, layers=None):
        """
        Creates a feature extractor model based on the given parameters.
        Distinguishes between different types of feature extractors such as Hookmodels
        and initializes them accordingly.

        Parameters:
        ----------
        name : str
            Name of the feature extractor model
        use_cuda : bool, optional
            Whether to use CUDA for the feature extractor
        layers : list[str], optional
            List of layer names to extract features from, by default None

        Returns:
        ----------
        fe_model : FeatureExtractor
            The created feature extractor model
        """
        
        # Check if name is valid and create the feature extractor object
        if not name in ConvpaintModel.FE_MODELS_TYPES_DICT:
            raise ValueError(f'Feature extractor model {name} not found.')
        fe_model_class = ConvpaintModel.FE_MODELS_TYPES_DICT.get(name)
        
        # Initialize the feature extractor model
        if fe_model_class is Hookmodel:
            fe_model = fe_model_class(
                model_name=name,
                use_cuda=use_cuda,
                layers=layers
        )
        else:
            fe_model = fe_model_class(
                model_name=name,
                use_cuda=use_cuda
            )

        # Check if the model was created successfully
        if fe_model is None:
            raise ValueError(f'Feature extractor model {name} could not be created.')

        return fe_model

    def get_fe_defaults(self):
        """
        Return the default params for the feature extractor.
        Where they are not sepecified, the default Convpaint params are used.

        Parameters:
        ----------
        None

        Returns:
        ---------
        new_param : Param
            Conpaint model defaults adjusted to the feature extractor defaults
        """
        cpm_defaults = ConvpaintModel.get_default_params() # Get ConvPaint defaults
        new_param = self.fe_model.get_default_params(cpm_defaults) # Overwrite defaults defined in the FE model
        return new_param

    def get_fe_layer_keys(self):
        """
        Returns the keys of the feature extractor layers (None if the model uses no layers).

        Parameters:
        ----------
        None

        Returns:
        ---------
        keys : list[str] or None
            List of keys of the feature extractor layers, or None if the model uses no layers
        """
        keys = self.fe_model.get_layer_keys()
        return keys


### USER METHODS FOR TRAINING AND PREDICTION
    
    def train(self, image, annotations, memory_mode=False, img_ids=None, use_rf=False, allow_writing_files=False, skip_norm=False):
        """
        Trains the Convpaint model's classifier given images and annotations.
        Uses the Parameter and the FeatureExtractor model to extract features.
        Then uses the features of annotated pixels to train the classifier.
        Trains the internal classifier, which is also returned.

        Parameters:
        ----------
        image : np.ndarray or list[np.ndarray]
            Image to train the classifier on or list of images
        annotations : np.ndarray or list[np.ndarray]
            Annotations to train the classifier on or list of annotations
            Image and annotation lists must correspond to each other
        memory_mode : bool, optional
            Whether to use memory mode, by default False
            If True, the annotations are registered and updated, and only features for new pixels are extracted
        img_ids : str or list[str], optional
            Image IDs to register the annotations with, by default None
        use_rf : bool, optional
            Whether to use a RandomForestClassifier instead of a CatBoostClassifier, by default False
        allow_writing_files : bool, optional
            Whether to allow writing files for the CatBoostClassifier, by default False
        skip_norm : bool, optional
            Whether to skip normalization of the images before training, by default False
            If True, the images are not normalized according to the parameter `normalize` in the model parameters

        Returns:
        ----------
        clf : CatBoostClassifier or RandomForestClassifier
            Trained classifier
        """
        clf = self._train(image, annotations, memory_mode=memory_mode, img_ids=img_ids, use_rf=use_rf,
                          allow_writing_files=allow_writing_files, skip_norm=skip_norm)
        return clf

    def segment(self, image, skip_norm=False):
        """
        Segments images by predicting its most probable classes using the trained classifier.
        Uses the feature extractor model to extract features from the image.
        Predicts the classes of the pixels in the image using the trained classifier.

        Parameters:
        ----------
        image : np.ndarray or list[np.ndarray]
            Image to segment or list of images
        skip_norm : bool, optional
            Whether to skip normalization of the images before segmentation, by default False
            If True, the images are not normalized according to the parameter `normalize` in the model parameters

        Returns:
        ----------
        seg : np.ndarray or list[np.ndarray]
            Segmented image or list of segmented images (according to the input)
            Dimensions are equal to the input image(s) without the channel dimension
        """
        _, seg = self._predict(image, add_seg=True, skip_norm=skip_norm)
        return seg

    def predict_probas(self, image, skip_norm=False):
        """
        Predicts the probabilities of the classes of the pixels in an image using the trained classifier.
        Uses the feature extractor model to extract features from the image.
        Estimates the probablity of each class based on the features and the trained classifier.

        Parameters:
        ----------
        image : np.ndarray or list[np.ndarray]
            Image to predict probabilities for or list of images
        skip_norm : bool, optional
            Whether to skip normalization of the images before prediction, by default False
            If True, the images are not normalized according to the parameter `normalize` in the model parameters

        Returns:
        ----------
        probas : np.ndarray or list[np.ndarray]
            Predicted probabilities of the classes of the pixels in the image or list of images
            Dimensions are equal to the input image(s) without the channel dimension,
            with the class dimension added first
        """
        probas = self._predict(image, add_seg=False, skip_norm=skip_norm)
        return probas


### FEATURE EXTRACTION
    
    def get_feature_image(self, data, annotations=None, memory_mode=False, restore_input_form=True, img_ids=None, skip_norm=False):
        """
        Returns the features of images extracted by the feature extractor model.
        If annotations are given, the features are extracted from the annotated planes,
        optionally tiled around the annotations.
        Downsamples and pads images and annotations (if given) as required by the feature extractor.
        If restore_input_form is True, the features are reshaped to the original image shape,
        without the channel dimension, but with the features dimension added first.

        Parameters:
        ----------
        data : np.ndarray or list[np.ndarray]
            Image(s) to extract features from
        annotations : np.ndarray or list[np.ndarray], optional
            Annotations to extract features from, by default None
            Image and annotation lists must correspond to each other
        restore_input_form : bool, optional
            Whether to restore the input form of the features, by default True
            If True, no annotations are returned
            If False, features are returned in their original form if no annotations are given,
            or paired with the processed annotations if they are given
        memory_mode : bool, optional
            Whether to use memory mode, by default False
            If True, the annotations are registered and updated, and only features for new pixels are extracted
        img_ids : str or list[str], optional
            Image IDs to register the annotations with, by default None (assuming a single image input)
        skip_norm : bool, optional
            Whether to skip normalization of the images before feature extraction, by default False
            If True, the images are not normalized according to the parameter `normalize` in the model parameters

        Returns:
        ----------
        features : np.ndarray or list[np.ndarray]
            Extracted features of the image(s) or list of images
            Features dimension is added first
        annotations : np.ndarray or list[np.ndarray], optional
            Processed annotations of the image(s) or list of images, if annotations are given
        """
        
        # Check if we process annotations and if we have only a single image input
        use_annots = annotations is not None
        single_input = isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)
        input_shapes = [data.shape] if single_input else [d.shape for d in data]
        # Make sure img_ids are compatible and is made into a list
        if img_ids is not None and not single_input and not len(img_ids) == len(data):
            raise ValueError('Image IDs must have the same length as the data.')
        img_ids = [img_ids] if single_input else img_ids

        # Run checks on the model to ensure that it is ready to be used
        self._run_model_checks()

        # Make sure data is a list of images with [C, Z, H, W] shape
        data, annotations = self._prep_dims(data, annotations, get_coords=False)
        # print("Prepped data shapes:")
        # print(data[0].shape)
        # print("Memory mode:", memory_mode)
        # print("Annotations in get_feature_image:", len(annotations), "annot of shape", annotations[0].shape, "with values", np.unique(annotations[0]))

        # If not done previously, normalize the images (separately and according to the parameter)
        if not skip_norm:
            data = [self._norm_single_image(d) for d in data]

        # Save the original data shapes for reshaping/rescaling later
        self.original_shapes = [d.shape for d in data]

        # Get the parameters for feature extraction
        params_for_extract = self._enforce_fe_params(self._param)

        # Downsample the images and annotations (if given)
        data = [conv_paint_utils.scale_img(d,
                                           params_for_extract.image_downsample)
                for d in data]
        if use_annots:
            annotations = [conv_paint_utils.scale_img(a,
                                                      params_for_extract.image_downsample,
                                                      input_type="labels")
                           for a in annotations]
        if memory_mode:
            annotations = self._register_and_update_annots(annotations, img_ids,
                                                           params_for_extract.image_downsample)
            num_new_annot_pix = np.sum([annot > 0 for annot in annotations])
            # print(f"Number of new annotated pixels: {num_new_annot_pix}")
            if num_new_annot_pix == 0:
                warnings.warn("No new annotations. Train with existing features.")
                # print("No new annotations (just in case warning not shown).")
                return [], [], [], [], params_for_extract.image_downsample
            coords = [conv_paint_utils.get_coordinates_image(d) for d in data]
            # print("Downsampled data shapes:")
            # print(data[0].shape, coords[0].shape)
            # print(coords[0][:, 0, 20, 20])
        else:
            coords = [None for _ in data]  # No coordinates if not in memory mode
  
        # Get the correct padding size for the feature extraction    
        padding = self._get_total_pad_size(params_for_extract)
        # Pad the images and annotations with the correct padding size
        data = [conv_paint_utils.pad(d, padding) for d in data]
        # Save the padded shapes for reshaping/rescaling later
        self.padded_shapes = [d.shape for d in data]
        if use_annots:
            annotations = [conv_paint_utils.pad(a, padding, input_type="labels") for a in annotations]
        if memory_mode:
            coords = [conv_paint_utils.pad(c, padding, input_type="coords") for c in coords]        
            # print("Padded data shapes:")
            # print(data[0].shape, coords[0].shape)
            # print(coords[0][:, 0, 20, 20])
        else:
            coords = [None for _ in data]  # No coordinates if not in memory mode
        # print("Annotations after padding:", len(annotations), "annot of shape", annotations[0].shape, "with values", [np.unique(a[0]) for a in annotations])

        # Extract annotated planes and flatten them (treating as individual images)
        if use_annots:
            planes = [conv_paint_utils.get_annot_planes(d, a, c) for d, a, c in zip(data, annotations, coords)]
            data = [plane for plane_tuple in planes for plane in plane_tuple[0]]
            annotations = [plane for plane_tuple in planes for plane in plane_tuple[1]]
        if memory_mode:
            coords = [plane for plane_tuple in planes for plane in plane_tuple[2]]
            # If img_ids are given, flatten them as well (corresponding id for each plane)
            if img_ids is not None:
                img_ids = [img_ids[i] for i, plane_tuple in enumerate(planes) for _ in range(len(plane_tuple[0]))]
            else:
                img_ids = [None] * len(data)
            # print("Extracted annotated planes shapes:")
            # print(data[0].shape, coords[0].shape)
            # print(coords[0][:, 0, 20, 20])
            # print(img_ids)
        else:
            coords = [None for _ in data]
        # print("Annotations after extraction:", len(annotations), "annot of shape", annotations[0].shape, "with values", [np.unique(a[0]) for a in annotations])


        # Get annotated tiles (if enabled) and flatten them (treating as individual images)
        if params_for_extract.tile_annotations:
            if use_annots:
                # print(coords)
                # Make sure we have a list of coords for each annot, even if each is None
                coords = [None for _ in data] if coords is None else coords
                # print("coords", len(coords))
                # print("data:", len(data), [d.shape for d in data])
                # print("annotations:", len(annotations), [np.unique(a) for a in annotations])
                tiles = [conv_paint_utils.tile_annot(d, a, c, padding, plot_tiles=False)
                        for d, a, c in zip(data, annotations, coords)]
                # print("tiles:", len(tiles))
                data = [tile for tile_tuple in tiles for tile in tile_tuple[0]]
                annotations = [tile for tile_tuple in tiles for tile in tile_tuple[1]]
            if memory_mode:
                coords = [tile for tile_tuple in tiles for tile in tile_tuple[2]]
                # If img_ids are given, flatten them as well (corresponding id for each tile)
                if img_ids is not None:
                    img_ids = [img_ids[i] for i, tile_tuple in enumerate(tiles) for _ in range(len(tile_tuple[0]))]
                else:
                    img_ids = [None] * len(data)
                # print("Extracted annotated tiles shapes:")
                # print(data[0].shape, coords[0].shape)
                # print(img_ids)
            else:
                coords = [None for _ in data]
        # print("Annotations after tiling:", len(annotations), "annot of shape", annotations[0].shape, "with values", [np.unique(a[0]) for a in annotations])

        # Extract features from the images for each scaling in the pyramid
        # For training, we want to "unpatch" the features to match the annotations
        keep_patched = not use_annots
        features = [self.fe_model.get_feature_pyramid(d, params_for_extract, patched=keep_patched)
                    for d in data]

        # Return the raw features if required
        if not restore_input_form:
            if memory_mode:
                return features, annotations, coords, img_ids, params_for_extract.image_downsample
            elif use_annots:
                return features, annotations
            else:
                return features

        # Reshape the features to the original image shape
        padded_shapes = self.padded_shapes
        original_shapes = self.original_shapes
        features = [self._restore_shape(features[i],
                                       padded_shapes[i],
                                       padding,
                                       original_shapes[i],
                                       class_preds=False)
                    for i in range(len(features))]
            
        # Restore input dimensionality (especially see if we want to remove z dimension)
        features = [self._restore_dims(features[i], input_shapes[i])
                            for i in range(len(data))]

        # If input was a single image, return the first (= only) features
        if single_input:
            return features[0]

        return features


### CLASSIFIER METHODS

    def _clf_train(self, features, targets, use_rf=False, allow_writing_files=False):
        """
        Trains a classifier given a set of features and targets.
        If use_rf is False, a CatBoostClassifier is trained, otherwise a RandomForestClassifier.
        The trained classifier is saved in the model, but also returned.

        Parameters:
        ----------
        features : np.ndarray
            Features to train the classifier on
        targets : np.ndarray
            Targets to train the classifier on
        use_rf : bool, optional
            Whether to use a RandomForestClassifier, by default False
        allow_writing_files : bool, optional
            Allow writing files for the CatBoostClassifier, by default False

        Returns:
        ----------
        clf : CatBoostClassifier or RandomForestClassifier
            Trained classifier
        """
        if not use_rf:
            # NOTE: THIS, FOR NOW, ASSUMES THAT GPU SHALL BE USED FOR CLF IF CHOSEN FOR FE
            use_gpu = self._param.clf_use_gpu if self._param.clf_use_gpu is not None else self._param.fe_use_cuda
            task_type = conv_paint_utils.get_catboost_device(use_gpu)
            self.classifier = CatBoostClassifier(iterations=self._param.clf_iterations,
                                                 learning_rate=self._param.clf_learning_rate,
                                                 depth=self._param.clf_depth,
                                                 allow_writing_files=allow_writing_files,
                                                 task_type=task_type
                                                 )
            self.classifier.fit(features, targets)
            self._param.classifier = 'CatBoost'
        else:
                # train a random forest classififer (does not support GPU)
                self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                self.classifier.fit(features, targets)
                self._param.classifier = 'RandomForest'

        clf = self.classifier
        return clf
    
    def _clf_predict(self, features, return_proba=True):
        """
        Uses the trained classifier to predict classes based on features of the image.
        Returns an image containing the probabilities of each class for each pixel.
        If return_proba is False, the predicted class is returned instead.

        Parameters:
        ----------
        features : np.ndarray
            Features to use for the prediction; features dimension is expected first
        return_proba : bool, optional
            Whether to return the prediction as class probabilities, by default True

        Returns:
        ----------
        predictions : np.ndarray
            Predicted classes or probabilities of the classes for each pixel in the image
            Class dimension is added first (if return_proba is True)
        """
        nb_features = features.shape[0] # [nb_features, width, height]

        # Move features to last dimension and flatten
        features = np.moveaxis(features, 0, -1)
        features = np.reshape(features, (-1, nb_features)) # flatten

        # Predict
        if return_proba:
            predictions = self.classifier.predict_proba(features)
            predictions = np.moveaxis(predictions, -1, 0) # [nb_classes, width*height]
        else:
            predictions = self.classifier.predict(features)

        return predictions

    def reset_classifier(self):
        """
        Resets the classifier of the model.
        """
        self.classifier = None
        self._param.classifier = None


### BACKEND TRAINING AND PREDICTION METHODS

    def _train(self, data, annotations, memory_mode=False, img_ids=None, use_rf=False, allow_writing_files=False, skip_norm=False):
        """
        Backend training method for the Convpaint model.
        """

        if not memory_mode:
            # Use get_feature_image to extract features and the suiting annotation parts (returns lists if restore_input_form=False)
            features, annot_parts = self.get_feature_image(data, annotations, memory_mode=memory_mode, restore_input_form=False, img_ids=img_ids, skip_norm=skip_norm)
            # Get the annotated pixels and targets, and concatenate each
            f_t_tuples = [conv_paint_utils.get_features_targets(f, a)
                        for f, a in zip(features, annot_parts)] # f and t are linearized
            features = [ft[0] for ft in f_t_tuples] # Create a list of features
            targets = [ft[1] for ft in f_t_tuples] # Create a list of targets
            features = np.concatenate(features, axis=0) # Concatenate the features into a single array
            targets = np.concatenate(targets, axis=0) # Concatenate the targets into a single array

        else: # memory mode
            # Use get_feature_image to extract features and the suiting annotation parts (returns lists if restore_input_form=False)
            features, annot_parts, coords, img_ids, scale = self.get_feature_image(data, annotations, memory_mode=memory_mode, restore_input_form=False, img_ids=img_ids, skip_norm=skip_norm)
            # Get all annotations and features from the table
            features, targets = self._register_and_get_all_features_annots(features, annot_parts, coords, img_ids, scale)

        # Check if we have features and targets, and that we have at least two classes
        if len(features) == 0 or len(targets) == 0:
            raise ValueError('No features or targets found. Please check the input data and annotations.')
        if len(np.unique(targets)) < 2:
            # print("Targets:", targets, "values:", np.unique(targets))
            raise ValueError('Not enough classes found in the targets. At least two classes are required for training.')

        # Train the classifier
        self._clf_train(features, targets,
                               use_rf=use_rf, allow_writing_files=allow_writing_files)

        return self.classifier

    def reset_training(self):
        """
        Resets the features and targets of the model. These can be used to
        iteratively train the model with the option memory_mode.

        Parameters:
        ----------
        None
        """
        self.table = pd.DataFrame(columns=['img_id', 'scale', 'coordinates', 'label', 'features_array'])
        self.table.index.name = "row_key"
        self.annot_dict = {}

    def _register_and_update_annots(self, annotations, img_ids, scale=1):
        """
        Registers new annotations in the table and deletes existing ones from the new annotations.

        Parameters:
        ----------
        annotations : list[np.ndarray]
            List of annotations to register, each with shape [C, Z, H, W]
        img_ids : list[str]
            List of image IDs corresponding to the annotations, each as a string

        Returns:
        ----------
        all_new_annots : list[np.ndarray]
            List of new annotations with the same shape as the input annotations,
            where existing annotations are set to zero in the new annotations.
        """

        if not len(annotations) == len(img_ids):
            raise ValueError('Annotations and image IDs must have the same length.')

        # First update the table with the new annotations and features
        all_new_annots = []
        for a, id in zip(annotations, img_ids):
            dict_key = f"{id}_scale{scale}"
            old_annot = self.annot_dict.get(dict_key, None)
            # print("Annotated in old:", np.sum(old_annot>0) if old_annot is not None else None)
            # Save the newest version of annotations in the dictionary
            self.annot_dict[dict_key] = a.copy()
            # Save the annotations to extract in a copy, from which duplicates will be removed
            annot_to_extract = a.copy()

            if old_annot is not None:
                # Which ones can be ignored for feature extraction, since they are already in the table
                mask_to_ignore = old_annot == annot_to_extract
                # print("Num pixels to ignore:", np.sum(mask_to_ignore))
                # Which ones can be removed from the table, since they are different to the new annotations
                mask_to_remove = (old_annot > 0) & (old_annot != annot_to_extract)
                # print("Num pixels to remove:", np.sum(mask_to_remove))
                coords_to_remove = np.argwhere(mask_to_remove)

                # Set the pixels that are the same as the old annotation to zero (no need to extract features)
                annot_to_extract[mask_to_ignore] = 0

                # coords_to_remove: shape [N, 3], rows of (z, h, w)
                if len(coords_to_remove) > 0:
                    row_keys_to_remove = [
                        f"{id}_scale{scale}_{z}-{h}-{w}_{old_annot[z, h, w]}"
                        for z, h, w in coords_to_remove
                    ]
                    # Drop all entries in the table that match the row keys to remove
                    self.table.drop(index=row_keys_to_remove, errors='ignore', inplace=True)

            # Add the new annotations to the list
            all_new_annots.append(annot_to_extract)

        return all_new_annots
    
    def _register_and_get_all_features_annots(self, features, annotations, coords, img_ids, scale):
        """
        Registers new features in the table and returns all features and annotations.
        IMPORTANT: This assumes that double entries have been removed from the annotations and old entries from the table.

        Parameters:
        ----------
        features : list[np.ndarray]
            List of features to register, each with shape [C, Z, H, W]
        annotations : list[np.ndarray]
            List of annotations to register, each with shape [C, Z, H, W]
        coords : list[np.ndarray]
            List of coordinates corresponding to the annotations, each with shape [C, Z, H, W]
        img_ids : list[str]
            List of image IDs corresponding to the annotations, each as a string
        
        Returns:
        ----------
        features : list[np.ndarray]
            List of all features with the same shape as the input features,
            where existing features were kept and new features were added.
        """
        if not (len(features) == len(annotations)
                and len(features) == len(img_ids) 
                and len(features) == len(coords)):
            raise ValueError('Features, annotations, coordinates, and image IDs must have the same length.')

        # First add the new annotations and features to the annotations table
        for f, a, c, id in zip(features, annotations, coords, img_ids):
            entries = []
            all_non_empties = np.argwhere(a > 0)
            for non_empty in all_non_empties:
                # Get the coordinates of the non-empty pixels
                new_features = f[:, non_empty[0], non_empty[1], non_empty[2]]
                non_empty = tuple(non_empty)
                new_annot = a[non_empty]
                coord = tuple(c[:, non_empty[0], non_empty[1], non_empty[2]])

                # Add the new entry to the table
                new_entry = {"img_id": id,
                             "scale": scale,
                             "coordinates": coord,
                             "label": new_annot,
                             "features_array": new_features,
                             "row_key": f"{id}_scale{scale}_{coord[0]}-{coord[1]}-{coord[2]}_{new_annot}"}
                entries.append(new_entry)

            # Create a DataFrame from the entries and append it to the table
            new_entries_df = pd.DataFrame(entries)
            new_entries_df.set_index("row_key", inplace=True)
            self.table = pd.concat([self.table, new_entries_df])

        # Now extract all features and annotations from the table
        features = list(self.table['features_array'].values)
        annotations = list(self.table['label'].values)

        return features, annotations

    def _predict(self, data, add_seg=False, skip_norm=False):
        """
        Backend method to predict images as a whole or tiling and parallelizing the prediction.
        Returns the class probabilities and optionally the segmentation of the images.
        Results are reshaped to the original image shape, and dimensions are restored to the ones
        of the original image, without the channel dimension,
        but with the class dimension (for class probabilities) added first.
        """

        # Check if there is a trained classifier
        if self.classifier is None:
            raise ValueError('No trained classifier found.')

        # Check if we process annotations and if we have only a single image input
        single_input = isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)
        input_shapes = [data.shape] if single_input else [d.shape for d in data]

        # Make sure data is a list of images with [C, Z, H, W] shape
        data, _ = self._prep_dims(data)

        # If not done previously, normalize the images (separately and according to the parameter)
        if not skip_norm:
            data = [self._norm_single_image(d) for d in data]

        # Get class probabilities, using tiling if enabled
        if self._param.tile_image:
            probas = [self._parallel_predict_image(d, return_proba=True)
                      for d in data]
        else:
            probas = self._predict_image(data, return_proba=True) # Can handle lists

        # Restore input dimensionality (especially see if we want to remove z dimension)
        probas = [self._restore_dims(probas[i], input_shapes[i])
                            for i in range(len(data))]

        # Determine what to return
        if add_seg:
            seg = [self._probas_to_classes(p) for p in probas]
            if single_input:
                return probas[0], seg[0]
            else:
                return probas, seg
        else:
            if single_input:
                return probas[0]
            else:
                return probas

    def _predict_image(self, image, return_proba=True):
        """
        Backend method to predict images without tiling and parallelization.
        Returns the class probabilities and optionally the segmentation of the images.
        Reshapes the predictions to the original image shape, without the channel dimension,
        but always keeps the Z dim --> [C, Z, H, W] for class probas, [Z, H, W] for segmentation.
        NOTE: This allows to use this method interchangably with _parallel_predict_image,
              but also within the _parallel_predict_image method for each tile.
        """

        # Use get_feature_image to extract features
        features = self.get_feature_image(image, restore_input_form=False, skip_norm=True) # NOTE: No normalization here, as it is done outside

        # Predict pixels based on the features and classifier
        # NOTE: We always first predict probabilities and then take the argmax
        predictions = [self._clf_predict(f, return_proba=True)
                       for f in features]

        # Get the parameters that were used for feature extraction (for reshaping)
        params_for_extract = self._enforce_fe_params(self._param)
        padding = self._get_total_pad_size(params_for_extract)

        # Reshape the predictions to the original image shape
        padded_shapes = self.padded_shapes # Saved when extracting features
        original_shapes = self.original_shapes # Saved when extracting features
        pred_reshaped = [self._restore_shape(predictions[i],
                                            padded_shapes[i],
                                            padding,
                                            original_shapes[i],
                                            class_preds=False)
                         for i in range(len(predictions))]
    
        # If we want classes, take the argmax of the probabilities
        if not return_proba:
            pred_reshaped = [self._probas_to_classes(p) for p in pred_reshaped]

        # If input was a single image, return the first prediction
        if isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
            return pred_reshaped[0]

        return pred_reshaped

    def _parallel_predict_image(self, image, return_proba=True):
        """
        Backend method to predict an image using tiling and parallelization.
        Returns the class probabilities and optionally the segmentation of the images.
        Reshapes the predictions to the original image shape, without the channel dimension,
        but always keeps the Z dim --> [C, Z, H, W] for class probas, [Z, H, W] for segmentation.
        NOTE: As opposed to other methods, this method only takes single images as input.
        """

        maxblock = 1000
        nblocks_rows = image.shape[-2] // maxblock
        nblocks_cols = image.shape[-1] // maxblock
        margin = 50
        use_dask = self._param.use_dask

        image = self._prep_dims_single(image)[0] # NOTE: should technically not be necessary, as done outside

        # Prepare array to write the predictions to
        if return_proba:
            num_classes = self.classifier.classes_.shape[0]
            z, h, w = image.shape[-3:]
            predicted_image_complete = np.zeros((num_classes, z, h, w),
                                                dtype=(np.float32))
        else:
            predicted_image_complete = np.zeros(image.shape[-3:], dtype=(np.uint8))

        # Prepare dask client if enabled
        if use_dask:
            from dask.distributed import Client
            import dask
            dask.config.set({'distributed.worker.daemon': False})
            client = Client()
            processes = []

        # Iterate over the blocks of the image and predict each block separately
        min_row_ind_collection = []
        min_col_ind_collection = []
        max_row_ind_collection = []
        max_col_ind_collection = []
        new_max_col_ind_collection = []
        new_max_row_ind_collection = []
        new_min_col_ind_collection = []
        new_min_row_ind_collection = []

        for row in range(nblocks_rows+1):
            for col in range(nblocks_cols+1):
                min_row = np.max([0, row*maxblock-margin])
                min_col = np.max([0, col*maxblock-margin])
                max_row = np.min([image.shape[-2], (row+1)*maxblock+margin])
                max_col = np.min([image.shape[-1], (col+1)*maxblock+margin])

                min_row_ind = 0
                new_min_row_ind = 0
                if min_row > 0:
                    min_row_ind = min_row + margin
                    new_min_row_ind = margin
                min_col_ind = 0
                new_min_col_ind = 0
                if min_col > 0:
                    min_col_ind = min_col + margin
                    new_min_col_ind = margin

                max_col = (col+1)*maxblock+margin
                max_col_ind = np.min([min_col_ind+maxblock,image.shape[-1]])
                new_max_col_ind = new_min_col_ind + (max_col_ind-min_col_ind)
                if max_col > image.shape[-1]:
                    max_col = image.shape[-1]
                max_row = (row+1)*maxblock+margin
                max_row_ind = np.min([min_row_ind+maxblock,image.shape[-2]])
                new_max_row_ind = new_min_row_ind + (max_row_ind-min_row_ind)
                if max_row > image.shape[-2]:
                    max_row = image.shape[-2]

                image_block = image[..., min_row:max_row, min_col:max_col]

                # Predict the block using dask or directly
                if use_dask:
                    processes.append(client.submit(
                        self._predict_image, image=image_block, return_proba=return_proba))
                    
                    min_row_ind_collection.append(min_row_ind)
                    min_col_ind_collection.append(min_col_ind)
                    max_row_ind_collection.append(max_row_ind)
                    max_col_ind_collection.append(max_col_ind)
                    new_max_col_ind_collection.append(new_max_col_ind)
                    new_max_row_ind_collection.append(new_max_row_ind)
                    new_min_col_ind_collection.append(new_min_col_ind)
                    new_min_row_ind_collection.append(new_min_row_ind)

                else:
                    predicted_image = self._predict_image(image_block, return_proba=return_proba)
                    crop_pred = predicted_image[...,
                        new_min_row_ind: new_max_row_ind,
                        new_min_col_ind: new_max_col_ind]
                    if not return_proba:
                        crop_pred = crop_pred.astype(np.uint8)
                    predicted_image_complete[...,
                        min_row_ind:max_row_ind,
                        min_col_ind:max_col_ind] = crop_pred

        # Terminate dask processes if enabled
        if use_dask:
            for k in range(len(processes)):
                future = processes[k]
                out = future.result()
                crop_out = out[...,
                    new_min_row_ind_collection[k]:new_max_row_ind_collection[k],
                    new_min_col_ind_collection[k]:new_max_col_ind_collection[k]]
                if not return_proba:
                    crop_out = crop_out.astype(np.uint8)
                # Write the result to the complete image
                future.cancel()
                del future
                predicted_image_complete[...,
                    min_row_ind_collection[k]:max_row_ind_collection[k],
                    min_col_ind_collection[k]:max_col_ind_collection[k]] = crop_out
            client.close()
        
        return predicted_image_complete


### HELPER METHODS

    def _run_model_checks(self):
        """
        Run checks on the model to ensure that it is ready to be used.
        """
        if self.fe_model is None:
            raise ValueError('No feature extractor model set. Please set a feature extractor model first.')
    
    def _prep_dims(self, data, annotations=None, get_coords=False):
        """
        Preprocess the dimensions of the data and annotations (if given).
        Ensures the data and annotations have the correct number of dimensions,
        where the data is a list of images with [C, Z, H, W] shape,
        and the annotations are a list of images with [Z, H, W] shape.
        Assumes that a possible channels dimension is the first one.
        Handles both single images/annotations and lists of images/annotations.
        """
        # Ensure data and annotations are lists
        if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            img_list = [data]
        elif isinstance(data, tuple):
            img_list = list(data)
        elif isinstance(data, list):
            img_list = data
        else:
            raise ValueError('Data must be a numpy array, torch tensor, or a list/tuple of images.')
        if annotations is None:
            annot_list = [None] * len(img_list)
        elif isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
            annot_list = [annotations]
        elif isinstance(data, tuple):
            annot_list = list(annotations)
        elif isinstance(data, list):
            annot_list = annotations
        else:
            raise ValueError('Annotations must be a numpy array, torch tensor, or a list/tuple of images.')

        # Check if the lengths of the data and annotations lists are equal
        if len(img_list) != len(annot_list):
            raise ValueError('Data and annotations lists must have the same length.')
        
        # Check and preprocess the dimensions of each data and annotations pairs
        prep_imgs, prep_annots = zip(*[
            self._prep_dims_single(item, annot)
            for item, annot in zip(img_list, annot_list)
        ])

        # Check if the dimension type of all images is the same (same num of dims and channels)
        for i, img in enumerate(prep_imgs):
            if img.ndim != prep_imgs[0].ndim:
                raise ValueError(f'Image {i} has different number of dimensions than the first image.')
            if img.shape[0] != prep_imgs[0].shape[0]:
                raise ValueError(f'Image {i} has different number of channels than the first image.')

        # If we want coordinates, create a coordinates img for each image in data
        if get_coords:
            coords = [conv_paint_utils.get_coordinates_image(img)
                      for img in prep_imgs]
            return list(prep_imgs), list(prep_annots), coords
        # Otherwise, just return the preprocessed images and annotations
        else:
            return list(prep_imgs), list(prep_annots)
        
    def _prep_dims_single(self, img, annotations=None):
        """
        Preprocess the dimensions of the img and annotations (if given).
        Ensures the data and annotations have the correct number of dimensions,
        where the data is a list of images with [C, Z, H, W] shape,
        and the annotations are a list of images with [Z, H, W] shape.
        Assumes that a possible channels dimension is the first one.
        
        Returns: img, annotations (= None if not given)
        """
        num_dims = img.ndim
        if num_dims == 2:
            # Add a channels and a z dimension
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)
        elif num_dims == 3:
            if self._param.multi_channel_img or self._param.rgb_img:
                # Add a z dimension at the second position
                img = np.expand_dims(img, axis=1)
            else:
                # Add a channels dimension at the first position
                img = np.expand_dims(img, axis=0)
        elif num_dims == 4:
            # Data is already in [C, Z, H, W] format
            pass
        else:
            raise Exception('Image has wrong number of dimensions')
        
        if annotations is None:
            return img, None
        
        num_dims_annots = annotations.ndim
        if num_dims_annots == 2:
            # Add a z dimension at the first position
            annotations = np.expand_dims(annotations, axis=0)
        elif num_dims_annots == 3:
            # Annotations are already in [Z, H, W] format
            pass
        else:
            raise Exception('Annotations have wrong number of dimensions')
        
        if img.shape[1:] != annotations.shape:
            raise Exception('Image and annotations have different dimensions')

        return img, annotations
    
    def _norm_single_image(self, img):
        """
        Normalize a single image.
        Assumes that the image is in [C, Z, H, W] format.
        """
        if img.ndim != 4:
            raise ValueError('Image must be given as 4 dimensions (C, Z, H, W).')
        
        # Get the normalization mode from the parameters
        if self._param.normalize is None:
            norm_mode = 1 # If not specified, do not normalize
        else:
            norm_mode = self._param.normalize
        
        # If normalization is off (= 1), we do not normalize
        if norm_mode == 1:
            return img

        # Otherwise we generally normalize each channel separately
        # if normalization is "by image" (= 3), we additionally keep the z dimension
        num_ignored_channels = 2 if norm_mode == 3 else 1
        mean, sd = conv_paint_utils.compute_image_stats(img, ignore_n_first_dims=num_ignored_channels)
        
        # Normalize using these statistics
        img_norm = conv_paint_utils.normalize_image(img, mean, sd)

        # print(f'Image normalized with shapes for mean {mean.shape} and sd {sd.shape}.')
        # print(f'Channel means: {np.mean(img_norm, axis=(1,2,3))}, planes means: {np.mean(img_norm, axis=(0,2,3))}')
        # print("Overall mean:", np.mean(img_norm), "overall sd:", np.std(img_norm))

        return img_norm

    def _enforce_fe_params(self, old_param):
        """
        Returns the parameters specified by the feature extractor to enforce for the extraction.
        Raises a warning if any of the parameters are enforced by the feature extractor model.
        """
        enforced_params = self.fe_model.get_enforced_params()
        new_param = old_param.copy()
        # Raise a warning if anything is enforced
        for key in enforced_params.get_keys():
            if (enforced_params.get(key) is not None and
                old_param.get(key) != enforced_params.get(key)):
                warnings.warn(f'Parameter {key} is enforced by the feature extractor model.')
                new_param.set_single(key, enforced_params.get(key))
        return new_param
    
    def _get_total_pad_size(self, param):
        """
        Returns the required padding size for the feature extraction.
        The padding size is calculated based on the feature extractor model and the parameters.
        Takes into account, padding size, patch size and scaling factors.
        Half the patch size is added to the padding size to ensure that the image is not
        reduced too much when ensuring the patch size.
        The padding size is finally also scaled by the highest scaling factor.
        """
        # Get the padding size from the feature extractor model
        fe_pad = self.fe_model.get_padding()
        # Make sure the reducing to patch size will not reduce below the necessary padding
        fe_patch = self.fe_model.get_patch_size()
        # Scale the padding by the highest scaling factors (such that the image is not reduced too much)
        max_scale = np.max(param.fe_scalings)
        # Calculate the final padding size
        return (fe_pad + fe_patch//2) * max_scale
    
    def _restore_shape(self, outputs, processed_shape, padding, original_shape, class_preds=False):
        """
        Reshape the outputs (prediction, probabilities, features) to the original image shape.
        """
        # Prepare the variables
        prep_z, prep_h, prep_w = processed_shape[-3:]
        if self.fe_model.gives_patched_features():
            patch_h = prep_h // self.fe_model.get_patch_size()
            patch_w = prep_w // self.fe_model.get_patch_size()
        else:
            patch_h, patch_w = prep_h, prep_w
        orig_z, orig_h, orig_w = original_shape[-3:]
        num_outputs = None if class_preds else outputs.shape[0]

        # Reshape to (possibly patched) spatial dimensions
        if class_preds:
            outputs_image = np.reshape(outputs, [prep_z, patch_h, patch_w])
        else:
            outputs_image = np.reshape(outputs, [num_outputs, prep_z, patch_h, patch_w])

        # Then resize to downsampled, padded shape
        if class_preds:
            new_shape = (prep_z, prep_h, prep_w)
            outputs_image = conv_paint_utils.rescale_class_labels(
                outputs_image, output_shape=new_shape)
        else:
            new_shape = (num_outputs, prep_z, prep_h, prep_w)
            outputs_image = conv_paint_utils.rescale_outputs(
                outputs_image, output_shape=new_shape,
                order=1)

        # Remove padding
        outputs_image = outputs_image[..., padding:-padding, padding:-padding]

        # Then resize to original shape
        if self._param.image_downsample > 1:
            if class_preds:
                outputs_image = conv_paint_utils.rescale_class_labels(
                    outputs_image, output_shape=(orig_z, orig_h, orig_w))
            else:
                new_shape = (num_outputs, orig_z, orig_h, orig_w)
                outputs_image = conv_paint_utils.rescale_outputs(
                    outputs_image, output_shape=new_shape,
                    order=1)

        return outputs_image

    def _probas_to_classes(self, probas):
        """
        Convert probabilities to classes. Smoothen output if requested.
        """
        class_labels = self.classifier.classes_
        max_prob = np.argmax(probas, axis=0)
        seg = class_labels[max_prob].astype(np.uint8)

        # Smoothen the segmentation if desired
        if self._param.seg_smoothening > 1:
            kernel = skimage.morphology.disk(self._param.seg_smoothening)
            if seg.ndim == 3: # 3D image --> add z dimension to kernel
                kernel = np.expand_dims(kernel, axis=0)
            seg = skimage.filters.rank.majority(seg, footprint=kernel)

        return seg

    def _restore_dims(self, pred, original_shape):
        """
        Restore the dimensions of the prediction to the original shape.
        """
        is_2d = len(original_shape) == 2
        is_3d = len(original_shape) == 3
        is_3d_multi = is_3d and (self._param.multi_channel_img or self._param.rgb_img)
        # if is 2d or 3D with multiple channels (including RGB), we remove the z dimension
        if is_2d or is_3d_multi:
            if pred.shape[-3] != 1:
                raise ValueError('Prediction shape does not match original shape.')
            # [Z,H,W] for prediction, [proba/F,Z,H,W] for probabilities and features
            pred = np.squeeze(pred, axis=-3)
        return pred