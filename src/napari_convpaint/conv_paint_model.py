
import pickle
from pyexpat import features
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
import skimage
import torch
import pandas as pd
from typing import Tuple
from math import lcm

from napari_convpaint.conv_paint_nnlayers import AVAILABLE_MODELS as NN_MODELS
from napari_convpaint.conv_paint_gaussian import AVAILABLE_MODELS as GAUSSIAN_MODELS
from napari_convpaint.conv_paint_dino import AVAILABLE_MODELS as DINO_MODELS
from napari_convpaint.conv_paint_dino_jafar import AVAILABLE_MODELS as DINO_JAFAR_MODELS
from napari_convpaint.conv_paint_combo_fe import AVAILABLE_MODELS as COMBO_MODELS
from napari_convpaint.conv_paint_nnlayers import Hookmodel
from napari_convpaint.conv_paint_gaussian import GaussianFeatures
from napari_convpaint.conv_paint_dino import DinoFeatures
from napari_convpaint.conv_paint_dino_jafar import DinoJafarFeatures
from napari_convpaint.conv_paint_combo_fe import ComboFeatures
from napari_convpaint.conv_paint_param import Param
from . import conv_paint_utils

class ConvpaintModel:
    """
    The `ConvpaintModel` class is the core of Convpaint and **combines feature extraction with pixel classification**.
    It is the main class **used for both the Convpaint napari plugin and the Convpaint API**.
    This setup allows for a similar workflow in both environments and lets the user **transition between them seamlessly**.

    Each `ConvpaintModel` instance in essence consists of three **components**:

    - a `feature extractor` model, defined in a separate class
    - a `classifier`, which is responsible for the final pixel classification
    - a `Param` object, which defines the details of the model procedures, and is also defined in a separate class

    Note that the `ConvpaintModel` and its **`FeatureExtractor` model** are closely linked to each other. The intended way to
    use them is to create a `ConvpaintModel` instance, which will in turn create the corresponding `FeatureExtractor` instance.
    If a `ConvpaintModel` with another feature extractor is desired (including different configurations in layers or GPU usage),
    a new `ConvpaintModel` instance should be created. Other parameters of the `ConvpaintModel`, though, can easily be changed later.

    Input **image dimensions** and channels (convention across all training, prediction and feature extraction methods):

    - 2D inputs: Will be treated as single-channel (gray-scale) images; if the feature extractor takes multiple input channels,
      the input will be repeated across channels.
    - 3D inputs: Dependent on the `channel_mode` parameter in the `Param` object, the first dimension will either be treated as channels
      (if `channel_mode` is "multi" or "rgb") or as a stack of 2D images (if `channel_mode="single"`);
      in the "single" case, if the feature extractor takes multiple input channels, the single input channel will be repeated accordingly.
    - 4D inputs: Will be treated as a stack of multi-channel images, with the first dimension as channels.
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
                  "cellpose": Param(fe_name="cellpose_backbone"),
                  "ilastik": Param(fe_name="ilastik_2d"),
                  "dino-jafar": Param(fe_name="dino_jafar_small"),
                  }
    
    allowed_param_vals = {
        'channel_mode': ['multi', 'rgb', 'single'],
        'normalize': [1, 2, 3],
        'image_downsample': list(range(-20, 21)), # from -20 (upsampling by factor 20) to 20 (downsampling by factor 20)
        'seg_smoothening': list(range(0, 21)), # from 0 (no smoothening) to 20
        'umpatch_order': list(range(0, 6)), # from 0 (nearest) to 5
        'fe_order': list(range(0, 6)), # from 0 (nearest) to 5
    }

    def __init__(self, alias=None, model_path=None, param=None, fe_name=None, **kwargs):
        """
        **Initializes a Convpaint model**. This can be done with an alias, a model path, a param object, or a feature extractor name.
        If initialized by FE name, also other parameters can be given to override the defaults of the FE model.

        If neither an alias, a model path, a param object, nor a feature extractor name is given,
        a **default Convpaint model** is created (defined in the `get_default_params()` method).

        The four **ways for initialization** in detail:

        1. By providing an **alias** (e.g., "vgg-s", "dino", "gaussian", etc.), in which case a corresponding model
        configuration will be loaded.
        2. By providing a **saved model path** (model_path) to load a model defined earlier.
        This can be a .pkl file (holding the FE model, classifier and Param object) or .yml file (only defining the model parameters).
        3. By providing a **Param object**, which contains model parameters.
        4. By providing the **name of the feature extractor** and other parameters such as GPU usage, and feature extraction layers,
        in which case these kwargs will override the defaults of the feature extractor model.
        
        Parameters
        ----------
        alias : str, optional
            Alias of a predefined model configuration
        model_path : str, optional
            Path to a saved model configuration
        param : Param, optional
            Param object with the model parameters (see Param class for details)
        fe_name : str, optional
            Name of the feature extractor model
        **kwargs : additional parameters, optional
            Additional parameters to override defaults for the model or feature extractor (see Param class for details)

        Raises
        ------
        ValueError
            If more than one of alias, model_path, param, or fe_name is provided.
        """

        # Initialize parameters and features
        self._param = Param()
        self.reset_training()
        self.num_trainings = 0
        self.num_features = 0
        self._params_to_reset_training = ['channel_mode',
                                          'normalize',
                                        #   'image_downsample',
                                        #   'seg_smoothening',
                                        #   'tile_annotations',
                                        #   'tile_image',
                                        #   'use_dask',
                                        #   'unpatch_order',
                                          'fe_name',
                                        #   'fe_use_gpu',
                                          'fe_layers',
                                          'fe_scalings',
                                          'fe_order',
                                          'fe_use_min_features',
                                        #   'clf_iterations',
                                        #   'clf_learning_rate',
                                        #   'clf_depth',
                                        #   'clf_use_gpu'
                                          ]
        self._params_to_reset_clf = ['channel_mode',
                                     'normalize',
                                    #  'image_downsample',
                                    #  'seg_smoothening',
                                    #  'tile_annotations',
                                    #  'tile_image',
                                    #  'use_dask',
                                    # 'unpatch_order',
                                     'fe_name',
                                    #  'fe_use_gpu',
                                     'fe_layers',
                                     'fe_scalings',
                                     'fe_order',
                                     'fe_use_min_features',
                                    #  'clf_iterations',
                                    #  'clf_learning_rate',
                                    #  'clf_depth',
                                    #  'clf_use_gpu'
                                     ]

        # Initialize the dictionary of all available models
        if not ConvpaintModel.FE_MODELS_TYPES_DICT:
            ConvpaintModel._init_fe_models_dict()

        num_kwargs_given =  + len(kwargs)
        if (alias is not None) + (model_path is not None) + (param is not None) + (fe_name is not None) > 1:
            raise ValueError('Please provide either an alias, a model path, a param object, or ' +
                             'a feature extractor name (and optionally additional kwargs) but not multiples.\n' +
                             'You can still set parameters later using the set_params() method.')
        
        if len(kwargs) > 0 and fe_name is None:
            raise ValueError('Providing kwargs is only intended when a feature extractor name is specified ' +
                             '(not with any other initialization method).')

        # If an alias is given, create an corresponding model
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
            fe_use_gpu = kwargs.pop('fe_use_gpu', None)
            fe_layers = kwargs.pop('fe_layers', None)
            self._set_fe(fe_name, fe_use_gpu, fe_layers)
            self._param = self.get_fe_defaults()
            self.set_params(ignore_warnings=True, # Here at initiation, it is intended to set FE parameters...
                            fe_layers = fe_layers, # Overwrite the layers with the given layers
                            fe_use_gpu = fe_use_gpu, # Overwrite the gpu usage with the given gpu usage
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
        ConvpaintModel.FE_MODELS_TYPES_DICT.update({x: DinoJafarFeatures for x in DINO_JAFAR_MODELS})

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

        Returns
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

        Returns
        ----------
        def_param : Param
            Param object with the default Convpaint model parameters.
        """

        def_param = Param()

        # Image type parameters
        def_param.channel_mode = "single" # Interpret the first dimension Z/t (as opposed to as channels)
        def_param.normalize = 2  # 1: no normalization, 2: normalize stack, 3: normalize each image

        # Input and output parameters
        def_param.image_downsample = 1
        def_param.seg_smoothening = 1
        def_param.tile_annotations = True
        def_param.tile_image = False
        def_param.use_dask = False
        def_param.unpatch_order = 1

        # FE parameters
        def_param.fe_name = "vgg16"
        def_param.fe_layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))']
        # def_param.fe_name = "convnext"
        # def_param.fe_layers = [0,1]
        def_param.fe_use_gpu = False
        def_param.fe_scalings = [1, 2, 4]
        def_param.fe_order = 0
        def_param.fe_use_min_features = False

        # Classifier parameters
        def_param.classifier = None
        def_param.clf_iterations = 100
        def_param.clf_learning_rate = 0.1
        def_param.clf_depth = 5

        return def_param
    
    def get_param(self, key):
        """
        Returns the value of the given parameter key.

        Parameters
        ----------
        key : str
            Key of the parameter to get
        
        Returns
        ----------
        param_value : any
            Value of the parameter with the given key
        """
        param_value = self._param.get(key)
        return param_value
    
    def get_params(self):
        """
        Returns all parameters of the model in form of a copy of the Param object.

        Returns
        ----------
        param : Param
            Copy of the Param object with all parameters of the model
        """
        param = self._param.copy()
        return param
    
    def set_param(self, key, val, ignore_warnings=False):
        """
        Sets the value of the given parameter key.

        Raises a warning if key FE parameters are set (intended only for initiation),
        unless ignore_warnings is set to True.

        Parameters
        ----------
        key : str
            Key of the parameter to set
        val : any
            Value to set the parameter to
        ignore_warnings : bool, optional
            Whether to suppress the warning for setting FE parameters
        """
        if not key in Param.get_keys():
            warnings.warn(f'Parameter "{key}" not found in the model parameters.')
            return
        if key in self.allowed_param_vals and val not in self.allowed_param_vals[key]:
            warnings.warn(f'Parameter "{key}" has value "{val}", which is not in the allowed values: {self.allowed_param_vals[key]}')
            return
        # if val == self._param.get(key):
        #     print(f"Parameter '{key}' already has the value {val}.")
        #     return
        if key in self._params_to_reset_training and self.num_trainings > 0 and not ignore_warnings:
            warnings.warn(f"When changing the parameter {key}, the features saved in memory mode should be reset.\n" +
                "The parameter was not changed. To do so, either first reset the features by calling the reset_training() method.\n"
                "Or, if you are aware of the consequences, you can also change the parameter without resetting by setting ignore_warnings to True.")
            return
        if key in self._params_to_reset_clf and self.classifier is not None and not ignore_warnings:
            warnings.warn(f"When changing the parameter {key}, the classifier should be reset.\n" +
                "The parameter was not changed. To do so, either first reset the classifier by calling the reset_classifier() method.\n"
                "Or, if you are aware of the consequences, you can also change the parameter without resetting by setting ignore_warnings to True.")
            return
        if key in ['fe_name', 'fe_use_gpu', 'fe_layers'] and not ignore_warnings:
            warnings.warn("Setting the parameters fe_name, fe_use_gpu, or fe_layers is not intended. " +
                "You should create a new ConvpaintModel instead.\n" +
                "If you are aware of the consequences, you can set ignore_warnings to True to change the parameter anyway.")
            return
        # Set the parameter
        self._param.set_single(key, val)

    def set_params(self, param=None, ignore_warnings=False, **kwargs):
        """
        Sets the parameters, given either as a Param object or as keyword arguments.

        Note that the model is not reset and no new FE model is created.
        If fe_name, fe_use_gpu, and fe_layers change, you should create a new ConvpaintModel.

        Parameters
        ----------
        param : Param, optional
            Param object with the parameters to set
        ignore_warnings : bool, optional
            Whether to suppress the warning for setting FE parameters
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
                self.set_param(key, val, ignore_warnings=ignore_warnings)

    def save(self, model_path, create_pkl=True, create_yml=True):
        """
        Saves the model to a pkl and/or yml file.

        The pkl file includes the classifier and the param object, as well as the
        features and targets (when using memory_mode).
        The yml file includes only the parameters of the model.

        Note: Loading a saved model is only intended at model initiation.

        Parameters
        ----------
        model_path : str
            Path to save the model to.
        create_pkl : bool, optional
            Whether to create a pkl file
        create_yml : bool, optional
            Whether to create a yml file
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
                    'annotations': self.annot_dict,
                    'num_features': self.num_features,
                    'num_trainings': self.num_trainings
                }
                pickle.dump(data, f)

        if create_yml:
            yml_path = model_path + ".yml"
            self._param.save(yml_path)

    def _load(self, model_path):
        """
        Loads the model from a file based on its extension.
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
        new_param = data.get('param', None)
        self._set_fe(new_param.fe_name, new_param.fe_use_gpu, new_param.fe_layers)
        self._param = new_param.copy()
        self.classifier = data.get('classifier', None)
        self.num_features = data.get('num_features', self.classifier.n_features_in_)
        self.num_trainings = data.get('num_trainings', 0)
        # If there is a features table and an annotations dictionary, load them
        if 'table' in data and 'annotations' in data:
            if not isinstance(data['table'], pd.DataFrame):
                raise ValueError('Table must be a pandas DataFrame.')
            if not isinstance(data['annotations'], dict):
                raise ValueError('Annotations must be a dictionary.')
            self.table = data['table']
            self.annot_dict = data['annotations']

    def _load_yml(self, yml_path):
        """
        Loads the model from a yml file.
        Only intended for internal use at model initiation.
        """
        new_param = ConvpaintModel.get_default_params() # Start with the default parameters
        loaded_param = Param.load(yml_path)
        if loaded_param is not None:
            # Only set the parameters that are not None in the loaded_param
            params_to_set = {key: value for key, value in loaded_param.__dict__.items() if value is not None}
            new_param.set(**params_to_set)
        self._set_fe(new_param.fe_name, new_param.fe_use_gpu, new_param.fe_layers)
        self._param = new_param

    def _load_param(self, param: Param):
        """
        Loads the given param object into the model and sets the model accordingly.
        Only intended for internal use at model initiation.
        """
        self._set_fe(param.fe_name, param.fe_use_gpu, param.fe_layers)
        self._param = self.get_fe_defaults()
        self.set_params(ignore_warnings=True, **param.__dict__) # Overwrite the parameters with the given parameters


### FE METHODS

    def _set_fe(self, fe_name=None, fe_use_gpu=None, fe_layers=None):
        """
        Sets the FE model based on the given FE parameters.
        Creates new feature extractor, and resets the classifier.
        Only intended for internal use at model initiation.
        """

        # Reset the model and classifier
        self.reset_classifier()
        self.reset_training()

        # Check if we need to create a new FE model
        fe_name_changed = fe_name is not None and fe_name != self._param.get("fe_name")
        fe_use_gpu_changed = fe_use_gpu is not None and fe_use_gpu != self._param.get("fe_use_gpu")
        fe_layers_changed = fe_layers is not None and fe_layers != self._param.get("fe_layers")

        # Create the feature extractor model
        if fe_name_changed or fe_use_gpu_changed or fe_layers_changed:
            self.fe_model = ConvpaintModel.create_fe(
                name=fe_name,
                use_gpu=fe_use_gpu,
                layers=fe_layers
            )
        
        # Set the parameters
        self._param.set(fe_name=fe_name, fe_use_gpu=fe_use_gpu, fe_layers=fe_layers)

    @staticmethod
    def create_fe(name, use_gpu=None, layers=None):
        """
        Creates a feature extractor model based on the given parameters.
        
        Importantly, the feature extractor model is returned, but not saved in the model.
        If you want to initialize a ConvpaintModel with this feature extractor,
        create a new instance of ConvpaintModel and specify the feature extractor parameters.

        Distinguishes between different types of feature extractors such as Hookmodels
        and initializes them accordingly.

        Parameters
        ----------
        name : str
            Name of the feature extractor model
        use_gpu : bool, optional
            Whether to use GPU for the feature extractor
        layers : list[str], optional
            List of layer names to extract features from

        Returns
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
                use_gpu=use_gpu,
                layers=layers
        )
        else:
            fe_model = fe_model_class(
                model_name=name,
                use_gpu=use_gpu
            )

        # Check if the model was created successfully
        if fe_model is None:
            raise ValueError(f'Feature extractor model {name} could not be created.')
        
        # Perform checks and fixes for RGB input and imagenet_normalization
        fe_model = ConvpaintModel._check_fix_fe_channels(fe_model)

        return fe_model
    
    @staticmethod
    def _check_fix_fe_channels(fe_model):
        """
        Checks and fixes channel_mode and imagenet_normalization of the feature extractor model,
        making sure it works with the input channels defined by the FE model.

        Result: Imagenet_norm is only used when the FE model takes RGB inputs, and RGB input is only
        used when the FE model takes 3 input channels.

        Parameters
        ----------
        fe_model : FeatureExtractor
            The feature extractor model to check

        Returns
        ----------
        fe_model : FeatureExtractor
            The (possibly modified) feature extractor model
        """
        # Make sure RGB is only used when the FE model has 3 input channels
        if fe_model.rgb_input and 3 not in fe_model.get_num_input_channels():
            warnings.warn(f'Feature extractor model {fe_model.model_name} does not take 3 input channels, ' +
                          'assuming FE takes non-RGB input.')
            fe_model.rgb_input = False
        
        # Make sure imagenet_normalization is only used when the FE model takes RGB inputs
        if fe_model.norm_mode == "imagenet" and not fe_model.rgb_input:
            warnings.warn(f'Feature extractor model {fe_model.model_name} does not take RGB inputs, ' +
                          'changing norm_mode from "imagenet" to "default".')
            fe_model.norm_mode = "default"

        return fe_model

    def get_fe_defaults(self):
        """
        Returns the default parameters of the feature extractor.

        Where they are not sepecified by the feature extractor model, the default Convpaint params are used.

        Returns
        ---------
        new_param : Param
            Convpaint model defaults adjusted to the feature extractor defaults
        """
        cpm_defaults = ConvpaintModel.get_default_params() # Get ConvPaint defaults
        new_param = self.fe_model.get_default_params(cpm_defaults) # Overwrite defaults defined in the FE model
        return new_param

    def get_fe_layer_keys(self):
        """
        Returns the keys of the feature extractor layers (None if the model uses no layers).

        Returns
        ---------
        keys : list[str] or None
            List of keys of the feature extractor layers, or None if the model uses no layers
        """
        keys = self.fe_model.get_layer_keys()
        return keys


### USER METHODS FOR TRAINING AND PREDICTION
    
    def train(self, image, annotations, memory_mode=False, img_ids=None, use_rf=False, allow_writing_files=False, in_channels=None, skip_norm=False):
        """
        Trains the Convpaint model's classifier given images and annotations.

        Uses the FeatureExtractor model to extract features according to
        the criteria specified in the Param object.
        Then uses the features of annotated pixels to train the classifier.
        
        Trains the internal classifier, which is also returned.

        Parameters
        ----------
        image : np.ndarray or list[np.ndarray]
            Image to train the classifier on or list of images
        annotations : np.ndarray or list[np.ndarray]
            Annotations to train the classifier on or list of annotations.
            Image and annotation lists must correspond to each other.
        memory_mode : bool, optional
            Whether to use memory mode.
            If True, the annotations are registered and updated, and only features for new pixels are extracted.
        img_ids : str or list[str], optional
            Image IDs to register the annotations with (when using memory_mode)
        use_rf : bool, optional
            Whether to use a RandomForestClassifier instead of a CatBoostClassifier
        allow_writing_files : bool, optional
            Whether to allow writing files for the CatBoostClassifier
        in_channels : list[int], optional
            List of channels to use for training
        skip_norm : bool, optional
            Whether to skip normalization of the images before training.
            If True, the images are not normalized according to the parameter `normalize` in the model parameters.

        Returns
        ----------
            clf : CatBoostClassifier or RandomForestClassifier
                Trained classifier (also saved inside the model instance)
        """
        clf, _, _ = self._train(image, annotations, memory_mode=memory_mode, img_ids=img_ids, use_rf=use_rf,
                          allow_writing_files=allow_writing_files, in_channels=in_channels, skip_norm=skip_norm)
        return clf

    def segment(self, image, in_channels=None, skip_norm=False, use_dask=False):
        """
        Segments images by predicting the most probable class of each pixel using the trained classifier.

        Uses the feature extractor model to extract features from the image.
        Predicts the classes of the pixels in the image using the trained classifier.

        Parameters
        ----------
        image : np.ndarray or list[np.ndarray]
            Image to segment or list of images
        in_channels : list[int], optional
            List of channels to use for segmentation
        skip_norm : bool, optional
            Whether to skip normalization of the images before segmentation.
            If True, the images are not normalized according to the parameter `normalize` in the model parameters.
        use_dask : bool, optional
            Whether to use dask for parallel processing

        Returns
        ----------
        seg : np.ndarray or list[np.ndarray]
            Segmented image or list of segmented images (according to the input)
            Dimensions are equal to the input image(s) without the channel dimension
        """
        _, seg = self._predict(image, add_seg=True, in_channels=in_channels, skip_norm=skip_norm, use_dask=use_dask)
        return seg

    def predict_probas(self, image, in_channels=None, skip_norm=False, use_dask=False):
        """
        Predicts the probabilities of the classes of the pixels in an image using the trained classifier.

        Uses the feature extractor model to extract features from the image.
        Estimates the probability of each class based on the features and the trained classifier.

        Parameters
        ----------
        image : np.ndarray or list[np.ndarray]
            Image to predict probabilities for or list of images
        in_channels : list[int], optional
            List of channels to use for prediction
        skip_norm : bool, optional
            Whether to skip normalization of the images before prediction.
            If True, the images are not normalized according to the parameter `normalize` in the model parameters.
        use_dask : bool, optional
            Whether to use dask for parallel processing

        Returns
        ----------
        probas : np.ndarray or list[np.ndarray]
            Predicted probabilities of the classes of the pixels in the image or list of images
            Dimensions are equal to the input image(s) without the channel dimension,
            with the class dimension added first
        """
        probas = self._predict(image, add_seg=False, in_channels=in_channels, skip_norm=skip_norm, use_dask=use_dask)
        return probas
    
    def get_feature_image(self, data, in_channels=None, skip_norm=False, pca_components=0, kmeans_clusters=0):
        """
        Returns the feature images extracted by the feature extractor model.
        For details, see the `_extract_features` method.
        
        Parameters
        ----------
        data : np.ndarray or list[np.ndarray]
            Image(s) to extract features from
        memory_mode : bool, optional
            Whether to use memory mode.
            If True, the annotations are registered and updated, and only features for new pixels are extracted.
        img_ids : str or list[str], optional
            Image IDs to register the annotations with (when using memory_mode)
        in_channels : list[int], optional
            List of channels to use for feature extraction
        skip_norm : bool, optional
            Whether to skip normalization of the images before feature extraction.
            If True, the images are not normalized according to the parameter `normalize` in the model parameters.

        Returns
        ----------
        features : np.ndarray or list[np.ndarray]
            Extracted features of the image(s) or list of features for each image if input is a list.
            Reshaped to the input imges' shapes. Features dimension is added first (FHW or FZHW).    
        """
        # Extract features
        features = self._extract_features(
                data,
                annotations=None,
                restore_input_form=True,
                memory_mode=False, # Only valid when using annotations
                img_ids=None, # Only needed when using memory_mode
                in_channels=in_channels,
                skip_norm=skip_norm,
                pca_components=pca_components,
                kmeans_clusters=kmeans_clusters
            )

        return features

    def _extract_features(self, data, annotations=None, restore_input_form=True,
                          memory_mode=False, img_ids=None,
                          in_channels=None, skip_norm=False,
                          pca_components=0, kmeans_clusters=0):
        """
        Returns the features of images extracted by the feature extractor model.

        Extract (and optionally reshape) feature images.

        Processing pipeline:

        1. Prepare dims -> list of [C,Z,H,W]
        2. Optional channel subset
        3. Optional normalization
        4. Down/upsample (param.image_downsample)
        5. Register/update annotations (memory_mode)
        6. Padding (if needed)
        7. (If annotations) extract planes / optionally tile
        8. FE extraction
        9. Optional reshaping: If restore_input_form & not annotations
        10. Restore original dimension semantics (drop singleton Z when appropriate)

        Returns either a single feature array or lists (and optionally annotations / coords when
        restore_input_form=False).

        Parameters
        ----------
        data : np.ndarray or list[np.ndarray]
            Image(s) to extract features from
        annotations : np.ndarray or list[np.ndarray], optional
            Annotations to extract features from.
            Image and annotation lists must correspond to each other.
        restore_input_form : bool, optional
            Whether to restore the input form of the features.
            If True, no annotations are returned.
            If False, features are returned in their original form if no annotations are given,
            or paired with the processed annotations if they are given.
        memory_mode : bool, optional
            Whether to use memory mode.
            If True, the annotations are registered and updated, and only features for new pixels are extracted.
        img_ids : str or list[str], optional
            Image IDs to register the annotations with (when using memory_mode)
        in_channels : list[int], optional
            List of channels to use for feature extraction
        skip_norm : bool, optional
            Whether to skip normalization of the images before feature extraction.
            If True, the images are not normalized according to the parameter `normalize` in the model parameters.

        Returns
        ----------
        features : np.ndarray or list[np.ndarray]
            Extracted features of the image(s) or list of features for each image if input is a list.
            Features dimension is added first.
        annotations : np.ndarray or list[np.ndarray], optional
            Processed annotations of the image(s) or list of images, if annotations are given.
        """

        # --- Basic bookkeeping ---------------------------------------------------
        # Check if we are processing any annotations
        use_annots   = annotations is not None
        # Check for single input
        single_input = hasattr(data, 'ndim') and data.ndim >= 2 and not isinstance(data, list)
        # Record original input shapes for reshaping and rescaling later
        input_shapes = [data.shape] if single_input else [d.shape for d in data]
        # Make sure img_ids are compatible and is made into a list
        both_single = isinstance(img_ids, (str, int)) and single_input
        if img_ids is not None and not both_single and len(img_ids) != len(data):
            raise ValueError("Image IDs must be passed as a list with the same length as the data (or a single string for a single image).")
        img_ids = [img_ids] if single_input else img_ids
        # Run checks on the model to ensure that it is ready to be used
        self._run_model_checks()
        # Make sure data is a list of images with [C, Z, H, W] shape
        data, annotations = self._prep_dims(data, annotations, get_coords=False)

        # --- Channel subset ------------------------------------------------------
        if in_channels is not None:
            self._check_in_channels(data, in_channels)
            data = [d[in_channels] for d in data]

        # --- Normalization -------------------------------------------------------
        if not skip_norm:
            data = [self._norm_single_image(d) for d in data]
        # if self.fe_model.norm_mode == "imagenet":
        #     data = [conv_paint_utils.normalize_image_imagenet(d) for d in data]

        # Record originals BEFORE any padding / resampling for reshaping and rescaling later
        self.original_shapes = [d.shape for d in data]  # list of (C,Z,H,W)

        # Adjust the parameters for feature extraction by enforcing parameters as defined in the FE
        params_for_extract = self._enforce_fe_params(self._param)

        # --- (Down/Up)sample -----------------------------------------------------
        upscale = params_for_extract.image_downsample < 1 # Register that we upscale if the parameter is negative
        factor  = params_for_extract.image_downsample
        factor  = (-factor) if upscale else factor # Ensure that the factor is positive
        data = [conv_paint_utils.scale_img(d, factor, input_type="img", upscale=upscale)
                for d in data]
        if use_annots:
            annotations = [conv_paint_utils.scale_img(a, factor, input_type="labels", upscale=upscale)
                        for a in annotations]

        # --- Memory mode: annotation registering & updating ---------------------------------------
        if memory_mode:
            annotations = self._register_and_update_annots(
                annotations, img_ids, params_for_extract.image_downsample
            )
            num_new = np.sum([ann > 0 for ann in annotations])
            if num_new == 0:
                warnings.warn("No new annotations. Train with existing features.")
                return [], [], [], [], params_for_extract.image_downsample
            coords = [conv_paint_utils.get_coordinates_image(d) for d in data]
        else:
            coords = [None for _ in data]  # No coordinates if not in memory mode

        # --- Padding ----------------------------
        # Record originals after resampling but BEFORE padding for reshaping and rescaling later
        self.pre_pad_shapes = [d.shape for d in data]  # list of (C,Z,H,W)
        # Get the correct padding size for the feature extraction
        paddings = [self._get_overall_paddings(params_for_extract, d.shape) for d in data]
        # Pad the images and annotations with the correct padding size
        data = [conv_paint_utils.pad(d, p, input_type="img") for d, p in zip(data, paddings)]
        # Save the padded shapes for reshaping/rescaling later
        self.padded_shapes = [d.shape for d in data]
        if use_annots:
            annotations = [conv_paint_utils.pad(a, p, input_type="labels") for a, p in zip(annotations, paddings)]
        if memory_mode:
            coords = [conv_paint_utils.pad(c, p, input_type="coords") for c, p in zip(coords, paddings)]
        else:
            coords = [None for _ in data]  # No coordinates if not in memory mode

        # --- Annotation plane extraction -----------------------------------------
        # Note: annotated planes are flattened (treating them as individual images)
        if use_annots:
            planes = [conv_paint_utils.get_annot_planes(d, a, c)
                    for d, a, c in zip(data, annotations, coords)]
            data        = [p for trio in planes for p in trio[0]]
            annotations = [p for trio in planes for p in trio[1]]
            # Flat-repeat paddings for each plane
            paddings    = [paddings[i]                      # the padding for that image
                           for i, trio in enumerate(planes) # for each original image
                           for _ in range(len(trio[0]))]    # repeat for each plane in that image
            if memory_mode:
                coords = [p for trio in planes for p in trio[2]]
                # If img_ids are given, flat-repeat them as well (corresponding id for each plane)
                if img_ids is not None:
                    img_ids = [img_ids[i]
                            for i, trio in enumerate(planes)
                            for _ in range(len(trio[0]))]
                else:
                    img_ids = [None] * len(data)
            else:
                coords = [None for _ in data]
        else:
            coords = [None for _ in data]

        # --- Annotation tiling (optional) ----------------------------------------
        # Note: tiles are flattened (treating them as individual images)
        if params_for_extract.tile_annotations:
            if use_annots:
                coords = [None for _ in data] if coords is None else coords
                tiles = [conv_paint_utils.tile_annot(d, a, c, p, plot_tiles=False)
                        for d, a, c, p in zip(data, annotations, coords, paddings)]
                data        = [t for trio in tiles for t in trio[0]]
                annotations = [t for trio in tiles for t in trio[1]]
                # Flat-repeat paddings for each tile (though not needed anymore, in case they are used later)
                paddings    = [paddings[i]                      # the padding for that image
                               for i, trio in enumerate(tiles)  # for each original image
                               for _ in range(len(trio[0]))]    # repeat for each tile in that image
                if memory_mode:
                    coords = [t for trio in tiles for t in trio[2]]
                    # If img_ids are given, flatten them as well (corresponding id for each tile)
                    if img_ids is not None:
                        img_ids = [img_ids[i]
                                for i, trio in enumerate(tiles)
                                for _ in range(len(trio[0]))]
                    else:
                        img_ids = [None] * len(data)
                else:
                    coords = [None for _ in data]
            else:
                coords = [None for _ in data]

        # --- Feature extraction --------------------------------------------------
        # Note: For training, we want to "unpatch" the features to match the annotations
        # Note: For prediction, we want to keep the features patched if the model supports it
        keep_patched = (not use_annots) and self.fe_model.gives_patched_features()
        features = [self.fe_model.get_feature_pyramid(d, params_for_extract, patched=keep_patched)
                    for d in data]
        
        if pca_components:
            features = [conv_paint_utils.apply_pca_to_f_image(f, n_components=pca_components)
                        for f in features]

        # --- Raw early return ----------------------------------------------------
        if not restore_input_form:
            if memory_mode:
                return features, annotations, coords, img_ids, params_for_extract.image_downsample
            if use_annots:
                return features, annotations
            return features

        # Apply Kmeans clustering if requested (afterwards treat just like a segmentation)
        class_output = False # Whether the output is a class prediction (change if we do Kmeans)
        if kmeans_clusters: # NOTE: random_state -> fix for reproducibility; OR None for random
            features = [conv_paint_utils.apply_kmeans_to_f_image(f, n_clusters=kmeans_clusters, random_state=0)
                        for f in features]
            class_output = True

        # --- Reshape back to original spatial form (for display) ---------
        padded_shapes   = self.padded_shapes # After both down/upsampling and padding
        pre_pad_shapes  = self.pre_pad_shapes # After down/upsampling, before padding
        original_shapes = self.original_shapes # Before down/upsampling and padding

        # Only original (non-plane) images are expected here
        features = [
            self._restore_shape(
                features[i],
                padded_shapes[i],
                pre_pad_shapes[i],
                original_shapes[i],
                class_preds=class_output,
                patched_features=self.fe_model.gives_patched_features()
            )
            for i in range(len(features))
        ]

        # Smoothen the Kmeans output if desired
        if kmeans_clusters and self._param.seg_smoothening > 1:
            kernel = skimage.morphology.disk(self._param.seg_smoothening)
            if features[0].ndim == 3: # 3D images (all images need to have same dimensionality)
                kernel = np.expand_dims(kernel, axis=0) # --> add z dimension to kernel
            features = [skimage.filters.rank.majority(f, footprint=kernel) for f in features]

        # Final dimension restoration (remove singleton Z if needed)
        features = [self._restore_dims(features[i], input_shapes[i])
                    for i in range(len(features))]

        if single_input:
            return features[0]
        return features

### CLASSIFIER METHODS

    def _clf_train(self, features, targets, use_rf=False, allow_writing_files=False):
        """
        Trains a classifier given a set of features and targets.

        If use_rf is False, a CatBoostClassifier is trained, otherwise a RandomForestClassifier.
        The trained classifier is saved in the model, but also returned.

        When using CatBoost, the model is trained on the GPU if specified in the clf_use_gpu parameter.
        If this parameter is not specified, the model will infer GPU usage from the fe_use_gpu parameter.
        If this is not specified either, or RandomForest is used, the model will be trained on the CPU.

        Parameters
        ----------
        features : np.ndarray
            Features to train the classifier on
        targets : np.ndarray
            Targets to train the classifier on
        use_rf : bool, optional
            Whether to use a RandomForestClassifier instead of a CatBoostClassifier
        allow_writing_files : bool, optional
            Allow writing files for the CatBoostClassifier

        Returns
        ----------
        clf : CatBoostClassifier or RandomForestClassifier
            Trained classifier (also saved in the model instance)
        """
        if not use_rf:
            # NOTE: THIS, FOR NOW, ASSUMES THAT GPU SHALL BE USED FOR CLF IF CHOSEN FOR FE
            use_gpu = self._param.clf_use_gpu if self._param.clf_use_gpu is not None else self._param.fe_use_gpu
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
        self.num_features = features.shape[1] if isinstance(features, np.ndarray) else features[0].shape[1] if isinstance(features, list) else None
        return clf
    
    def _clf_predict(self, features, return_proba=True):
        """
        Uses the trained classifier to predict classes based on features of the image.
        Returns an image containing the probabilities of each class for each pixel.
        If return_proba is False, the predicted class is returned instead.

        Parameters
        ----------
        features : np.ndarray
            Features to use for the prediction; features dimension is expected first
        return_proba : bool, optional
            Whether to return the prediction as class probabilities

        Returns
        ----------
        predictions : np.ndarray
            Predicted classes or probabilities of the classes for each pixel in the image.
            Pixel dimension is linearized (width*height),
            class dimension is added first (if return_proba is True).
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

        - Resets the classifier object to None
        - Resets possible path to the classifier saved in the Param object
        - Resets the training features used with memory_mode
        """
        self.classifier = None
        self._param.classifier = None
        self.reset_training()


### BACKEND TRAINING AND PREDICTION METHODS

    def _train(self, data, annotations, memory_mode=False, img_ids=None, use_rf=False, allow_writing_files=False, in_channels=None, skip_norm=False):
        """
        Backend training method for the Convpaint model.
        """

        # Sanity check if we even got any annotations
        if annotations is None or all(a is None for a in annotations) or all(np.sum(a) == 0 for a in annotations):
            warnings.warn('No annotations provided for training.')
            return self.classifier, None, None

        if not memory_mode:
            # Use _extract_features to extract features and the suiting annotation parts (returns lists if restore_input_form=False)
            feature_parts, annot_parts = self._extract_features(data, annotations, restore_input_form=False, memory_mode=memory_mode, in_channels=in_channels, skip_norm=skip_norm)
            # Get the annotated pixels and targets, and concatenate each
            f_t_tuples = [conv_paint_utils.get_features_targets(f, a)
                        for f, a in zip(feature_parts, annot_parts)] # f and t are linearized
            features = [ft[0] for ft in f_t_tuples] # Create a list of features
            targets = [ft[1] for ft in f_t_tuples] # Create a list of targets
            features = np.concatenate(features, axis=0) # Concatenate the features into a single array
            targets = np.concatenate(targets, axis=0) # Concatenate the targets into a single array

        else: # memory mode
            # Use _extract_features to extract features and the suiting annotation parts (returns lists if restore_input_form=False)
            feature_parts, annot_parts, coords, img_ids, scale = self._extract_features(data, annotations, restore_input_form=False, memory_mode=memory_mode, img_ids=img_ids, in_channels=in_channels, skip_norm=skip_norm)
            # Get all annotations and features from the table
            features, targets = self._register_and_get_all_features_annots(feature_parts, annot_parts, coords, img_ids, scale)

        # Check if we have features and targets, and that we have at least two classes
        if len(features) == 0 or len(targets) == 0:
            raise ValueError('No features or targets found. Please check the input data and annotations.')
        if len(np.unique(targets)) < 2:
            # print("Targets:", targets, "values:", np.unique(targets))
            raise ValueError('Not enough classes found in the targets. At least two classes are required for training.')

        # Train the classifier
        self._clf_train(features, targets, use_rf=use_rf, allow_writing_files=allow_writing_files)

        return self.classifier, feature_parts, annot_parts

    def reset_training(self):
        """
        Resets the features and targets of the model which are used to
        iteratively train the model with the option memory_mode.
        """
        self.table = pd.DataFrame(columns=['img_id', 'scale', 'coordinates', 'label', 'features_array'])
        self.table.index.name = "row_key"
        self.annot_dict = {}
        self.num_trainings = 0

    def _register_and_update_annots(self, annotations, img_ids, scale=1):
        """
        Registers new annotations in the table and deletes existing ones from the new annotations.

        Parameters
        ----------
        annotations : list[np.ndarray]
            List of annotations to register, each with shape [C, Z, H, W]
        img_ids : list[str]
            List of image IDs corresponding to the annotations, each as a string
        scale : int, optional
            Scale factor, used for creating the unique table entries

        Returns
        ----------
        all_new_annots : list[np.ndarray]
            List of new annotations with the same shape as the input annotations,
            where existing annotations are set to zero in the new annotations.
        """
        # Allow for single inputs and in that case make both inputs lists
        if isinstance(annotations, np.ndarray):
            annotations = [annotations]
        if isinstance(img_ids, str):
            img_ids = [img_ids]
        if not len(annotations) == len(img_ids):
            raise ValueError('Annotations and image IDs must have the same length.')

        # First update the table with the new annotations
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
    
    def _register_and_get_all_features_annots(self, features, annotations, coords, img_ids, scale=1):
        """
        Registers new features in the table and returns all features and annotations.

        IMPORTANT: This assumes that double entries have been removed from the annotations and old entries from the table.

        Parameters
        ----------
        features : list[np.ndarray]
            List of features to register, each with shape [C, Z, H, W]
        annotations : list[np.ndarray]
            List of annotations to register, each with shape [C, Z, H, W]
        coords : list[np.ndarray]
            List of coordinates corresponding to the annotations, each with shape [C, Z, H, W]
        img_ids : list[str]
            List of image IDs corresponding to the annotations, each as a string
        scale : int
            Scale factor, used for creating the unique table entries
        
        Returns
        ----------
        features : list[np.ndarray]
            List of all features with the same shape as the input features,
            where existing features were kept and new features were added.
        annotations : list[np.ndarray]
            List of all annotations with the same shape as the input annotations,
            where existing annotations were kept and new annotations were added.
        """
        # Allow for single inputs and in that case make both inputs lists
        if isinstance(features, np.ndarray):
            features = [features]
        if isinstance(annotations, np.ndarray):
            annotations = [annotations]
        if isinstance(coords, np.ndarray):
            coords = [coords]
        if isinstance(img_ids, str):
            img_ids = [img_ids]
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
        features_list = list(self.table['features_array'].values)
        annotations_list = list(self.table['label'].values)

        # Convert features to 2D array: [num_entries, num_features]
        features = np.stack(features_list, axis=0) if len(features_list) > 0 else np.empty((0, self.num_features))
        # Convert annotations to 1D array
        annotations = np.array(annotations_list)

        return features, annotations

    def _predict(self, data, add_seg=False, in_channels=None, skip_norm=False, use_dask=False):
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

        # Check if we have only a single image input
        single_input = isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)
        input_shapes = [data.shape] if single_input else [d.shape for d in data]

        # Make sure data is a list of images with [C, Z, H, W] shape
        data, _ = self._prep_dims(data)

        # If in_channels is given, extract the channels selected by in_channels from the data
        if in_channels is not None:
            self._check_in_channels(data, in_channels)
            data = [d[in_channels] for d in data]

        # If not done previously, normalize the images (separately and according to the parameter)
        if not skip_norm:
            data = [self._norm_single_image(d) for d in data]

        # Get class probabilities, using tiling if enabled
        if self._param.tile_image:
            probas = [self._parallel_predict_image(d, return_proba=True, use_dask=use_dask)
                      for d in data]
        else:
            probas = self._predict_image(data, return_proba=True) # Can handle lists directly

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

    def _predict_image(self, image, return_proba=True, feature_img=None):
        """
        Backend method to predict images without tiling and parallelization.
        Returns the class probabilities and optionally the segmentation of the images.

        Reshapes the predictions to the original image shape, without the channel dimension,
        but always keeps the Z dim --> [C, Z, H, W] for class probas, [Z, H, W] for segmentation.

        A feature image can be given, in which case the features are used for prediction.
        If no feature image is given, the features are extracted from the image using the feature extractor.

        NOTE: This allows to use this method interchangably with _parallel_predict_image,
              but also within the _parallel_predict_image method for each tile.
        """

        # Check if features are given, if not, extract them
        if feature_img is None:
            feature_img = self._extract_features(image,
                                            restore_input_form=False,
                                            in_channels=None, # already extracted outside
                                            skip_norm=True)  # already normalized outside

        num_f = feature_img[0].shape[0] if isinstance(feature_img, list) else feature_img.shape[0]
        num_f_clf = self.num_features
        if num_f != num_f_clf:
            raise ValueError(
                f"Extracted features ({num_f}) have a different length than classifier expects ({num_f_clf}). "
                "(channel count mismatch / changed fe_scalings or channels)."
            )
        # Predict pixels based on the features and classifier
        # NOTE: We always first predict probabilities and then take the argmax
        feature_img = feature_img if isinstance(feature_img, list) else [feature_img]
        predictions = [self._clf_predict(f, return_proba=True) for f in feature_img]
        # Reshape the predictions to the original image shape
        padded_shapes = self.padded_shapes # Saved when extracting features
        pre_pad_shapes = self.pre_pad_shapes # Saved when extracting features
        original_shapes = self.original_shapes # Saved when extracting features
        pred_reshaped = [self._restore_shape(predictions[i],
                                            padded_shapes[i],
                                            pre_pad_shapes[i],
                                            original_shapes[i],
                                            class_preds=False,
                                            patched_features=self.fe_model.gives_patched_features())
                         for i in range(len(predictions))]

        # If we want classes, take the argmax of the probabilities
        if not return_proba:
            pred_reshaped = [self._probas_to_classes(p) for p in pred_reshaped]
        # If input was a single image, return the first prediction
        if isinstance(image, (np.ndarray, torch.Tensor)):
            return pred_reshaped[0]
        return pred_reshaped

    def _parallel_predict_image(self, image, return_proba=True, use_dask=False):
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

                # Predict the block using dask or directly (with no normalization, as it is done outside)
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

    def _train_predict_image(self, image, annotations, use_rf=False, allow_writing_files=False, in_channels=None, skip_norm=False, add_seg=False):
        """
        Extracts features from the image and uses them to both train and predict the image.

        If multiple planes are given, only the annotated planes will be processed and predicted.

        Currently does not support image tiling and memory mode.
        """

        # Find lanes/slices containing annotations
        annot_slice_mask = np.any(annotations > 0, axis=(-2, -1))
        # Extract the annotated image slices
        if annotations.ndim > 2:
            annot_img = image[..., annot_slice_mask, :, :]
            annot_annot = annotations[..., annot_slice_mask, :, :]
            assert annot_img.shape[-3:] == annot_annot.shape, "Annotated image and annotations must have the same (spatial) shape."
        else:
            annot_img = image
            annot_annot = annotations

        # Save the old parameters, so we can change it specifically for this method
        old_params = self._param.copy()
        # Turn off the tiling of annotations, since we want to extract the entire image to be re-used for prediction
        self._param.set_single('tile_annotations', False)

        # Train, and get the features at the same time
        # Note: This is a hack to use the training method for prediction, but it works
        _, feature_parts, _ = self._train( # feature_parts corresponds to the annotated feature image slices (since no tiling)
            data=annot_img, annotations=annot_annot, memory_mode=False,
            use_rf=use_rf, allow_writing_files=allow_writing_files, in_channels=in_channels, skip_norm=skip_norm
        )
        feature_img = np.concatenate(feature_parts, axis=1)

        # Prediction expects a list of patched features, but training extracts per-pixel
        if self.fe_model.gives_patched_features():
            p = self.fe_model.get_patch_size()
            feature_img = [feature_img[..., ::p, ::p]]
        # Predict the image using the features extracted
        # Note: This is a hack to use the prediction method for prediction, but it works
        probas = self._predict_image(
            image, return_proba=True, feature_img=feature_img)
        
        # Create a probability image with the original shape of the image and the results in the annotated slices
        if annotations.ndim > 2:
            probas_img = np.zeros((probas.shape[0],) + annotations.shape, dtype=probas.dtype)
            probas_img[..., annot_slice_mask, :, :] = probas
        else:
            probas_img = probas

        # Restore input dimensionality (especially see if we want to remove z dimension)
        probas_img = self._restore_dims(probas_img, image.shape)

        # Reset the parameters to the old ones
        self._param = old_params

        # Determine what to return
        if add_seg:
            seg = self._probas_to_classes(probas_img)
            return probas_img, seg
        else:
            return probas_img

### HELPER METHODS

    def _run_model_checks(self):
        """
        Runs checks on the model to ensure that it is ready to be used.
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
        # Use hasattr(data, 'ndim') to support array-like objects beyond numpy/torch (e.g. dask, zarr)
        if isinstance(data, list):
            img_list = data
        elif isinstance(data, tuple):
            img_list = list(data)
        elif hasattr(data, 'ndim'):
            img_list = [data]
        else:
            raise ValueError('Data must be an array-like (numpy, torch, dask, etc.) or a list/tuple of images.')
        if annotations is None:
            annot_list = [None] * len(img_list)
        elif isinstance(data, list):
            annot_list = annotations
        elif isinstance(data, tuple):
            annot_list = list(annotations)
        elif hasattr(data, 'ndim'):
            annot_list = [annotations]
        else:
            raise ValueError('Annotations must be an array-like or a list/tuple of images.')

        # Check if the lengths of the data and annotations lists are equal
        if len(img_list) != len(annot_list):
            raise ValueError('Data and annotations lists must have the same length.')
        
        # Ensure annotations are integer values
        for i, annot in enumerate(annot_list):
            if annot is not None and np.issubdtype(annot.dtype, np.integer) == False:
                warnings.warn(f'Annotations for image {i} are not of type int. Converting to int32.')
                annot_list[i] = annot.astype(np.int32)

        # Check and preprocess the dimensions of each data and annotations pairs
        prep_imgs, prep_annots = zip(*[
            self._prep_dims_single(item, annot)
            for item, annot in zip(img_list, annot_list)
        ])

        # Check if the dimension type of all images is the same (same num of dims and channels)
        for i, img in enumerate(prep_imgs):
            if img.ndim != prep_imgs[0].ndim:
                raise ValueError(f'Image {i} has different number of dimensions to the first image.')
            if img.shape[0] != prep_imgs[0].shape[0]:
                raise ValueError(f'Image {i} has different number of channels to the first image.')

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
        Preprocesses the dimensions of the img and annotations (if given).

        Ensures the data and annotations have the correct number of dimensions,
        where the data is a list of images with [C, Z, H, W] shape,
        and the annotations are a list of images with [Z, H, W] shape.

        Assumes that a possible channels dimension is the first one.
        
        Returns img, annotations (= None if not given)
        """
        num_dims = img.ndim
        if num_dims == 2:
            if self._param.channel_mode in ['rgb', 'multi']:
                warnings.warn(f'Image has only 2 dimensions, but the parameter channel_mode is {self._param.channel_mode}. ' +
                              'Assuming that we are working with single channel image.')
                self._param.channel_mode = "single"

            # Add a channels and a z dimension (with size = 1)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)
        elif num_dims == 3:
            if self._param.channel_mode in ['rgb', 'multi']:
                # Add a z dimension at the second position
                img = np.expand_dims(img, axis=1)
            else: # single channel
                # Add a channels dimension at the first position
                img = np.expand_dims(img, axis=0)
        elif num_dims == 4:
            if img.shape[0] > 1 and self._param.channel_mode == "single":
                warnings.warn(f'Image has {img.shape[0]} channels, but channel_mode is set to "single". ' +
                                'Convpaint only works with 4D data as 3D multi-channel (or RGB) images with [C, Z, H, W]. ' +
                                'Setting channel_mode to "multi".')
                self._param.channel_mode = "multi"
            pass  # Image is already in [C, Z, H, W] format
        else:
            raise ValueError(f'Image has wrong number of dimensions: {num_dims}')

        # Check that RGB is only used for 3-channel images
        if self._param.channel_mode == 'rgb' and img.shape[0] != 3:
            warnings.warn(f'Image has {img.shape[0]} channels, but channel_mode is "rgb". ' +
                          'Assuming non-RGB input.')
            if img.shape[0] == 1:
                self._param.channel_mode = "single"
            else:
                self._param.channel_mode = "multi"

        # Early return if no annotations are given
        if annotations is None:
            return img, None

        # ANNOTATIONS
        num_dims_annots = annotations.ndim
        if num_dims_annots == 2:
            # Add a z dimension at the first position
            annotations = np.expand_dims(annotations, axis=0)
        elif num_dims_annots == 3:
            # Annotations are already in [Z, H, W] format
            pass
        else:
            raise ValueError(f'Annotations have wrong number of dimensions ({num_dims_annots})')

        if img.shape[1:] != annotations.shape:
            raise ValueError(f'Image and annotations have different (non-channel) dimensions: {img.shape[1:]} vs {annotations.shape}')

        return img, annotations
    
    def _check_in_channels(self, data, in_channels=None):
        """
        Checks if the conditions for using the given in_channels are met, and raise an error if not.
        Assumes that the channels dimension is the first one.
        """
        if in_channels is None:
            print("No in_channels given for _check_in_channels.")
            return
        
        # In channels makes only sense on multi-channel images
        if self._param.channel_mode == 'single':
            raise ValueError("in_channels is not valid for single channel images (defined in the model parameters). " +
                                "Please either remove in_channels or make sure the channel_mode is 'multi' or 'rgb'.")
        
        # Check that in_channels is a list of integers
        if not isinstance(in_channels, list):
            raise ValueError("in_channels must be a list of integers. Please provide a list of channel indices to use.")
        if not all(isinstance(ch, int) for ch in in_channels):
            raise ValueError("in_channels must be a list of integers. Please provide a list of channel indices to use.")

        # Check that all in_channels are valid channel indices
        channels_in_data = data[0].shape[0] if isinstance(data, list) else data.shape[0]
        if not all(0 <= ch < channels_in_data for ch in in_channels):
            raise ValueError("All in_channels must be valid channel indices. Please adjust in_channels to be within the range of channels in the data.")
    
    def _norm_single_image(self, img):
        """
        Checks the norm mode and normalizes a single image accordingly.
        For default mode, the image stats are computed according to the scope given in the parameters.
        Assumes that the image is in [C, Z, H, W] format.
        """
        
        # If normalization scope is 1 or not specified, no normalization is applied
        if self._param.normalize is None or self._param.normalize == 1:
            return img

        # Get the normalization scope from the parameters
        norm_scope = self._param.normalize

        # Check that the image has 4 dimensions
        if img.ndim != 4:
            raise ValueError('Image must be given as 4 dimensions (C, Z, H, W).')

        # FE-SPECIFIC NORMALIZATION
        # If the FE demands imagenet or percentile normalization, do that if image is compatible
        fe_norm = self.fe_model.norm_mode
        if fe_norm == 'imagenet':
            if self._param.channel_mode == 'rgb':
                img_norm = conv_paint_utils.normalize_image_imagenet(img)
                return img_norm
            else:
                print("FE model is designed for imagenet normalization, but image is not declared as 'rgb' (parameter channel_mode). " +
                "Using default normalization instead.")
        elif fe_norm == 'percentile':
            num_ignored_dims = 1 if norm_scope == 2 else 2 # Ignore C dim for stack, C and Z for by image
            img_norm = conv_paint_utils.normalize_image_percentile(img, ignore_n_first_dims=num_ignored_dims)
            return img_norm

        # DEFAULT NORMALIZATION
        # Otherwise we normalize to mean and std, generally normalizing each channel separately
        # This means that if we normalize by stack (= 2), we ignore the C dimension (= first) only
        # If normalization scope is "by image" (= 3), we additionally keep the Z dimension (= second)
        num_ignored_dims = 1 if norm_scope == 2 else 2 # Ignore C dim for stack, C and Z for by image
        mean, sd = conv_paint_utils.compute_image_stats(img, ignore_n_first_dims=num_ignored_dims)
        
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
        enforced_params = self.fe_model.get_enforced_params(old_param)
        new_param = old_param.copy()
        # Raise a warning if anything is enforced
        for key in enforced_params.get_keys():
            if (enforced_params.get(key) is not None and
                old_param.get(key) != enforced_params.get(key)):
                model_name = self.fe_model.get_name()
                warnings.warn(f'Parameter {key} is enforced by the feature extractor model ({model_name}).')
                new_param.set_single(key, enforced_params.get(key))
        return new_param
    
    def _get_overall_paddings(self, param, img_shape: Tuple[int, ...]):
        """
        Returns the overall padding sizes for the image in form ((top, bottom), (left, right)).

        This takes into account the feature extractor's padding, its patch size and the image shape.
        Makes sure, padding is at least the FE's padding at the largest downscaling factor,
        and that the image is padded to a multiple of the patch size times at all scalings,
        if the FE model uses patches.
        """
        # Get the maximum scaling factor, base padding size and patch_size from the param and feature extractor
        fe_scalings = param.fe_scalings
        max_scale = np.max(fe_scalings)
        fe_pad   = self.fe_model.get_padding()
        fe_patch = self.fe_model.get_patch_size()

        # We need to pad at least the feature extractor's padding at the maximum scaling factor on each side
        min_pad = fe_pad * max_scale
        min_h = img_shape[-2] + 2 * min_pad
        min_w = img_shape[-1] + 2 * min_pad

        # Calculate the least common multiple (LCM) of the feature extractor's scalings
        scalings_lcm = lcm(*fe_scalings)
        # We need to pad to a multiple of the patch size at the lcm scaling factor (as it will be downscaled accordingly)
        patch_multiple = scalings_lcm * fe_patch

        # Calculate the padding sizes for each dimension
        padded_h = ( (min_h + patch_multiple - 1) // patch_multiple ) * patch_multiple
        padded_w = ( (min_w + patch_multiple - 1) // patch_multiple ) * patch_multiple
        pad_h = padded_h - img_shape[-2]
        pad_w = padded_w - img_shape[-1]

        # Ensure that the padding is at least the minimum padding on each side, and a multiple of the lcm-scaled patch size
        assert pad_h >= 2 * min_pad, f"Padding height {pad_h} is less than minimum padding 2x{min_pad}."
        assert pad_w >= 2 * min_pad, f"Padding width {pad_w} is less than minimum padding 2x{min_pad}."
        assert padded_h % patch_multiple == 0, f"Padded height {padded_h} is not a multiple of patch size {patch_multiple}."
        assert padded_w % patch_multiple == 0, f"Padded width {padded_w} is not a multiple of patch size {patch_multiple}."

        # Distribute to left/right and top/bottom, the bottom and right being 1 larger in uneven cases
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Return the overall padding sizes for the image
        return (pad_top, pad_bottom), (pad_left, pad_right)

    def _restore_shape(self, outputs: np.ndarray, 
                             padded_shape: Tuple[int, ...],
                             pre_pad_shapes: Tuple[int, ...],
                             original_shape: Tuple[int, ...],
                             class_preds: bool = False, patched_features: bool = False):
        """
        Reshapes outputs (prediction, probabilities, features) back to original spatial size,
        removing the padding and rescaling.

        Parameters
        ----------
        outputs : np.ndarray
            Flattened or semi‐flattened output.
        padded_shape : tuple
            Shape after all padding and rescaling: (..., Z, H_proc, W_proc)
        pre_pad_shapes : tuple
            The shape before padding, after rescaling: (..., Z, H_padded, W_padded)
        original_shape : tuple
            The original shape before any padding and rescaling: (..., Z, H_orig, W_orig)
        class_preds : bool
            If True, we’re dealing with integer class labels (without additional dimension).
        patched_features : bool
            If True, the features were extracted on a patch‐size resolution (e.g. DINOv2),
            so we need to rescale the outputs to the patch resolution before upsampling.

        Returns
        ----------
        outputs_image : np.ndarray
            The reshaped and upsampled output image.
            Form [Z, H, W] for class predictions, [C, Z, H, W] for other outputs.
        """
        # Prepare the variables
        padded_z, padded_h, padded_w = padded_shape[-3:]

        # 1) Reshape to (possibly patched) spatial dimensions
        # If FE was patched, outputs are on a coarser grid
        if patched_features:
            p = self.fe_model.get_patch_size()
            patch_h = padded_h // p
            patch_w = padded_w // p
        else:
            patch_h, patch_w = padded_h, padded_w
        if class_preds:
            outputs_image = np.reshape(outputs, [padded_z, patch_h, patch_w])
        else:
            num_outputs = outputs.shape[0]
            outputs_image = np.reshape(outputs, [num_outputs, padded_z, patch_h, patch_w])

        # 2) Upsample from grid to pixel resolution (but still downsampled and padded)
        # Note: This does nothing if the image is already at pixel resolution (= not "patched")
        if patched_features:
            patch_multi_h = patch_h * p
            patch_multi_w = patch_w * p
        else:
            patch_multi_h, patch_multi_w = padded_h, padded_w
        if class_preds:
            new_shape = (padded_z, patch_multi_h, patch_multi_w)
            outputs_image = conv_paint_utils.rescale_class_labels(
                outputs_image, output_shape=new_shape)
        else:
            new_shape = (num_outputs, padded_z, patch_multi_h, patch_multi_w)
            order = self._param.unpatch_order
            outputs_image = conv_paint_utils.rescale_outputs(
                outputs_image, output_shape=new_shape,
                order=order)

        # 3) Remove padding
        # Ensure to only remove the part of padding that was not removed by reduce_to_patch_multiple in the FE model
        outputs_image = conv_paint_utils.crop_to_shape(
            outputs_image,
            pre_pad_shapes[-3:]
        )

        # 4) Resize to original shape
        orig_z, orig_h, orig_w = original_shape[-3:]
        if self._param.image_downsample not in (-1, 0, 1):
            if class_preds:
                outputs_image = conv_paint_utils.rescale_class_labels(
                    outputs_image,
                    output_shape=(orig_z, orig_h, orig_w)
                )
            else:
                new_shape = (num_outputs, orig_z, orig_h, orig_w)
                outputs_image = conv_paint_utils.rescale_outputs(
                    outputs_image,
                    output_shape=new_shape,
                    order=1
                )

        return outputs_image

    def _probas_to_classes(self, probas):
        """
        Converts probabilities to classes. Smoothen output if requested in parameters.
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
        Restores the dimensions of the prediction to the original shape.
        """
        is_2d = len(original_shape) == 2
        is_3d = len(original_shape) == 3
        is_3d_multi = is_3d and (self._param.channel_mode in ['rgb', 'multi'])
        # if is 2d or 3D with multiple channels (including RGB), we remove the z dimension
        if is_2d or is_3d_multi:
            if pred.shape[-3] != 1:
                raise ValueError('Prediction shape does not match original shape.')
            # [Z,H,W] for prediction, [proba/F,Z,H,W] for probabilities and features
            pred = np.squeeze(pred, axis=-3)
        return pred