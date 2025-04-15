
import pickle
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
import pandas as pd
import skimage
import torch

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

    Parameters
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

    ALL_MODELS_TYPES_DICT = {}

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

        The constructor can be initialized in three ways:
        1. By providing an alias (e.g., "vgg-s", "dino", "gaussian", etc.), in which case a corresponding model
        configuration will be loaded.
        2. By providing a saved model path (model_path) to load a pre-trained model.
        3. By providing a Param object, which contains model parameters.
        4. By providing the name of the feature extractor, CUDA usage, and feature extraction layers, in which case
        the additional kwargs will be used to override the defaults of the feature extractor model.
        
        If none of the options are provided, a default Convpaint model will be created.

        Parameters
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
        **kwargs : additional parameters
            Additional parameters to override defaults for the model or feature extractor.

        Raises
        ------
        ValueError
            If more than one of alias, model_path, param, or fe_name is provided.
        """
        # Initialize parameters and features
        self._param = Param()
        self.features = None
        self.targets = None
        self.num_trainings = 0

        # Initialize the dictionary of all available models
        if not ConvpaintModel.ALL_MODELS_TYPES_DICT:
            ConvpaintModel._init_models_dict()

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
    def _init_models_dict():
        """
        Initializes the dictionary of all available feature extractor models.
        """
        # Initialize the MODELS TO TYPES dictionary with the models that are always available
        ConvpaintModel.ALL_MODELS_TYPES_DICT = {x: Hookmodel for x in NN_MODELS}
        ConvpaintModel.ALL_MODELS_TYPES_DICT.update({x: GaussianFeatures for x in GAUSSIAN_MODELS})
        ConvpaintModel.ALL_MODELS_TYPES_DICT.update({x: DinoFeatures for x in DINO_MODELS})
        ConvpaintModel.ALL_MODELS_TYPES_DICT.update({x: ComboFeatures for x in COMBO_MODELS})

        # Try to import CellposeFeatures and update the MODELS TO TYPES  dictionary if successful
        # Cellpose is only installed with pip install napari-convpaint[cellpose]
        try:
            from napari_convpaint.conv_paint_cellpose import AVAILABLE_MODELS as CELLPOSE_MODELS
            from napari_convpaint.conv_paint_cellpose import CellposeFeatures
            ConvpaintModel.ALL_MODELS_TYPES_DICT.update({x: CellposeFeatures for x in CELLPOSE_MODELS})
        except ImportError:
            # Handle the case where CellposeFeatures or its dependencies are not available
            print("Info: Cellpose is not installed and is not available as feature extractor.\n"
                "Run 'pip install napari-convpaint[cellpose]' to install it.")
        # Same for ilastik
        try:
            from napari_convpaint.conv_paint_ilastik import AVAILABLE_MODELS as ILASTIK_MODELS
            from napari_convpaint.conv_paint_ilastik import IlastikFeatures
            ConvpaintModel.ALL_MODELS_TYPES_DICT.update({x: IlastikFeatures for x in ILASTIK_MODELS})
        except ImportError:
            # Handle the case where IlastikFeatures or its dependencies are not available
            print("Info: Ilastik is not installed and is not available as feature extractor.\n"
                "Run 'pip install napari-convpaint[ilastik]' to install it.\n"
                "Make sure to also have fastfilters installed ('conda install -c ilastik-forge fastfilters').")

    @staticmethod
    def get_all_fe_models():
        """
        Returns a dictionary of all available feature extractor models
        """
        return ConvpaintModel.ALL_MODELS_TYPES_DICT
    
    @staticmethod
    def get_default_params():
        """
        Returns a param object, which defines the default Convpaint model.
        """

        def_param = Param()

        # Image processing parameters
        def_param.multi_channel_img = False
        def_param.rgb_img = False
        def_param.normalize = 2  # 1: no normalization, 2: normalize stack, 3: normalize each image

        # Acceleration parameters
        def_param.image_downsample = 1
        def_param.seg_smoothening = 1
        def_param.tile_annotations = True
        def_param.tile_image = False

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
        """
        return self._param.get(key)
    
    def get_params(self):
        """
        Returns all parameters of the model (= a copy of the Param object).
        """
        return self._param.copy()
    
    def set_param(self, key, val, no_warning=False):
        """
        Sets the value of the given parameter key.
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
        Sets the parameters if given. Note that the model is not reset and no FE model is created.
        If fe_name, fe_use_cuda, and fe_layers changes, you should create a new ConvpaintModel.
        """
        if param is not None:
            kwargs = param.__dict__
        for key, val in kwargs.items():
            if val is not None:
                self.set_param(key, val, no_warning=no_warning)

    def save(self, model_path, create_pkl=True, create_yml=True):
        """
        Saves the model to a file. Includes the classifier, the param object, and the
        state of the feature extractor model in a pickle file.

        Parameters
        ----------
        model_path : str
            Path to save the model to.
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
                    'model_state': self.fe_model.state_dict() if hasattr(self.fe_model, 'state_dict') else None
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
        if 'model_state' in data:
            self.fe_model_state = data['model_state']
            if hasattr(self.fe_model, 'load_state_dict'): # TODO: CHECK IF THIS MAKES SENSE
                self.fe_model.load_state_dict(data['model_state'])
    
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

    def _set_fe(self, fe_name=None, fe_use_cuda=None, fe_layers=None):
        """
        Sets the model based on the given FE parameters.
        Creates new feature extracture, and resets the model state and classifier.
        Only intended for internal use at model initiation.
        """

        # Reset the model and classifier
        self.fe_model_state = None
        self.reset_classifier()

        # Check if we need to create a new FE model
        new_fe_name = fe_name is not None and fe_name != self._param.get("fe_name")
        new_fe_use_cuda = fe_use_cuda is not None and fe_use_cuda != self._param.get("fe_use_cuda")
        new_fe_layers = fe_layers is not None and fe_layers != self._param.get("fe_layers")

        # Create the feature extractor model
        if new_fe_name or new_fe_use_cuda or new_fe_layers:
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

        Parameters
        ----------
        name : str
            Name of the feature extractor model
        use_cuda : bool, optional
            Whether to use CUDA for the feature extractor
        layers : list[str], optional
            List of layer names to extract features from, by default None

        Returns
        -------
        FeatureExtractor
            The created feature extractor model
        """
        
        # Check if name is valid and create the feature extractor object
        if not name in ConvpaintModel.ALL_MODELS_TYPES_DICT:
            raise ValueError(f'Feature extractor model {name} not found.')
        fe_model_class = ConvpaintModel.ALL_MODELS_TYPES_DICT.get(name)
        
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
        """
        cpm_defaults = ConvpaintModel.get_default_params() # Get ConvPaint defaults
        new_param = self.fe_model.get_default_params(cpm_defaults) # Overwrite defaults defined in the FE model
        return new_param

    def get_fe_layer_keys(self):
        """
        Returns the keys of the feature extractor layers.
        """
        return self.fe_model.get_layer_keys()

    def get_features_img(self, image):
        """
        Extracts features from an image using the Convpaint model's Parameters and feature extractor model.
        Returns the extracted features with the spatial dimensions identical to the input image.

        Parameters
        ----------
        image : np.ndarray
            Image to extract features from
        
        Returns
        -------
        f_img : np.ndarray
            Extracted features [nb_features, height, width]
        """
        features = self.fe_model.get_features_scaled(image=image, param=self._param)
        return features
    
    def train(self, image, annotations, use_rf=False, allow_writing_files=False):
        """
        Trains the Convpaint model's classifier given an image and annotations.
        Uses the Parameter and the FeatureExtractor model to extract features.
        Then uses the features of annotated pixels to train the classifier.
        Trains the internal classifier, which is also returned.

        Parameters
        ----------
        image : np.ndarray
            Image to train the classifier on
        annotations : np.ndarray
            Annotations to train the classifier on

        Returns
        -------
        CatBoostClassifier or RandomForestClassifier
            Trained classifier
        """
        features, targets = self.get_features_current_layers(image, annotations)
        self._train_classifier(features, targets, use_rf=use_rf, allow_writing_files=allow_writing_files)
        if self.classifier is None:
            raise ValueError('Training failed. No classifier was trained.')
        return self.classifier

    def segment(self, image):
        """
        Segments an image by predicting its classes using the trained classifier.
        Uses the feature extractor model to extract features from the image.
        Predicts the classes of the pixels in the image using the trained classifier.

        Parameters
        ----------
        image : np.ndarray
            Image to segment
        """
        if self._param.tile_image:
            return self._parallel_predict_image(image, return_proba=False, use_dask=False)
        else:
            return self._predict_image(image, return_proba=False)

    def predict_probas(self, image):
        """
        Predicts the probabilities of the classes of the pixels in an image using the trained classifier.
        Uses the feature extractor model to extract features from the image.
        Estimates the probablity of each class based on the features and the trained classifier.

        Parameters
        ----------
        image : np.ndarray
            Image to predict probabilities for

        Returns
        -------
        np.ndarray
            Predicted probabilities of the classes of the pixels in the image
        """
        if self._param.tile_image:
            return self._parallel_predict_image(image, return_proba=True, use_dask=False)
        else:
            return self._predict_image(image, return_proba=True)

    def predict(self, image):
        probas = self.predict_probas(image)
        seg = self._probas_to_classes(probas)
        return probas, seg


    ### CLASSIFIER METHODS

    def _train_classifier(self, features, targets, use_rf=False, allow_writing_files=False):
        """
        Trains a classifier given a set of features and targets.
        If use_rf is False, a CatBoostClassifier is trained, otherwise a RandomForestClassifier.
        The trained classifier is saved in the model, but also returned.

        Parameters
        ----------
        features : np.ndarray
            Features to train the classifier on
        targets : np.ndarray
            Targets to train the classifier on
        use_rf : bool, optional
            Whether to use a RandomForestClassifier, by default False

        Returns
        -------
        CatBoostClassifier or RandomForestClassifier
            Trained classifier
        """
        if not use_rf:
            self.classifier = CatBoostClassifier(iterations=self._param.clf_iterations,
                                                 learning_rate=self._param.clf_learning_rate,
                                                 depth=self._param.clf_depth,
                                                 allow_writing_files=allow_writing_files)
            self.classifier.fit(features, targets)
            self._param.classifier = 'CatBoost'
        else:
                # train a random forest classififer
                self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                self.classifier.fit(features, targets)
                self._param.classifier = 'RandomForest'

        return self.classifier
    
    def _clf_predict(self, features, return_proba=False):
        """
        Uses the trained classifier to predict classes based on features of the image.
        Returns an image containing the most probable class of each pixel, if return_proba is False.
        Otherwise, returns the probabilities of each class for each pixel.

        Parameters
        ----------
        features : np.ndarray
            Features to use for the prediction
        return_proba : bool, optional
            Whether to return the prediction as class probabilities, by default False

        Returns
        -------
        predictions : np.ndarray
            Predicted classes or probabilities of the classes for each pixel in the image
        """
        nb_features = features.shape[0] # [nb_features, width, height]

        # Move features to last dimension and flatten
        features = np.moveaxis(features, 0, -1)
        features = np.reshape(features, (-1, nb_features)) # flatten
        features_df = pd.DataFrame(features)

        # Predict
        if return_proba:
            predictions = self.classifier.predict_proba(features_df)
        else:
            predictions = self.classifier.predict(features_df)

        return predictions

    def reset_classifier(self):
        """
        Resets the classifier of the model.
        """
        self.classifier = None
        self._param.classifier = None


    ### OLD FE METHODS

    def _predict_image_old(self, image, return_proba=False):
        # Pad Image
        padding = self.fe_model.get_padding() * np.max(self._param.fe_scalings) * self._param.image_downsample
        if image.ndim == 2:
            image = np.pad(image, padding, mode='reflect')
        elif image.ndim == 3:
            image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')
        elif image.ndim == 4:
            image = np.pad(image, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='reflect')
        else:
            raise ValueError('Image must be 2D, 3D, or 4D.')

        # Extract features
        features = self.fe_model.get_features_scaled(image=image,param=self._param)
        nb_features = features.shape[0] #[nb_features, width, height]

        # Move features to last dimension and flatten
        features = np.moveaxis(features, 0, -1)
        features = np.reshape(features, (-1, nb_features)) # flatten
        features_df = pd.DataFrame(features)

        rows = np.floor(image.shape[-2] / self._param.image_downsample).astype(int)
        cols = np.floor(image.shape[-1] / self._param.image_downsample).astype(int)

        # Predict
        # First reshape prediction to downsampled shape
        if return_proba:
            predictions = self.classifier.predict_proba(features_df)
            predicted_image = np.reshape(predictions, [rows, cols, -1])
        else:
            predictions = self.classifier.predict(features_df)
            predicted_image = np.reshape(predictions, [rows, cols])

        # Then resize to original shape
        if self._param.image_downsample > 1:
            predicted_image = skimage.transform.resize(
                image=predicted_image,
                output_shape=(image.shape[-2], image.shape[-1]),
                preserve_range=True, order=0)
                # OLD: order=self._param.fe_order

        # Remove padding and return
        if return_proba:
            predicted_image = predicted_image[..., padding:-padding, padding:-padding, :]
            return predicted_image
        else:
            predicted_image = predicted_image[..., padding:-padding, padding:-padding]
            predicted_image = predicted_image.astype(np.uint8)
            if self._param.seg_smoothening > 1:
                predicted_image = skimage.filters.rank.majority(predicted_image,
                                    footprint=skimage.morphology.disk(self._param.seg_smoothening))
            return predicted_image
    
    def _parallel_predict_image(self, image, return_proba=False, use_dask=False):
        """
        Given a filter model and a classifier, predict the class of 
        each pixel in an image.

        Parameters
        ----------
        image: 2d array
            image to segment

        Given inside the ConvpaintModel class:    
            self.model: FeatureExtractor model
                model to extract features from
            self.classifier: CatBoostClassifier
                classifier to use for prediction
            self._param.scalings: list of ints
                downsampling factors
            self._param.order: int
                interpolation order for low scale resizing
            self._param.use_min_features: bool
                if True, use the minimum number of features per layer
            self._param.image_downsample: int, optional
                downsample image by this factor before extracting features, by default 1
            self._param.seg_smoothening: int, optional
                smoothening factor for segmentation, by default 1
        
        use_dask: bool
            if True, use dask for parallel processing

        Returns
        -------
        predicted_image: 2d array
            predicted image with classes
        """
        
        maxblock = 1000
        nblocks_rows = image.shape[-2] // maxblock
        nblocks_cols = image.shape[-1] // maxblock
        margin = 50

        if return_proba:
            num_classes = self.classifier.classes_.shape[0]
            z, h, w = image.shape[-3:]
            predicted_image_complete = np.zeros((num_classes, z, h, w),
                                                dtype=(np.float32))
        else:
            predicted_image_complete = np.zeros(image.shape[-3:], dtype=(np.uint8))

        if use_dask:
            from dask.distributed import Client
            import dask
            dask.config.set({'distributed.worker.daemon': False})
            client = Client()
            processes = []

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

        if use_dask:
            for k in range(len(processes)):
                future = processes[k]
                out = future.result()
                future.cancel()
                del future
                predicted_image_complete[...,
                    min_row_ind_collection[k]:max_row_ind_collection[k],
                    min_col_ind_collection[k]:max_col_ind_collection[k]] = out.astype(np.uint8)
            client.close()
        
        return predicted_image_complete

    def get_features_current_layers(self, image, annotations):        # ORIGINALLY FROM CONV_PAINT SCRIPT
        """Given a potentially multidimensional image and a set of annotations,
        extract multiscale features from multiple layers of a model.
        
        Parameters
        ----------
        image : np.ndarray
            2D or 3D Image to extract features from. If 3D but annotations are 2D,
            image is treated as multichannel and not image series
        annotations : np.ndarray
            2D, 3D Annotations (1,2) to extract features from

        Given inside the ConvPaint class:
            self.fe_model : feature extraction model
                Model to extract features from the image
            self._param.scalings : list of ints
                Downsampling factors
            self._param.order : int, optional
                Interpolation order for low scale resizing, by default 0
            self._param.use_min_features : bool, optional
                Use minimal number of features, by default True
            self._param.image_downsample : int, optional
                Downsample image by this factor before extracting features, by default 1

        Returns
        -------
        features : pandas DataFrame
            Extracted features (rows are pixel, columns are features)
        targets : pandas Series
            Target values
        """

        if self.fe_model is None:
            raise ValueError('Model must be provided. Define and set (or load) a model first.')

        # get indices of first dimension of non-empty annotations. Gives t/z indices
        if annotations.ndim == 3:
            non_empty = np.unique(np.where(annotations > 0)[0])
            if len(non_empty) == 0:
                warnings.warn('No annotations found')
                return None, None
        elif annotations.ndim == 2:
            non_empty = [0]
        else:
            raise Exception('Annotations must be 2D or 3D')

        all_values = []
        all_targets = []

        # find maximal padding necessary
        padding = self.fe_model.get_padding() * np.max(self._param.fe_scalings) * self._param.image_downsample

        # iterating over non_empty iteraties of t/z for 3D data
        for ind, t in enumerate(non_empty):

            # PADDING
            if annotations.ndim == 2:
                if image.ndim == 3:
                    current_image = np.pad(image, ((0,0), (padding,padding), (padding,padding)), mode='reflect')
                else:
                    current_image = np.pad(image, padding, mode='reflect')
                current_annot = np.pad(annotations, padding, mode='constant')
            else:
                if image.ndim == 3:
                    current_image = np.pad(image[t], padding, mode='reflect')
                    current_annot = np.pad(annotations[t], padding, mode='constant')
                elif image.ndim == 4:
                    current_image = np.pad(image[:, t], ((0,0),(padding, padding),(padding, padding)), mode='reflect')
                    current_annot = np.pad(annotations[t], padding, mode='constant')

            # TILE ANNOTATIONS (applying the padding to each annotations part)
            annot_regions = skimage.morphology.label(current_annot > 0)
            if self._param.tile_annotations:
                boxes = skimage.measure.regionprops_table(annot_regions, properties=('label', 'bbox'))
            else:
                boxes = {'label': [1], 'bbox-0': [padding], 'bbox-1': [padding], 'bbox-2': [current_annot.shape[0]-padding], 'bbox-3': [current_annot.shape[1]-padding]}
            for i in range(len(boxes['label'])):
                # NOTE: This assumes that the image is already padded correctly, and the padded boxes cannot go out of bounds
                # pad_size = self.fe_model.get_padding() # NOTE ROMAN: This is wrong; the padding should be scaled by the scaling factor and the check below should not be necessary !!!
                pad_size = padding
                x_min = boxes['bbox-0'][i]-pad_size
                x_max = boxes['bbox-2'][i]+pad_size
                y_min = boxes['bbox-1'][i]-pad_size
                y_max = boxes['bbox-3'][i]+pad_size

                # temporary check that bounds are not out of image
                x_min = max(0, x_min)
                x_max = min(current_image.shape[-2], x_max)
                y_min = max(0, y_min)
                y_max = min(current_image.shape[-1], y_max)

                img_tile = current_image[...,
                    x_min:x_max,
                    y_min:y_max
                ]
                annot_tile = current_annot[
                    x_min:x_max,
                    y_min:y_max
                ]
                extracted_features = self.fe_model.get_features_scaled(image=img_tile, param=self._param)

                if self._param.image_downsample > 1: # Downsample the ANNOTATION (img is done inside get_features_scaled)
                    annot_tile = conv_paint_utils.scale_img(annot_tile, self._param.image_downsample, use_labels=True)
                    # annot_tile = annot_tile[::self._param.image_downsample, :: self._param.image_downsample]

                # EXTRACT TARGETED FEATURES
                #from the [features, w, h] make a list of [features] with len nb_annotations
                mask = annot_tile > 0
                nb_features = extracted_features.shape[0]

                extracted_features = np.moveaxis(extracted_features, 0, -1) #move [w,h,features]

                extracted_features = extracted_features[mask]
                all_values.append(extracted_features)

                targets = annot_tile[annot_tile > 0]
                targets = targets.flatten()
                all_targets.append(targets)

        # CONCATENATE FROM DIFFERENT TILES
        all_values = np.concatenate(all_values, axis=0)
        features = pd.DataFrame(all_values) # [pixels, features]

        all_targets = np.concatenate(all_targets, axis=0)
        targets = pd.Series(all_targets)

        return features, targets

### NEW METHODS

    def train2(self, data, annotations, extend_features=False, use_rf=False, allow_writing_files=False):

        # Use get_feature_image to extract features and the suiting annotation parts
        features, annot_parts = self.get_feature_image(data, annotations, restore_input_form=False)

        # Get the annotated pixels and targets, and concatenate each
        f_t_tuples = [self._get_features_targets(f, a)
                      for f, a in zip(features, annot_parts)]
        features = [ft[0] for ft in f_t_tuples] # Create a list of features
        targets = [ft[1] for ft in f_t_tuples] # Create a list of targets
        features = np.concatenate(features, axis=0) # Concatenate the features into a single array
        targets = np.concatenate(targets, axis=0) # Concatenate the targets into a single array

        # If we want to extend the features, we need to add the new to the existing features
        if extend_features:
            features, targets = self._extend_features(features, targets)

        # Convert features to a DataFrame and targets to a Series
        features = pd.DataFrame(features)
        targets = pd.Series(targets)

        # Train the classifier
        self._train_classifier(features, targets,
                               use_rf=use_rf, allow_writing_files=allow_writing_files)
        
    def reset_features(self):
        """
        Resets the features and targets of the model.
        """
        self.features = None
        self.targets = None
        self.num_trainings = 0

    def _extend_features(self, features, targets):
        """
        Extend the features and targets of the model with new features and targets.
        """
        if self.features is None:
            self.features = features
        else:
            self.features = np.concatenate([self.features, features], axis=0)
        if self.targets is None:
            self.targets = targets
        else:
            self.targets = np.concatenate([self.targets, targets], axis=0)
        print(features.shape, targets.shape)
        features = self.features
        targets = self.targets
        print(features.shape, targets.shape)
        self.num_trainings += 1

        return features, targets

    def _predict_image(self, data, return_proba=False, patched=True):

        # Use get_feature_image to extract features
        features = self.get_feature_image(data, restore_input_form=False)

        # Predict pixels based on the features and classifier
        # NOTE: We always first predict probabilities and then take the argmax
        predictions = [self._clf_predict(f, return_proba=True)
                       for f in features]
        predictions = [np.moveaxis(pred, -1, 0) for pred in predictions]
        
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

        # Restore input dimensionality (especially see if we want to remove z dimension)
        data_list = [data] if not isinstance(data, list) else data
        pred_reshaped = [self._restore_dims(pred_reshaped[i], data_list[i].shape)
                            for i in range(len(data_list))]

        # If input was a single image, return the first prediction
        if not isinstance(data, list):
            return pred_reshaped[0]

        return pred_reshaped
    
    def get_feature_image(self, data, annotations=None, restore_input_form=True):
        
        # Check if we process annotations and if we have only a single image input
        use_annots = annotations is not None
        single_input = not isinstance(data, list)

        # Run checks on the model to ensure that it is ready to be used
        self._run_model_checks()

        # Make sure data is a list of images with [C, Z, H, W] shape
        data, annotations = self._prep_dims(data, annotations)
        # Save the original data shapes for reshaping/rescaling later
        self.original_shapes = [d.shape for d in data]

        # Get the parameters for feature extraction
        params_for_extract = self._enforce_fe_params(self._param)

        # Downsample the images and annotations (if given)
        data = [conv_paint_utils.scale_img(d, params_for_extract.image_downsample)
                for d in data]
        if use_annots:
            annotations = [conv_paint_utils.scale_img(a,
                                                      params_for_extract.image_downsample,
                                                      use_labels=True)
                           for a in annotations]
        
        # Get the correct padding size for the feature extraction    
        padding = self._get_total_pad_size(params_for_extract)

        # Pad the images and annotations with the correct padding size
        data = [self._pad(d, padding) for d in data]
        # Save the padded shapes for reshaping/rescaling later
        self.padded_shapes = [d.shape for d in data]
        if use_annots:
            annotations = [self._pad(a, padding, use_labels=True) for a in annotations]

        # Extract annotated planes and flatten them (treating as individual images)
        if use_annots:
            planes = [self._get_annot_planes(d, a) for d, a in zip(data, annotations)]
            data = [plane for plane_tuple in planes for plane in plane_tuple[0]]
            annotations = [plane for plane_tuple in planes for plane in plane_tuple[1]]

        # Get annotated tiles (if enabled) and flatten them (treating as individual images)
        if use_annots and params_for_extract.tile_annotations:
            tiles = [self._tile_annot(d, a, padding)
                     for d, a in zip(data, annotations)]
            data = [tile for tile_tuple in tiles for tile in tile_tuple[0]]
            annotations = [tile for tile_tuple in tiles for tile in tile_tuple[1]]

        # Extract features from the images for each scaling in the pyramid
        # For training, we want to "unpatch" the features to match the annotations
        keep_patched = not use_annots
        features = [self.fe_model.get_feature_pyramid(d, params_for_extract, patched=keep_patched)
                    for d in data]
        
        # Return the raw features if required
        if not restore_input_form:
            if use_annots:
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
        data_list = [data] if not isinstance(data, list) else data
        features = [self._restore_dims(features[i], data_list[i].shape)
                            for i in range(len(data_list))]

        # If input was a single image, return the first (= only) features
        if single_input:
            return features[0]

        return features

    def _run_model_checks(self):
        """
        Run checks on the model to ensure that it is ready to be used.
        """
        if self.fe_model is None:
            raise ValueError('No feature extractor model set. Please set a feature extractor model first.')
    
    def _prep_dims(self, data, annotations=None):
        """
        Preprocess the dimensions of the data and annotations (if given).
        Ensures the data and annotations have the correct number of dimensions,
        where the data is a list of images with [C, Z, H, W] shape,
        and the annotations are a list of images with [Z, H, W] shape.
        Assumes that a possible channels dimension is the first one.
        Handles both single images and lists of images.
        """
        # Ensure data and annotations are lists
        if not isinstance(data, list):
            img_list = [data]
        else:
            img_list = data
        if annotations is None:
            annot_list = [None] * len(img_list)
        elif not isinstance(annotations, list):
            annot_list = [annotations]
        else:
            annot_list = annotations

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

        return prep_imgs, prep_annots
        
    def _prep_dims_single(self, img, annotations=None):
        """
        Preprocess the dimensions of the data and annotations (if given).
        Ensures the data and annotations have the correct number of dimensions,
        where the data is a list of images with [C, Z, H, W] shape,
        and the annotations are a list of images with [Z, H, W] shape.
        Assumes that a possible channels dimension is the first one.
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
        # Get the padding size from the feature extractor model
        fe_pad = self.fe_model.get_padding()
        # Make sure the reducing to patch size will not reduce below the necessary padding
        fe_patch = self.fe_model.get_patch_size()
        # Scale the padding by the highest scaling factors (such that the image is not reduced too much)
        max_scale = np.max(param.fe_scalings)
        # Calculate the final padding size
        return (fe_pad + fe_patch//2) * max_scale

    def _pad(self, img, padding, use_labels=False):
        """
        Pads the image with the padding size necessary for the feature extraction,
        given the feature extractor model and the parameters.
        The padding is done with the reflect mode for images and constant for annotations.
        Dimensions are assumed [C, Z, H, W] or [Z, H, W] for images and annotations, respectively.
        """
        # Pad the image with the padding size
        if use_labels:
            return np.pad(img, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            return np.pad(img, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='reflect')
        
    def _get_annot_planes(self, img, annot=None):
        """
        Extract the planes (of the z dimension) of the padded data and annotations
        where there is at least one annotation.
        Dimensions are assumed [C, Z, H, W] or [Z, H, W] for images and annotations, respectively.
        """
        if annot is None:
            img_planes = [img[:, z:z+1] for z in range(img.shape[1])]
            return img_planes, None
        
        non_empty = np.unique(np.where(annot > 0)[0])
        if len(non_empty) == 0:
            warnings.warn('No annotations found')
            return None, None
        
        # Extract the planes of the image and annotations
        img_planes = [img[:,z:z+1] for z in non_empty] # images have a channels dimension
        annot_planes = [annot[z:z+1] for z in non_empty] # annotations do not have a channels dimension

        return img_planes, annot_planes

    def _tile_annot(self, img, annot, padding):
        """
        Tile the image and annotations based on the annotations.
        Tile them into patches of the size of the annotations.
        Dimensions are assumed [C, Z, H, W] or [Z, H, W] for images and annotations, respectively.
        """
        # Find the bounding boxes of the annotations
        annot_regions = skimage.morphology.label(annot > 0)
        boxes = skimage.measure.regionprops_table(annot_regions, properties=('label', 'bbox'))

        # Create lists to store the tiles
        img_tiles = []
        annot_tiles = []

        for i in range(len(boxes['label'])):
            # Get the bounding box of the annotation
            x_min = boxes['bbox-1'][i] - padding
            x_max = boxes['bbox-4'][i] + padding
            y_min = boxes['bbox-2'][i] - padding
            y_max = boxes['bbox-5'][i] + padding

            # Pad the image and annotations with the correct padding size
            img_tile = img[..., x_min:x_max, y_min:y_max]
            annot_tile = annot[..., x_min:x_max, y_min:y_max]

            img_tiles.append(img_tile)
            annot_tiles.append(annot_tile)

        return img_tiles, annot_tiles
    
    def _get_features_targets(self, features, targets):
        """
        Given a set of features and targets, extract the annotated pixels and targets,
        and concatenate each.
        """
        # Get the annotated pixels and targets
        mask = targets > 0
        features = np.moveaxis(features, 0, -1) #move [z, h, w, features]
        # Select only the pixels that are annotated
        features = features[mask]
        targets = targets[mask]

        return features, targets
    
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
        seg = class_labels[max_prob]

        # Smoothen the segmentation if desired
        if self._param.seg_smoothening > 1:
            kernel = skimage.morphology.disk(self._param.seg_smoothening)
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
            if pred.shape[0] != 1:
                raise ValueError('Prediction shape does not match original shape.')
            # [Z,H,W] for prediction, [proba/F,Z,H,W] for probabilities and features
            pred = np.squeeze(pred, axis=-3)
        return pred