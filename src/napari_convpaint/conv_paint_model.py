
import pickle
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings
import pandas as pd
import skimage

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
        self._param = Param()
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
        return self._predict_image(image, return_proba=True)

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

    def reset_classifier(self):
        """
        Resets the classifier of the model.
        """
        self.classifier = None
        self._param.classifier = None


    ### OLD FE METHODS

    def _predict_image(self, image, return_proba=False):                    # FROM FEATURE EXTRACTOR CLASS
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
    
    def parallel_predict_image(self, image, use_dask=False):
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

        predicted_image_complete = np.zeros(image.shape[-2:], dtype=(np.uint8))
        
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
                #print(f'row {row}/{nblocks_rows}, col {col}/{nblocks_cols}')
                max_row = np.min([image.shape[-2], (row+1)*maxblock+margin])
                max_col = np.min([image.shape[-1], (col+1)*maxblock+margin])
                min_row = np.max([0, row*maxblock-margin])
                min_col = np.max([0, col*maxblock-margin])

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
                        self._predict_image, image=image_block))
                    
                    min_row_ind_collection.append(min_row_ind)
                    min_col_ind_collection.append(min_col_ind)
                    max_row_ind_collection.append(max_row_ind)
                    max_col_ind_collection.append(max_col_ind)
                    new_max_col_ind_collection.append(new_max_col_ind)
                    new_max_row_ind_collection.append(new_max_row_ind)
                    new_min_col_ind_collection.append(new_min_col_ind)
                    new_min_row_ind_collection.append(new_min_row_ind)

                else:
                    predicted_image = self._predict_image(image_block)
                    crop_pred = predicted_image[
                        new_min_row_ind: new_max_row_ind,
                        new_min_col_ind: new_max_col_ind]

                    predicted_image_complete[min_row_ind:max_row_ind, min_col_ind:max_col_ind] = crop_pred.astype(np.uint8)

        if use_dask:
            for k in range(len(processes)):
                future = processes[k]
                out = future.result()
                future.cancel()
                del future
                predicted_image_complete[
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

            annot_regions = skimage.morphology.label(current_annot > 0)

            # TILE ANNOTATIONS (applying the padding to each annotations part)
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


    ### NEW Feature extraction methods

    def run_model_checks(self):
        """Run checks on the model to ensure that it is ready to be used."""
        if self.fe_model is None:
            raise ValueError('No feature extractor model set. Please set a feature extractor model first.')
        if self.classifier is None:
            raise ValueError('No classifier set. Please train a classifier first.')
        # Check if the kernel_size and patch_size are compatible
        ks = self.fe_model.kernel_size
        ps = self.fe_model.patch_size
        if ks is not None and ps is not None:
            if ks[0] is None and ps[0] is not None:
                raise ValueError('Kernel size is 2D, but patch size is 3D.')
            if ks[0] is not None and ps[0] is None:
                raise ValueError('Kernel size is 3D, but patch size is 2D.')

    def pre_process_list(self, data_stack_list):
        """Preprocess a list of data stacks (can be image stack [C, Z, H, W] or annotations [Z, H, W])."""
        # If single data stack given, convert it to a list
        if not isinstance(data_stack_list, list):
            data_stack_list = [data_stack_list]
        
        # Preprocess each data_stack in the list
        output_data_stack_list = [self.pre_process_stack(data_stack) for data_stack in data_stack_list]

        return output_data_stack_list

    def pre_process_stack(self, data_stack):
        input_scaling = self._param.image_downsample
        effective_kernel_padding = self.fe_model.get_effective_kernel_padding()
        effective_patch_size = self.fe_model.get_effective_patch_size()
        # Process input data stack
        processed_data_stack = conv_paint_utils.pre_process_stack(data_stack,
                                                                 input_scaling,
                                                                 effective_kernel_padding,
                                                                 effective_patch_size)
        # Return processed stacks
        return processed_data_stack

    # Specific for training

    def get_annot_img_list(self, img_stack, annot_stack):
        effective_kernel_padding = self.fe_model.get_effective_kernel_padding()
        effective_patch_size = self.fe_model.get_effective_patch_size()
        kernel_size = self.fe_model.kernel_size
        patch_size = self.fe_model.patch_size
        extract_3d = ((kernel_size is not None and kernel_size[0] is not None) or
                      (patch_size is not None and patch_size[0] is not None))
        tile_annots = self._param.tile_annotations
        if extract_3d and tile_annots:
            annot_img_list = conv_paint_utils.tile_annots_3d(img_stack,
                                                             annot_stack,
                                                             effective_kernel_padding,
                                                             effective_patch_size)
        elif tile_annots: # 2D tile annotations
            annot_img_list = conv_paint_utils.tile_annots_2d(img_stack,
                                                             annot_stack,
                                                             effective_kernel_padding,
                                                             effective_patch_size)
        elif extract_3d: # 3D extract annotations without tiling
            annot_img_list = conv_paint_utils.extract_annots_3d(img_stack,
                                                                annot_stack,
                                                                effective_kernel_padding,
                                                                effective_patch_size)
        else: # 2D extract annotations without tiling
            annot_img_list = conv_paint_utils.extract_annots_2d(img_stack,
                                                                annot_stack,
                                                                effective_kernel_padding,
                                                                effective_patch_size)
        return annot_img_list

    def get_features_targets_img_stack(self, img_stack, annots_stack):
        # Extract stacks with annotations
        annotated_layers = np.unique(np.where(annots_stack > 0)[0])
        if len(annotated_layers) == 0:
            warnings.warn('No annotations found')
            return None, None
        np.moveaxis(img_stack, -3, 0)
        img_stack = img_stack[annotated_layers]
        np.moveaxis(img_stack, 0, -3)
        annots_stack = annots_stack[annotated_layers]
        # Preprocess 
        output_img_stack, output_annots_stack = [], []
        for img, annots in zip(img_stack, annots_stack):
            img_tiles, annots_tiles = self.pre_process_img_annots_for_training(img, annots)
            output_img_stack.append(img_tiles)
            output_annots_stack.append(annots_tiles)

        # Create a numpy array with stack dimension as second dimension
        output_img_stack = np.stack(output_img_stack, axis=1)
        output_annots_stack = np.stack(output_annots_stack, axis=1)
        return output_img_stack, output_annots_stack

    def get_features_targets_multi_image(self, image_list, annots_list):
        all_features, all_targets = [], []
        for img, annots in zip(image_list, annots_list):
            features, targets = self.get_features_targets_img_stack(img, annots)

######################################

    def train_new(self, img, annotations):
        """
        Train the model on the given image and annotations.
        """
        self.annots = annotations
        self.patch_size = self.fe_model.get_patch_size()
        self.padding = self.fe_model.get_padding()
        tile_annots = self._param.get("tile_annotations")
        self.run_checks()
        prep_img = self.prepare_img_dims(img) # -> Z, C, H, W
        prep_annots = self.prepare_annots_dims(annotations)
        annotated_slices, non_empty_annots = self.extract_annotated_slices(prep_img, prep_annots)
        features_lin = self.get_features_lin(annotated_slices, tile_annotations=tile_annots)
        f, t = self.get_features_annots(features_lin) # For patch-based FE, create image, re-scale and re-linearize first
        clf = self._train_classifier(f, t)
        return clf

    def prepare_img_dims(self, img):
        """
        Prepare the dimensions of the image.
        Makes sure the image is in Z,C,H,W format (if necessary with Z and/or C = 1).
        """
        ndim = img.ndim
        if ndim == 2:
            img = img[np.newaxis, np.newaxis, :, :]

        if ndim == 3 and self._param.get("rgb_img") and img.shape[2] == 3:
            img = np.moveaxis(img, 2, 0) # -> C, H, W -> Can now be treated together with multi-channel images
        if ndim == 3 and not self._param.get("multi_channel_img"):
            img = img[:, np.newaxis, :, :] # non-multichannel have Z -> C = 1 is added
        elif ndim == 3:
            img = img[np.newaxis, :, :, :] # multichannel (and RGB) have C -> Z = 1 is added

        return img
        
    def prepare_annots_dims(self, annotations):
        """
        Prepare the dimensions of the annotations.
        Makes sure the annotations are in Z,H,W format.
        """
        if annotations is not None:
            ndim = annotations.ndim
            if ndim == 2:
                annotations = annotations[np.newaxis, :, :] # add Z = 1
            if ndim == 3:
                annotations = annotations[:, :, :] # if 3D, it must already be Z, H, W
        return annotations
    
    def get_features(self,  img_list, annots=None, do_pad=True, do_downsample=True, tile_annotations=False):
        
        for img in img_list:
            self.run_img_checks(img)
            img = self.prepare_img_dims(img)
            img_scaled = self.downsample(img)
            if annots is not None:
                img, annots = self.extract_annotated_slices(img, annots)
            annot_scaled = self.downsample(self.annots)
            img_padded = self.pad_image(img_scaled)
            annot_padded = self.pad_image(self.annots)
        