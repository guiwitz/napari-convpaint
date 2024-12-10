
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
from napari_convpaint.conv_paint_gaussian import GaussianFeatures
from napari_convpaint.conv_paint_dino import DinoFeatures
from .conv_paint_nnlayers import Hookmodel
from .conv_paint_param import Param
from . import conv_paint_utils

class ConvpaintModel:

    def __init__(self, model_path=None):
        self._init_models_dict()
        if model_path is not None:
            self.load(model_path)
        else:
            self.param = self.get_default_param()
            self.set()

    def _init_models_dict(self):
        # Initialize the MODELS TO TYPES dictionary with the models that are always available
        self.ALL_MODELS_TYPES_DICT = {x: Hookmodel for x in NN_MODELS}
        self.ALL_MODELS_TYPES_DICT.update({x: GaussianFeatures for x in GAUSSIAN_MODELS})
        self.ALL_MODELS_TYPES_DICT.update({x: DinoFeatures for x in DINO_MODELS})

        # Try to import CellposeFeatures and update the MODELS TO TYPES  dictionary if successful
        # Cellpose is only installed with pip install napari-convpaint[cellpose]
        try:
            from napari_convpaint.conv_paint_cellpose import AVAILABLE_MODELS as CELLPOSE_MODELS
            from napari_convpaint.conv_paint_cellpose import CellposeFeatures
            self.ALL_MODELS_TYPES_DICT.update({x: CellposeFeatures for x in CELLPOSE_MODELS})
        except ImportError:
            # Handle the case where CellposeFeatures or its dependencies are not available
            print("Info: Cellpose is not installed and is not available as feature extractor.\n"
                "Run 'pip install napari-convpaint[cellpose]' to install it.")
            
    def load(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.param = data['param']
        self.set() # Resets classifier and model_state
        self.classifier = data['classifier']
        if 'model_state' in data:
            self.fe_model_state = data['model_state']

    def save(self, model_path):
        with open(model_path, 'wb') as f:
            data = {
                'classifier': self.classifier,
                'param': self.param,
                'model_state': self.fe_model.state_dict() if hasattr(self.fe_model, 'state_dict') else None
            }
            pickle.dump(data, f)

    def set(self, **kwargs):
        """Set the model based on the given parameters."""
        self.set_param(**kwargs)
        # Reset the model and classifier; create a new FE model
        self.fe_model_state = None
        self.classifier = None
        fe_model_class = self.ALL_MODELS_TYPES_DICT[self.param.fe_name]
        self.fe_model = fe_model_class(
            model_name=self.param.fe_name,
            use_cuda=self.param.fe_use_cuda
        )
        # Register hooks if the model is a Hookmodel
        if isinstance(self.fe_model, Hookmodel):
            if self.param.fe_layers:
                self.fe_model.register_hooks(selected_layers=self.param.fe_layers)
            elif len(self.fe_model.named_modules) == 1:
                self.fe_model.register_hooks(selected_layers=[list(self.fe_model.module_dict.keys())[0]])
        
        # SET ENFORCED/DEFAULT PARAMS ???


    def set_param(self,
                  multi_channel_img: bool = None,
                  normalize: int = None, # 1: no normalization, 2: normalize stack, 3: normalize each image
                  image_downsample: int = None,
                  tile_annotations: bool = False,
                  tile_image: bool = False,
                  fe_name: str = None,
                  fe_layers: list[str] = None,
                  fe_padding : int = 0,
                  fe_scalings: list[int] = None,
                  fe_order: int = None,
                  fe_use_min_features: bool = None,
                  fe_use_cuda: bool = None,
                  clf_iterations: int = None,
                  clf_learning_rate: float = None,
                  clf_depth: int = None):
        """Set the values of the param objects in the model."""
        for attr, val in {
            'multi_channel_img': multi_channel_img,
            'normalize': normalize,
            'image_downsample': image_downsample,
            'tile_annotations': tile_annotations,
            'tile_image': tile_image,
            'fe_name': fe_name,
            'fe_layers': fe_layers,
            'fe_padding': fe_padding,
            'fe_scalings': fe_scalings,
            'fe_order': fe_order,
            'fe_use_min_features': fe_use_min_features,
            'fe_use_cuda': fe_use_cuda,
            'clf_iterations': clf_iterations,
            'clf_learning_rate': clf_learning_rate,
            'clf_depth': clf_depth
        }.items():
            if val is not None:
                setattr(self.param, attr, val)

    def load_param(self, param: Param):
        """Load the given param object into the model and set the model accordingly."""
        self.param = param
        self.set()

    def get_default_param(self, param=None):
        """Return a default param object, which defines the default model."""

        if param is None:
            param = Param()

        # Image processing parameters
        param.multi_channel_img = False
        param.rgb_img = False
        param.normalize = 2  # 1: no normalization, 2: normalize stack, 3: normalize each image

        # Acceleration parameters
        param.image_downsample = 1
        param.tile_annotations = True
        param.tile_image = False

        # Model parameters
        param.fe_name = "vgg16"
        param.fe_layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))']
        param.fe_padding = 0
        param.fe_scalings = [1, 2, 4]
        param.fe_order = 0
        param.fe_use_min_features = False
        param.fe_use_cuda = False

        # Classifier parameters
        param.clf_iterations = 50
        param.clf_learning_rate = 0.1
        param.clf_depth = 5

        return param
    
    ### OLD FE and Classifier methods

    def predict_image(self, image):                    # FROM FEATURE EXTRACTOR CLASS
        features = self.fe_model.get_features_scaled(image=image,param=self.param)
        nb_features = features.shape[0] #[nb_features, width, height]

        # Move features to last dimension
        features = np.moveaxis(features, 0, -1)
        features = np.reshape(features, (-1, nb_features))

        rows = np.ceil(image.shape[-2] / self.param.image_downsample).astype(int)
        cols = np.ceil(image.shape[-1] / self.param.image_downsample).astype(int)

        all_pixels = pd.DataFrame(features)
        predictions = self.classifier.predict(all_pixels)

        predicted_image = np.reshape(predictions, [rows, cols])
        if self.param.image_downsample > 1:
            predicted_image = skimage.transform.resize(
                image=predicted_image,
                output_shape=(image.shape[-2], image.shape[-1]),
                preserve_range=True, order=self.param.fe_order).astype(np.uint8)
        return predicted_image

    def get_features_current_layers(self, image, annotations):        # FROM CONV_PAINT SCRIPT
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
            self.param.scalings : list of ints
                Downsampling factors
            self.param.order : int, optional
                Interpolation order for low scale resizing, by default 0
            self.param.use_min_features : bool, optional
                Use minimal number of features, by default True
            self.param.image_downsample : int, optional
                Downsample image by this factor before extracting features, by default 1
        Returns
        -------
        features : pandas DataFrame
            Extracted features (rows are pixel, columns are features)
        targets : pandas Series
            Target values
        """

        if self.fe_model is None:
            raise ValueError('Model must be provided')

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
        padding = self.param.fe_padding * np.max(self.param.fe_scalings)

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
            if self.param.tile_annotations:
                boxes = skimage.measure.regionprops_table(annot_regions, properties=('label', 'bbox'))
            else:
                boxes = {'label': [1], 'bbox-0': [padding], 'bbox-1': [padding], 'bbox-2': [current_annot.shape[0]-padding], 'bbox-3': [current_annot.shape[1]-padding]}
            for i in range(len(boxes['label'])):
                # NOTE: This assumes that the image is already padded correctly, and the padded boxes cannot go out of bounds
                pad_size = self.param.fe_padding # NOTE ROMAN: This is wrong; the padding should be scaled by the scaling factor and the check below should not be necessary !!!
                x_min = boxes['bbox-0'][i]-pad_size
                x_max = boxes['bbox-2'][i]+pad_size
                y_min = boxes['bbox-1'][i]-pad_size
                y_max = boxes['bbox-3'][i]+pad_size

                # temporary check that bounds are not out of image
                x_min = max(0, x_min)
                x_max = min(current_image.shape[-2], x_max)
                y_min = max(0, y_min)
                y_max = min(current_image.shape[-1], y_max)

                im_crop = current_image[...,
                    x_min:x_max,
                    y_min:y_max
                ]
                annot_crop = current_annot[
                    x_min:x_max,
                    y_min:y_max
                ]
                extracted_features = self.fe_model.get_features_scaled(image=im_crop,param=self.param)

                if self.param.image_downsample > 1:
                    annot_crop = annot_crop[::self.param.image_downsample, :: self.param.image_downsample]

                # EXTRACT TARGETED FEATURES
                #from the [features, w, h] make a list of [features] with len nb_annotations
                mask = annot_crop > 0
                nb_features = extracted_features.shape[0]

                extracted_features = np.moveaxis(extracted_features, 0, -1) #move [w,h,features]

                extracted_features = extracted_features[mask]
                all_values.append(extracted_features)

                targets = annot_crop[annot_crop > 0]
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

    def pre_process_stack(self, data_stack): # TODO: REMOVE EXAMPLE VALUES...
        input_scaling = 2 #self.param.image_downsample
        effective_kernel_padding = (0, 12, 12) #self.fe_model.get_effective_kernel_padding()
        effective_patch_size = None #self.fe_model.get_effective_patch_size()
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
        tile_annots = self.param.tile_annotations
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

    ### Classifier methods

    def train_classifier(self, features, targets, iterations = 50, learning_rate = 0.1, depth = 5, use_rf=False):
        """Train a classifier given a set of features and targets."""
        if not use_rf:
            self.classifier = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate,depth=depth)
            self.classifier.fit(features, targets)
        else:
                # train a random forest classififer
                self.classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                self.classifier.fit(features, targets)


