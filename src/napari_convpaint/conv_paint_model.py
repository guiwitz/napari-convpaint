
import pickle
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings

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
        self.init_models_dict()
        self.classifier = None
        self.param = None
        self.fe_model = None
        self.fe_model_state = None
        # self.total_pad = None # Parameter to keep track of the overall padding needed (considering all scalings and padding)
        if model_path is not None:
            self.load_model(model_path)
        else:
            self.set_default_model()

    def init_models_dict(self):
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
            
    def set_default_model(self):
        self.param = Param()
        self.param.fe_name = 'vgg16'

        # ToDo: DEFINE DEFAULT CONVPAINT MODEL HERE

        self.set_fe_model()

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.param = data['param']
        self.set_fe_model(self.param) # Resets classifier and model_state
        self.classifier = data['classifier']
        if 'model_state' in data:
            self.fe_model_state = data['model_state']
        # else:
        #     self.fe_model_state = None

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            data = {
                'classifier': self.classifier,
                'param': self.param,
                'model_state': self.fe_model.state_dict() if hasattr(self.fe_model, 'state_dict') else None
            }
            pickle.dump(data, f)

    def set_fe_model(self):
        """Set the model based on the given parameters."""

        # USE THESE AS ARGUMENTS:
        # # Image processing parameters
        # multi_channel_img: bool = None
        # normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image
        # # Acceleration parameters
        # image_downsample: int = None
        # tile_annotations: bool = False
        # tile_image: bool = False
        # # Model parameters
        # fe_name: str = None
        # fe_layers: list[str] = None
        # fe_padding : int = 0
        # fe_scalings: list[int] = None
        # fe_order: int = None
        # fe_use_min_features: bool = None
        # fe_use_cuda: bool = None
        # # Classifier parameters
        # clf_iterations: int = None
        # clf_learning_rate: float = None
        # clf_depth: int = None


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

    def train_classifier(self, features, targets, iterations = 50, learning_rate = 0.1, depth = 5, use_rf=False):
        """Train a classifier given a set of features and targets."""
        if not use_rf:
            self.classifier = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate,depth=depth)
            self.classifier.fit(features, targets)
        else:
                # train a random forest classififer
                classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                classifier.fit(features, targets)





    def get_features_targets_img_stack(self, img_stack, annots_stack):
        # Extract stacks with annotations
        annot_stacks = np.unique(np.where(annots_stack > 0)[0])
        if len(annot_stacks) == 0:
            warnings.warn('No annotations found')
            return None, None
        np.moveaxis(img_stack, -3, 0)
        img_stack = img_stack[annot_stacks]
        np.moveaxis(img_stack, 0, -3)
        annots_stack = annots_stack[annot_stacks]
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