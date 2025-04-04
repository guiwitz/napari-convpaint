from __future__ import annotations
from dataclasses import dataclass, field, asdict
import dataclasses
from pathlib import Path
import yaml

@dataclass
class Param:
    """
    Object storing relevant information regarding the processing,
    e.g. the window size (padding), the analyzed data, the type of segmentation used.

    Parameters saved in a Param object:
    ----------------
        classifier : str
            path to the classifier model
        multi_channel_img : bool = None
            if the image dimensions allow, use multichannel NOTE: Needs overthinking
        rgb_img : bool
            if True, RGB images are used
        normalize : int
            normalization mode
            1: no normalization, 2: normalize stack, 3: normalize each image
        image_downsample : int
            factor for downscaling the image right after input
            (predicted classes are upsampled accordingly for output)
        seg_smoothening : int
            factor for smoothening the segmentation output with a Majority filter
        tile_annotations : bool
            if True, extract only features of bounding boxes around annotated areas
        tile_image : bool
            if True, extract features in tiles (for large images)
        fe_name : str
            name of the feature extractor model
        fe_layers : list[str]
            list of layers (names) to extract features from
        fe_use_cuda : bool
            whether to use cuda (GPU) for feature extraction
        fe_scalings : list[int]
            list of scaling factors for the feature extractor, creating a pyramid of features
            (features are upscaled accordingly before input to classifier)
        fe_order : int
            interpolation order used for the upscaling of features for the pyramid
        fe_use_min_features : bool
            if True, use the minimum number of features among all layers
        clf_iterations : int
            number of iterations for the classifier
        clf_learning_rate : float
            learning rate for the classifier
        clf_depth : int = None
            depth of the classifier
    """
    classifier: str = None

    # Image processing parameters
    multi_channel_img: bool = None
    rgb_img: bool = None
    normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image

    # Acceleration parameters
    image_downsample: int = None
    seg_smoothening: int = None
    tile_annotations: bool = None
    tile_image: bool = None

    # Feature Extractor parameters
    fe_name: str = None
    fe_use_cuda: bool = None
    fe_layers: list[str] = None
    fe_scalings: list[int] = None
    fe_order: int = None
    fe_use_min_features: bool = None
    
    # Classifier parameters
    clf_iterations: int = None
    clf_learning_rate: float = None
    clf_depth: int = None
    
    def get(self, key):
        """
        Get the value of a parameter.

        Parameters:
        ----------
        key : str
            key of the parameter.

        Returns
        -------
        any : any
            value of the parameter.
        """
        return asdict(self)[key]

    @staticmethod
    def get_keys():
        """
        Get the keys of the parameters.

        Returns
        -------
        list[str] : list
            list of keys.
        """
        return list(asdict(Param()).keys())
    
    def set_single(self, key, value):
        """
        Set the value of a single parameter.

        Parameters:
        ----------
        key : str
            key of the parameter.
        value : any
            value of the parameter.
        """
        if key in Param.get_keys():
            setattr(self, key, value)
        else:
            raise ValueError(f"Parameter {key} not found in Param object.")

    def set(self, **kwargs):
        """
        Set the value of a parameter.

        Parameters:
        ----------
        kwargs : dict
            dictionary containing the key and value of the parameter.
        """
        for key, value in kwargs.items():
            self.set_single(key, value)
    
    def copy(self):
        """
        Copy the parameter object.

        Returns
        -------
        Param : Param
            copied parameter object.
        """
        return Param(**asdict(self))

    def save(self, save_path):
        """
        Save parameters as yml file.

        Parameters:
        ----------
        save_path : str or Path
            place where to save the parameters file.
        """
        save_path = Path(save_path)
    
        with open(save_path, "w") as file:
            dict_to_save = dataclasses.asdict(self)
            # dict_to_save = self.convert_path(dict_to_save, 'classifier')
            
            yaml.dump(dict_to_save, file)

    @staticmethod
    def load(load_path):
        """
        Load parameters from yml file.

        Parameters:
        ----------
        load_path : str or Path
            place where to load the parameters file.

        Returns
        -------
        Param : Param
            loaded parameter object.
        """
        load_path = Path(load_path)

        with open(load_path, "r") as file:
            dict_loaded = yaml.safe_load(file)

        return Param(**dict_loaded)
