from __future__ import annotations
from dataclasses import dataclass, field, asdict
import dataclasses
from pathlib import Path
import yaml

@dataclass
class Param:
    """
    The `Param` object organizes the **parameters that control the behavior of the feature extraction and classification processes**.
    
    It therefore defines the **processing and results** with Convpaint.
    
    These parameters can be adjusted to optimize the **performance** of the model for specific use cases.

    Parameters
    ----------
    classifier : str
        Path to the classifier model (if saved, otherwise None)
    multi_channel_img : bool = None
        Interpret the first dimension as channels (as opposed to z or time)
    normalize : int
        Normalization mode:
            1 = no normalization,
            2 = normalize stack,
            3 = normalize each image
    image_downsample : int
        Factor for downscaling the image right after input
        (predicted classes are upsampled accordingly for output).
        Hint: use negative numbers for upsampling instead.
    seg_smoothening : int
        Factor for smoothening the segmentation output with a Majority filter
    tile_annotations : bool
        If True, extract only features of bounding boxes around annotated areas when training
    tile_image : bool
        If True, extract features in tiles when running predictions (for large images)
    use_dask : bool
        If True, use dask for parallel processing (currently only used when tiling images)
    unpatch_order : int
        Order of interpolation for unpatching the output of patch-based FEs (default = 1 = bilinear interpolation)
    fe_name : str
        Name of the feature extractor model
    fe_layers : list[str]
        List of layers (names or indices among available layers) to extract features from
    fe_use_gpu : bool
        Whether to use GPU for feature extraction
    fe_scalings : list[int]
        List of scaling factors for the feature extractor, creating a pyramid of features
        (features are rescaled accordingly before input to classifier)
    fe_order : int
        Interpolation order used for the upscaling of features of the pyramid
    fe_use_min_features : bool
        If True, use the minimum number of features among all layers (simply taking the first x features)
    clf_iterations : int
        Number of iterations for the classifier
    clf_learning_rate : float
        Learning rate for the classifier
    clf_depth : int = None
        Depth of the classifier
    clf_use_gpu : bool = None
        Whether to use GPU for the classifier
        (if None, fe_use_gpu is used)
    """
    classifier: str = None

    # Image type parameters
    multi_channel_img: bool = None # Interpret the first dimension as channels
    normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image

    # Input and output parameters
    image_downsample: int = None
    seg_smoothening: int = None
    tile_annotations: bool = None
    tile_image: bool = None
    use_dask: bool = None
    unpatch_order: int = None # Order of interpolation for unpatching the output (default is 1, i.e. bilinear interpolation)

    # Feature Extractor parameters
    fe_name: str = None
    fe_use_gpu: bool = None
    fe_layers: list[str] = None
    fe_scalings: list[int] = None
    fe_order: int = None
    fe_use_min_features: bool = None
    
    # Classifier parameters
    clf_iterations: int = None
    clf_learning_rate: float = None
    clf_depth: int = None
    clf_use_gpu: bool = None # NOTE: If this is None, fe_use_gpu is used...
    
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
