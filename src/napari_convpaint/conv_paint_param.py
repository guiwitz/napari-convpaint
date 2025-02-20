from __future__ import annotations
from dataclasses import dataclass, field, asdict
import dataclasses
from pathlib import Path
import yaml

@dataclass
class Param:
    """Object storing relevant information regarding the processing,
    e.g. the window size (padding), the analyzed data, the type of segmentation used.

    Parameters saved in a Param object:
    ----------------
        classifier: str
            path to the classifier model
        multi_channel_img: bool = None
            if the image dimensions allow, use multichannel NOTE: Needs overthinking
        rgb_img: bool
            if True, RGB images are used
        normalize: int
            normalization mode
            1: no normalization, 2: normalize stack, 3: normalize each image
        image_downsample: int
            factor for downscaling the image right after input
            (predicted classes are upsampled accordingly for output)
        tile_annotations: bool
            if True, extract only features of bounding boxes around annotated areas
        tile_image: bool
            if True, extract features in tiles (for large images)
        fe_name: str
            name of the feature extractor model
        fe_layers: list[str]
            list of layers (names) to extract features from
        fe_scalings: list[int]
            list of scaling factors for the feature extractor, creating a pyramid of features
            (features are upscaled accordingly before input to classifier)
        fe_order: int
            interpolation order used for the upscaling of features for the pyramid
        fe_use_min_features: bool
            if True, use the minimum number of features among all layers
        fe_use_cuda: bool
            whether to use cuda (GPU) for feature extraction
        clf_iterations: int
            number of iterations for the classifier
        clf_learning_rate: float
            learning rate for the classifier
        clf_depth: int = None
            depth of the classifier
    """

    classifier: str = None

    # Image processing parameters
    multi_channel_img: bool = None
    rgb_img: bool = None
    normalize: int = None # 1: no normalization, 2: normalize stack, 3: normalize each image

    # Acceleration parameters
    image_downsample: int = None
    tile_annotations: bool = False
    tile_image: bool = False

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

    def save_parameters(self, save_path):
        """Save parameters as yml file.

        Parameters
        ----------
        save_path : str or Path
            place where to save the parameters file.
        """

        save_path = Path(save_path)
    
        with open(save_path, "w") as file:
            dict_to_save = dataclasses.asdict(self)
            # dict_to_save = self.convert_path(dict_to_save, 'classifier')
            
            yaml.dump(dict_to_save, file)

    # def convert_path(self, dict, path):
    #     """Convert a path to a str.

    #     Parameters
    #     ----------
    #     dict : dict
    #         dictionary containing the path.
    #     path : str
    #         path to convert.

    #     Returns
    #     -------
    #     dict: dict
    #         dict with converted path.
    #     """

    #     if dict[path] is not None:
    #         if not isinstance(dict[path], str):
    #             dict[path] = dict[path].as_posix()
        
    #     return dict
    
    def copy(self):
        """Copy the parameter object.

        Returns
        -------
        Param: Param
            copied parameter object.
        """

        return Param(**asdict(self))

    @staticmethod
    def get_keys():
        """Get the keys of the parameters.

        Returns
        -------
        list[str]: list
            list of keys.
        """

        return list(asdict(Param()).keys())
