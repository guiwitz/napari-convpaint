
from __future__ import annotations
from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import yaml

@dataclass
class Param:
    """Object storing relevant information regarding the processing,
    e.g. the window size (padding), the analyzed data, the type of segmentation used.
    """

    classifier: str = None
    # Data parameters
    multi_channel_training: bool = None
    # Processing parameters
    image_downsample: int = None
    tile_annotations: bool = False
    tile_image: bool = False
    normalize: int = None
    # Feature extractor parameters
    fe_name: str = None
    fe_layers: list[str] = None
    fe_scalings: list[int] = None
    fe_order: int = None
    fe_padding : int = 0
    fe_use_min_features: bool = None
    fe_use_cuda: bool = None
    # Classifier parameters
    classif_iterations: int = None
    classif_learning_rate: float = None
    classif_depth: int = None



    def __post_init__(self):
        self.scalings = [1, 2]

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
            dict_to_save = self.convert_path(dict_to_save, 'classifier')
            
            yaml.dump(dict_to_save, file)

    def convert_path(self, dict, path):
        """Convert a path to a str.

        Parameters
        ----------
        dict : dict
            dictionary containing the path.
        path : str
            path to convert.

        Returns
        -------
        dict: dict
            dict with converted path.
        """

        if dict[path] is not None:
            if not isinstance(dict[path], str):
                dict[path] = dict[path].as_posix()
        
        return dict