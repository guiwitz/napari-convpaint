
from __future__ import annotations
from dataclasses import dataclass, field
import dataclasses
from pathlib import Path
import yaml

@dataclass
class Param:
    """Object storing relevant information regarding the processing,
    e.g. the window size, the analyzed signal, the type of segmentation used.
    """

    random_forest: str = None
    model_name: str = None
    model_layers: list[str] = None
    scalings: list[int] = None
    order: int = None
    use_min_features: bool = None
    image_downsample: int = None
    normalize: int = None
    multi_channel_training: bool = None
    use_cuda: bool = None



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
            dict_to_save = self.convert_path(dict_to_save, 'random_forest')
            
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