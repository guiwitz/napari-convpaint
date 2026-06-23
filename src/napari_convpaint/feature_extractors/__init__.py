from .dino import Dinov2Features
from .dinov3 import Dinov3Features
from .dino_jafar import DinoJafarFeatures
from .gaussian import GaussianFeatures
from .nnlayers import Hookmodel
from .combo_fe import ComboFeatures

# Optional imports
try:
    from .ilastik import IlastikFeatures
except ImportError as e:
    pass
try:
    from .cellpose import CellposeFeatures
except ImportError as e:
    pass