import tifffile
import napari.types
from skimage.draw import disk
import numpy as np

def create_annotation_cell3d() -> list[napari.types.LayerDataTuple]:

    
    shape = (256,256)
    labels = np.zeros(shape, dtype=np.uint8)
    rr, cc = disk((135, 36), 6, shape=shape)
    labels[rr, cc] = 1
    rr, cc = disk((116, 93), 6, shape=shape)
    labels[rr, cc] = 2

    return [(labels, {}, 'labels')]
    
    