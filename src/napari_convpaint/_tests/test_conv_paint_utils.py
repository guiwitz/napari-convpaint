from napari_convpaint import conv_paint_utils
from napari_convpaint import conv_paint_nnlayers
from napari_convpaint.convpaint_sample import create_annotation_cell3d
from torch.nn.modules.container import Sequential
import numpy as np
import skimage

def test_hook_model():
    
    model = conv_paint_nnlayers.Hookmodel(model_name='vgg16')
    layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
          'features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
         ]
    model.register_hooks(selected_layers=layers)
    
    model.features_per_layer
    assert isinstance(model.features_per_layer, list), "Expect list of number of features"
    assert model.features_per_layer[0] == 64, f'Number of features expected from first layer is 64 but got {model.features_per_layer[0]}'
    assert model.features_per_layer[1] == 256, f'Number of features expected from first layer is 256 but got {model.features_per_layer[1]}'

def test_filter_image():

    model = conv_paint_nnlayers.Hookmodel(model_name='vgg16')
    layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
          'features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
         ]
    model.register_hooks(selected_layers=layers)

    image = skimage.data.cells3d()
    image = image[30,1]
    labels = create_annotation_cell3d()[0][0]
    image = image[60:188, 0:128]
    labels = labels[60:188, 0:128]

    features, targets = conv_paint_utils.get_features_current_layers(
        model=model, image=image, annotations=labels, use_min_features=False,tile_annotations=True)
    assert len(features.shape) == 2, f'Expecting dataframe with 2 dims but got {len(features.shape)}'
    assert features.shape[0] == 218, f'Expecting 218 annotated pixels but got {features.shape[0]}'
    assert features.shape[1] == 320, f'Expecting 320 features but got {features.shape[1]}'

    #disable annotation tiling should lead to the same results
    features, targets = conv_paint_utils.get_features_current_layers(
    model=model, image=image, annotations=labels, use_min_features=False,tile_annotations=False)
    assert len(features.shape) == 2, f'Expecting dataframe with 2 dims but got {len(features.shape)}'
    assert features.shape[0] == 218, f'Expecting 218 annotated pixels but got {features.shape[0]}'
    assert features.shape[1] == 320, f'Expecting 320 features but got {features.shape[1]}'