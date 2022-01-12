from napari_convpaint import conv_paint_utils
from torch.nn.modules.container import Sequential
import numpy as np

def test_load_model():
    model = conv_paint_utils.load_nn_model()
    isinstance(model,Sequential)
    
    first_layer = model.state_dict()['conv1.weight']
    assert first_layer.shape[0] == 64, f'Number of filter expected to be 64 but found {first_layer.shape[0]}'
    assert first_layer.shape[1] == 1, f'Expect 1 channel, found {first_layer.shape[1]}'

def test_filter_image():

    model = conv_paint_utils.load_nn_model()
    image = np.random.randint(0,255,(10,10))
    all_scales = conv_paint_utils.filter_image(image, model, scalings=[1,2])
    assert len(all_scales) == 2
    assert all_scales[0].shape == (1, 64, 10, 10)