from napari_convpaint.conv_paint import ConvPaintWidget
from napari_convpaint.utils import generate_synthetic_square, generate_synthetic_circle_annotation
import numpy as np
import os

def test_3d_single_channel(make_napari_viewer, capsys):
    """A 3D stack where 3rd dim is z or t"""

    multid_3d = np.stack([(i+1) * np.random.randint(0, 255, (100,100)) for i in range(3)], axis=0)
    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(multid_3d)

    my_widget.add_annotation_layer()

    assert viewer.layers['annotations'].data.ndim == 3, "Annotation layer should be 3D"

    # get stats and check dimensions and values
    my_widget.get_image_stats()

    # check that mean is single number ~127
    assert my_widget.image_mean.shape == ()
    assert 250 < my_widget.image_mean < 260

    normalized = my_widget.get_selectedlayer_data()

    # check that normalized image has correct dims
    assert normalized.shape == (3,100,100)

    # check that mean by channel is [<0, 0, >0]
    assert normalized[0].mean() < 0
    np.testing.assert_almost_equal(normalized[1].mean(), 0, decimal=1)
    assert normalized[2].mean() > 0.1

    # switch to by channel normalization
    my_widget.radio_normalize_by_image.setChecked(True)
    assert my_widget.image_mean is None, "Bad reset of image stats"
    my_widget.get_image_stats()
    assert my_widget.image_mean.shape == (3,1,1)

    normalized = my_widget.get_selectedlayer_data()
    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2)), np.zeros((3)))

def test_3d_multi_channel(make_napari_viewer, capsys):
    """A 3D stack where 3rd dim is channel"""
    
    multid_3d = np.stack([(i+1) * np.random.randint(0, 255, (100,100)) for i in range(3)], axis=0)
    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(multid_3d)

    my_widget.radio_multi_channel.setChecked(True)
    my_widget.add_annotation_layer()

    # check that stack normalization is off
    assert my_widget.radio_normalized_over_stack.isEnabled() == False

    assert viewer.layers['annotations'].data.ndim == 2, "Annotation layer should be 2D"

    # get stats and check dimensions and values
    my_widget.get_image_stats()

    # check that mean is single number ~127
    assert my_widget.image_mean.shape == (3,1,1)

    # check that mean is taken per channel
    assert 127-10 < my_widget.image_mean.flatten()[0] < 127+10
    assert 3*127-10 < my_widget.image_mean.flatten()[2] < 3*127+10

    # check that normalization per channel gives 0
    normalized = my_widget.get_selectedlayer_data()
    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2)), np.zeros((3)))

def test_RGB(make_napari_viewer, capsys):
    """A 3D stack where 3rd dim is channel"""
    
    multid_rgb = np.stack([(i+1) * np.random.randint(0, 255, (100,100)) for i in range(3)], axis=2)
    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(multid_rgb)

    my_widget.add_annotation_layer()

    # check that stack normalization is off
    assert my_widget.radio_normalized_over_stack.isEnabled() == False

    assert viewer.layers['annotations'].data.ndim == 2, "Annotation layer should be 2D"

    # get stats and check dimensions and values
    my_widget.get_image_stats()

    # check that mean is single number ~127
    assert my_widget.image_mean.shape == (3,1,1)

    # check that mean is taken per channel
    assert 127-10 < my_widget.image_mean.flatten()[0] < 127+10
    assert 3*127-10 < my_widget.image_mean.flatten()[2] < 3*127+10

    # check that normalization per channel gives 0
    normalized = my_widget.get_selectedlayer_data()
    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2)), np.zeros((3)))

def test_4d_image(make_napari_viewer, capsys):
    """For a 4D data (C, T, X, Y) check that normalization is done properly
    per channel and per stack or image"""

    multid_c_t = np.stack([i*np.random.randint(0, 255, (10,40,50)) for i in range(1,4)], axis=0).astype(float)

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(multid_c_t)
    my_widget.add_annotation_layer()

    assert viewer.layers['annotations'].data.ndim == 3, "Annotation layer should be 3D"
    
    # get stats and check dimensions and values
    my_widget.get_image_stats()
    assert my_widget.image_mean.ndim == 4, f"Wrong stats dims, expected 4 got {my_widget.image_mean.ndim}"
    assert my_widget.image_mean.shape == (3,1,1,1), f"Wrong number of values, expected (3,1,1,1) got {my_widget.image_mean.shape}"

    assert 255/2-10 < my_widget.image_mean.flatten()[0] < 255/2+10, "Mean over channels seems wrong"
    assert 255-10 < my_widget.image_mean.flatten()[1] < 255+10, "Mean over channels seems wrong"
    assert (3*255)/2-10 < my_widget.image_mean.flatten()[2] < (3*255)/2+10, "Mean over channels seems wrong"

    # make image time dependent
    for i in range(10):
        viewer.layers['multid_c_t'].data[:,i] = multid_c_t[:,i] *np.exp(-i)
    # update stats and normalize
    my_widget.get_image_stats()
    normalized = my_widget.get_selectedlayer_data()

    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2,3)), np.zeros((3)))

    # check that first time point is above 0 and last one below
    assert normalized[0].mean(axis=(1,2))[0] > 1.5
    assert normalized[0].mean(axis=(1,2))[-1] < 0

    # switch normalization
    # check stats reset
    # check that dims are correct
    my_widget.radio_normalize_by_image.setChecked(True)
    assert my_widget.image_mean is None, "Bad reset of image stats"
    my_widget.get_image_stats()

    assert my_widget.image_mean.ndim == 4, f"Wrong stats dims, expected 4 got {my_widget.image_mean.ndim}"
    assert my_widget.image_mean.shape == (3,10,1,1), f"Wrong number of values, expected (3,10,1,1) got {my_widget.image_mean.shape}"
    normalized = my_widget.get_selectedlayer_data()

    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2,3)), np.zeros((3)))

    # check that each single time point of a channel has mean 0 as expected when normalizing by plane
    np.testing.assert_array_almost_equal(normalized[0].mean(axis=(1,2)), np.zeros(10))

def test_RGBT_image(make_napari_viewer):

    # create time varying RGB time lapse
    multid_rgb_t = np.stack([i*np.random.randint(0, 255, (10,40,50)) for i in range(1,4)], axis=-1)#.astype(float)
    for i in range(10):
        multid_rgb_t[i] = (multid_rgb_t[i] * np.exp(-i/10)).astype(np.uint8)

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(multid_rgb_t)
    my_widget.add_annotation_layer()

    # check that for a time lapse RGB, annotations are 3D
    assert viewer.layers['annotations'].data.ndim == 3, "Annotation layer should be 3D"

    my_widget.get_image_stats()

    # check that default stack normalization generates one mean per RGB channel
    my_widget.image_mean.ndim == 4, f"Wrong stats dims, expected 4 got {my_widget.image_mean.ndim}"
    assert my_widget.image_mean.shape == (3,1,1,1), f"Wrong number of values, expected (3,1,1,1) got {my_widget.image_mean.shape}"

    normalized = my_widget.get_selectedlayer_data()

    # check that mean of per channel normalized stacks is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2,3)), np.zeros((3)))

    # check that first time point is above 0 and last one below
    assert normalized[0].mean(axis=(1,2))[0] > 0.5
    assert normalized[0].mean(axis=(1,2))[-1] < 0

    # switch to by image normalization
    my_widget.radio_normalize_by_image.setChecked(True)
    assert my_widget.image_mean is None, "Bad reset of image stats"
    my_widget.get_image_stats()

    normalized = my_widget.get_selectedlayer_data()

    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2,3)), np.zeros((3)))

    # check that each single time point of a channel has mean 0 as expected when normalizing by plane
    np.testing.assert_array_almost_equal(normalized[0].mean(axis=(1,2)), np.zeros(10))

