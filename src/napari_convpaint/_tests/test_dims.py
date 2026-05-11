from napari_convpaint.convpaint_widget import ConvpaintWidget
import numpy as np

def test_3d_single_channel(make_napari_viewer, capsys):
    """A 3D stack where 3rd dim is z or t"""

    multid_3d = np.stack([(i+1) * np.random.randint(0, 255, (100,100)) for i in range(3)], axis=0)
    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(multid_3d)

    # my_widget.rgb_img = False
    my_widget.cp_model.set_params(channel_mode='single')
    my_widget._on_add_annot_layer()

    assert viewer.layers['annotations'].data.ndim == 3, "Annotations layer should be 3D"

    # get stats and check dimensions and values
    my_widget.radio_normalize_over_stack.setChecked(True)
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)

    # check that mean is single number ~127
    assert my_widget.image_mean.shape == ()
    assert 250 < my_widget.image_mean < 260

    normalized = my_widget._get_data_channel_first_norm(img)

    # check that normalized image has correct dims
    assert normalized.shape == (3,100,100)

    # check that mean by channel is [<0, 0, >0]
    assert normalized[0].mean() < 0
    np.testing.assert_almost_equal(normalized[1].mean(), 0, decimal=1)
    assert normalized[2].mean() > 0.1

    # switch to by channel normalization
    my_widget.radio_normalize_by_image.setChecked(True)
    assert my_widget.image_mean is None, "Bad reset of image stats"
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)
    assert my_widget.image_mean.shape == (3,1,1)

    normalized = my_widget._get_data_channel_first_norm(img)
    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2)), np.zeros((3)))

def test_3d_multi_channel(make_napari_viewer, capsys):
    """A 3D stack where 3rd dim is channel"""
    
    multid_3d = np.stack([(i+1) * np.random.randint(0, 255, (100,100)) for i in range(3)], axis=0)
    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(multid_3d)

    # my_widget.rgb_img = False
    my_widget.cp_model.set_params(channel_mode='multi')
    my_widget._on_add_annot_layer()

    # check that stack normalization is off
    assert my_widget.radio_normalize_over_stack.isEnabled() == False

    assert viewer.layers['annotations'].data.ndim == 2, "Annotations layer should be 2D"

    # get stats and check dimensions and values
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)

    # check that mean is single number ~127
    assert my_widget.image_mean.shape == (3,1,1)

    # check that mean is taken per channel
    assert 127-10 < my_widget.image_mean.flatten()[0] < 127+10
    assert 3*127-10 < my_widget.image_mean.flatten()[2] < 3*127+10

    # check that normalization per channel gives 0
    normalized = my_widget._get_data_channel_first_norm(img)
    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2)), np.zeros((3)))

def test_RGB(make_napari_viewer, capsys):
    """A 3D stack where 3rd dim is channel"""
    
    side_len = 1000
    multid_rgb = np.stack([np.random.randint(0, 255, (side_len, side_len), dtype=np.uint8) for i in range(3)], axis=2)
    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(multid_rgb)

    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')
    my_widget._on_add_annot_layer()

    # check that stack normalization is off
    assert my_widget.radio_normalize_over_stack.isEnabled() == False

    assert viewer.layers['annotations'].data.ndim == 2, "Annotations layer should be 2D"

    # get stats and check dimensions and values
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)

    # check that mean is single number in each channel
    assert my_widget.image_mean.shape == (3,1,1)

    # check that mean is taken per channel
    assert 127-10 < my_widget.image_mean.flatten()[0] < 127+10
    assert 127-10 < my_widget.image_mean.flatten()[1] < 127+10
    assert 127-10 < my_widget.image_mean.flatten()[2] < 127+10

    # check that mean over each full channel is according to imagenet
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = my_widget._get_data_channel_first_norm(img)
    channel_means = normalized.mean(axis=(1,2))
    np.testing.assert_array_almost_equal(channel_means, (0.5 - imagenet_mean) / imagenet_std, decimal=2)
    assert normalized.shape == (3, side_len, side_len)
    assert normalized.dtype == np.float32
    assert ((0 - imagenet_mean < channel_means) & (channel_means < 1 - imagenet_mean)).all()  # means should be in this range
    # np.testing.assert_allclose(normalized.mean(axis=(1,2)), - imagenet_mean / imagenet_std, rtol=0.2)

def test_RGBA(make_napari_viewer, capsys):
    """RGBA images (4 channels) should be treated as RGB (alpha stripped).
    Napari loads RGBA as rgb=True, ndim=2, but data.shape[-1]=4."""

    side_len = 100
    # Create RGBA image (H, W, 4) with uint8 so napari sets rgb=True
    rgba = np.random.randint(0, 255, (side_len, side_len, 4), dtype=np.uint8)
    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(rgba)

    # Napari should detect this as RGB
    assert viewer.layers['rgba'].rgb == True
    assert viewer.layers['rgba'].ndim == 2
    assert viewer.layers['rgba'].data.shape == (side_len, side_len, 4)

    # Convpaint should handle it as RGB (stripping alpha)
    my_widget.cp_model.set_params(channel_mode='rgb')
    img = my_widget._get_selected_img()
    data_dims = my_widget._get_data_dims(img.data, img.ndim)
    assert data_dims == '2D_RGB', f"Expected '2D_RGB' but got '{data_dims}'"

    # Annotations should be 2D
    my_widget._on_add_annot_layer()
    assert viewer.layers['annotations'].data.ndim == 2

    # Channel-first data should have 3 channels (alpha stripped)
    data_cf = my_widget._get_data_channel_first(img.data, img.ndim)
    assert data_cf.shape == (3, side_len, side_len), f"Expected (3, {side_len}, {side_len}) but got {data_cf.shape}"

    # Stats should work with 3 channels
    my_widget._compute_image_stats(img)
    assert my_widget.image_mean.shape == (3, 1, 1)

    # Normalization should work
    normalized = my_widget._get_data_channel_first_norm(img)
    assert normalized.shape == (3, side_len, side_len)

def test_4d_image(make_napari_viewer, capsys):
    """For a 4D data (C, T, X, Y) check that normalization is done properly
    per channel and per stack or image"""

    multid_c_t = np.stack([i*np.random.randint(0, 255, (10,40,50)) for i in range(1,4)], axis=0).astype(float)

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(multid_c_t)

    # my_widget.rgb_img = False
    my_widget.cp_model.set_params(channel_mode='multi')
    my_widget._on_add_annot_layer()

    assert viewer.layers['annotations'].data.ndim == 3, "Annotations layer should be 3D"
    
    # get stats and check dimensions and values
    my_widget.radio_normalize_over_stack.setChecked(True)
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)
    assert my_widget.image_mean.ndim == 4, f"Wrong stats dims, expected 4 got {my_widget.image_mean.ndim}"
    assert my_widget.image_mean.shape == (3,1,1,1), f"Wrong number of values, expected (3,1,1,1) got {my_widget.image_mean.shape}"

    assert 255/2-10 < my_widget.image_mean.flatten()[0] < 255/2+10, "Mean over channels seems wrong"
    assert 255-10 < my_widget.image_mean.flatten()[1] < 255+10, "Mean over channels seems wrong"
    assert (3*255)/2-10 < my_widget.image_mean.flatten()[2] < (3*255)/2+10, "Mean over channels seems wrong"

    # make image time dependent
    for i in range(10):
        viewer.layers['multid_c_t'].data[:,i] = multid_c_t[:,i] *np.exp(-i)
    # update stats and normalize
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)
    normalized = my_widget._get_data_channel_first_norm(img)

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
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)

    assert my_widget.image_mean.ndim == 4, f"Wrong stats dims, expected 4 got {my_widget.image_mean.ndim}"
    assert my_widget.image_mean.shape == (3,10,1,1), f"Wrong number of values, expected (3,10,1,1) got {my_widget.image_mean.shape}"
    normalized = my_widget._get_data_channel_first_norm(img)

    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2,3)), np.zeros((3)))

    # check that each single time point of a channel has mean 0 as expected when normalizing by plane
    np.testing.assert_array_almost_equal(normalized[0].mean(axis=(1,2)), np.zeros(10))

'''
def test_RGBT_image(make_napari_viewer, capsys):

    # create time varying RGB time lapse
    steps = 5
    multid_rgb_t = np.stack([i*np.random.randint(0, 255, (steps,40,50)) for i in range(1,4)], axis=-1)#.astype(float)
    for i in range(steps):
        multid_rgb_t[i] = (multid_rgb_t[i] * np.exp(-i/steps)).astype(np.uint8)

    # UNTIL HERE: all pass

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(multid_rgb_t)

    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')
    my_widget._on_add_annot_layer()

    # UNTIL HERE: py3.12 fails

    # check that for a time lapse RGB, annotations are 3D
    assert viewer.layers['annotations'].data.ndim == 3, "Annotation layer should be 3D"

    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)

    # check that stack normalization generates one mean per RGB channel
    my_widget.radio_normalize_over_stack.setChecked(True)
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)
    my_widget.image_mean.ndim == 4, f"Wrong stats dims, expected 4 got {my_widget.image_mean.ndim}"
    assert my_widget.image_mean.shape == (3,1,1,1), f"Wrong number of values, expected (3,1,1,1) got {my_widget.image_mean.shape}"

    # UNTIL HERE: only py3.12 fails

    normalized = my_widget._get_data_channel_first_norm(img)

    # check that mean of per channel normalized stacks is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2,3)), np.zeros((3)))

    # check that first time point is above 0 and last one below
    assert normalized[0].mean(axis=(1,2))[0] > 0.5
    assert normalized[0].mean(axis=(1,2))[-1] < 0

    # switch to by image normalization
    my_widget.radio_normalize_by_image.setChecked(True)
    assert my_widget.image_mean is None, "Bad reset of image stats"
    img = my_widget._get_selected_img(check=True)
    my_widget._compute_image_stats(img)

    normalized = my_widget._get_data_channel_first_norm(img)

    # check that mean over each full channel is 0
    np.testing.assert_array_almost_equal(normalized.mean(axis=(1,2,3)), np.zeros((3)))

    # check that each single time point of a channel has mean 0 as expected when normalizing by plane
    np.testing.assert_array_almost_equal(normalized[0].mean(axis=(1,2)), np.zeros(steps))
    '''

def test_3d_stack_training_with_memory_mode(make_napari_viewer, capsys):
    """Test that training on a 3D stack with memory_mode=True and a single
    string img_id works. This is the bug reported in GitHub issues where
    dask arrays (or 3D stacks in general) caused:
    ValueError: Image IDs must be passed as a list with the same length as the data
    """
    from napari_convpaint.convpaint_model import ConvpaintModel

    # Create a small 3D stack (Z, H, W) and annotations with 2 classes
    np.random.seed(42)
    stack = np.random.randint(0, 255, (5, 50, 50)).astype(np.float32)
    annot = np.zeros((5, 50, 50), dtype=np.uint8)
    annot[0, 10:20, 10:20] = 1  # class 1 on first plane
    annot[0, 30:40, 30:40] = 2  # class 2 on first plane

    model = ConvpaintModel(alias="gaussian")
    model.set_params(channel_mode='single')

    # Training with memory_mode=True and a single string img_id should work
    model.train(stack, annot, memory_mode=True, img_ids='test_image')

    assert model.classifier is not None, "Classifier should be trained"

def test_3d_stack_training_with_dask_input(make_napari_viewer, capsys):
    """Test that training with a dask array input (as happens in the widget
    for large 3D stacks) works correctly with memory_mode and single img_id.
    Reproduces the exact bug from the user reports where dask arrays were not
    recognized as single inputs."""
    import dask.array as da
    from napari_convpaint.convpaint_model import ConvpaintModel

    np.random.seed(42)
    stack_np = np.random.randint(0, 255, (5, 50, 50)).astype(np.float64)
    # Create a dask array to simulate the widget's normalization output
    stack_dask = da.from_array(stack_np, chunks=(1, 50, 50))

    annot = np.zeros((5, 50, 50), dtype=np.uint8)
    annot[0, 10:20, 10:20] = 1
    annot[0, 30:40, 30:40] = 2

    model = ConvpaintModel(alias="gaussian")
    model.set_params(channel_mode='single')

    # This used to fail with:
    # ValueError: Image IDs must be passed as a list with the same length as the data
    model.train(stack_dask, annot, memory_mode=True, img_ids='dask_image')

    assert model.classifier is not None, "Classifier should be trained"
def test_incompatible_image_on_startup(make_napari_viewer):
    """Test that the plugin loads without crashing when an incompatible image
    (e.g. 5D) is already present in the viewer. The plugin should handle
    unsupported image dimensions gracefully instead of crashing during init."""

    # 5D image (e.g. TCZYX) - not supported by convpaint
    image_5d = np.random.randint(0, 255, (2, 3, 5, 40, 50)).astype(float)
    viewer = make_napari_viewer()
    viewer.add_image(image_5d)

    # Plugin should load without crashing even though a 5D image is selected
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    assert my_widget is not None

    # The widget should still be functional - adding a compatible image later should work
    image_2d = np.random.randint(0, 255, (100, 100)).astype(float)
    viewer.add_image(image_2d)
    my_widget._on_select_layer()
    img = my_widget._get_selected_img()
    assert img is not None

