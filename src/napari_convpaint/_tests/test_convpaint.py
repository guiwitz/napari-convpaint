from napari_convpaint.convpaint_widget import ConvpaintWidget
from napari_convpaint.testing_data import generate_synthetic_square, generate_synthetic_circle_annotation
import numpy as np
import os
from PIL import Image
import napari
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import pytest

def compute_precision_recall(ground_truth, recovered):

    all_precision = []
    all_recall = []

    for class_val in np.unique(ground_truth):
            tp = np.sum((recovered == class_val) & (ground_truth == class_val))
            fp = np.sum((recovered == class_val) & (ground_truth != class_val))
            fn = np.sum((recovered != class_val) & (ground_truth == class_val))
            tn = np.sum((recovered != class_val) & (ground_truth != class_val))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            all_precision.append(precision)
            all_recall.append(recall)

    return np.mean(all_precision), np.mean(all_recall)

    # tp = np.sum((recovered==2) & (ground_truth==2))# / np.sum(ground_truth == 1)
    # fp = np.sum((recovered==2) & (ground_truth!=2))#/ np.sum(ground_truth == 1)
    # fn = np.sum((recovered!=2) & (ground_truth==2))
    # precision = tp /  (tp + fp)
    # recall = tp / (tp + fn)
    # return precision, recall

def test_add_layers(make_napari_viewer, capsys):
    """Test that annotations and prediction layers are added correctly"""
    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(np.random.random((100, 100)))
    my_widget._on_add_annot_layer()

    assert 'annotations' in viewer.layers
    # assert 'segmentation' in viewer.layers

def test_annotations_layer_dims(make_napari_viewer, capsys):
    """Check that dimensions of annotations layer match image layer"""

    viewer = make_napari_viewer()
    # viewer = napari.Viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(np.random.random((100, 100, 3)))
    my_widget._on_add_annot_layer()
    assert "annotations" in viewer.layers
    assert viewer.layers['annotations'].data.shape == (100, 100)
    viewer.close()

    viewer = make_napari_viewer()
    # viewer = napari.Viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(np.random.random((3, 100, 100)))
    my_widget._on_add_annot_layer()
    assert "annotations" in viewer.layers
    assert viewer.layers['annotations'].data.shape == (3, 100, 100)
    viewer.close()

def test_correct_model_rgb(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()

    viewer.add_image(im, name='sample')
    my_widget._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')
    my_widget._on_train()
    assert my_widget.qcombo_fe_type.currentText() == 'vgg16', "Model type not updated correctly"

def test_correct_model_2d(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()

    viewer.add_image(im[:,:,0], name='sample')
    my_widget._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot
    # my_widget.rgb_img = False
    my_widget.cp_model.set_params(channel_mode='single')
    my_widget._on_train()
    assert my_widget.qcombo_fe_type.currentText() == 'vgg16', "Model type not updated correctly"

def test_rgb_prediction(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture

    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im)
    my_widget._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')
    my_widget._on_train()
    my_widget._on_predict()

    recovered = viewer.layers['segmentation'].data
    precision, recall = compute_precision_recall(ground_truth, recovered)
    
    assert precision > 0.9, f"Precision: {precision}, too low"
    assert recall > 0.9, f"Recall: {recall}, too low"

def test_multi_channel_prediction(make_napari_viewer, capsys):
    """Check that prediction is bad when disabling multi-channel training for
    image with signal in red channel and large noise in green channel"""

    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))
    im[:,:,1] = np.random.randint(0,200,(252,252))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(np.moveaxis(im,2,0))
    my_widget._on_add_annot_layer()
    my_widget.radio_single_channel.setChecked(True)
    viewer.layers['annotations'].data[0,:,:] = im_annot
    my_widget._on_train()
    my_widget._on_predict()

    recovered = viewer.layers['segmentation'].data[1]
    precision, recall = compute_precision_recall(ground_truth, recovered)
    
    assert precision < 0.8, f"Precision: {precision} is too high for non multi-channel training"


def test_save_model(make_napari_viewer, capsys):
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im)
    my_widget._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')
    my_widget._on_train()
    my_widget._on_predict()

    models_dir = os.path.join(os.path.dirname(__file__), 'model_dir')
    os.makedirs(models_dir, exist_ok=True)
    my_widget._on_save_model(save_file=os.path.join(models_dir, 'test_model.pkl'))
    assert os.path.exists(os.path.join(models_dir, 'test_model.pkl'))


def test_load_model(make_napari_viewer, capsys):
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im)
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')

    models_dir = os.path.join(os.path.dirname(__file__), 'model_dir')
    os.makedirs(models_dir, exist_ok=True)
    my_widget._on_load_model(save_file=os.path.join(models_dir, 'test_model.pkl'))  # Changed to .pkl
    my_widget._on_predict()

    # recovered = viewer.layers['segmentation'].data[ground_truth==1]
    # tp = np.sum(recovered == 2)
    # fp = np.sum(recovered == 1)
    # fn = np.sum(ground_truth == 1) - tp
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    recovered = viewer.layers['segmentation'].data
    precision, recall = compute_precision_recall(ground_truth, recovered)
    assert precision > 0.8, f"Precision: {precision}, too low"
    assert recall > 0.8, f"Recall: {recall}, too low"


def test_save_model_dino(make_napari_viewer, capsys):
    # im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    # im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))
    imgs_dir = os.path.join(os.path.dirname(__file__), 'test_imgs')
    im = np.array(Image.open(os.path.join(imgs_dir, '0000_img.png')))
    im_annot = np.array(Image.open(os.path.join(imgs_dir, '0000_scribbles_all_01500_w3.png')))
    ground_truth = np.array(Image.open(os.path.join(imgs_dir, '0000_ground_truth.png')))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im)
    my_widget._on_add_annot_layer()
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')

    # Simulate selecting the Dino model from the dropdown
    my_widget.qcombo_fe_type.setCurrentText('dinov2_vits14_reg')
    assert my_widget.qcombo_fe_type.currentText() == 'dinov2_vits14_reg'
    
    cp_model = my_widget.cp_model
    cp_model._param.fe_scalings = [1]
    cp_model._param.fe_order = 0  # Set interpolation order to 0
    cp_model._param.fe_name = 'dinov2_vits14_reg'
    # cp_model._param.fe_use_gpu = False
    cp_model._param.fe_use_min_features = False
    cp_model._param.tile_annotations = False
    cp_model._param.image_downsample = 1
    cp_model._param.normalize = 1 #no normalization (button id)
    my_widget._update_gui_from_params()
    my_widget.set_fe_btn.click()  # Load the model
    assert cp_model._param.fe_scalings == [1]
    assert cp_model._param.fe_name == 'dinov2_vits14_reg'

    viewer.layers['annotations'].data[...] = im_annot
    my_widget._on_train()  # Update the classifier with the new parameters
    my_widget._on_predict()
    models_dir = os.path.join(os.path.dirname(__file__), 'model_dir')
    os.makedirs(models_dir, exist_ok=True)
    model_path_dino = os.path.join(models_dir, 'test_model_dino.pkl')
    my_widget._on_save_model(save_file=model_path_dino)
    assert my_widget.qcombo_fe_type.currentText() == 'dinov2_vits14_reg'
    assert os.path.exists(model_path_dino)


def test_cross_attention_matches_mha_reference():
    """Verify that the manual attention score computation in CrossAttention
    produces identical results to the original nn.MultiheadAttention.forward()
    run on CPU."""
    from napari_convpaint.jafar.layers.attentions import CrossAttention

    torch.manual_seed(42)
    query_dim, key_dim, value_dim, num_heads = 128, 128, 384, 4
    B, N_q, N_k = 1, 64, 16

    ca = CrossAttention(query_dim, key_dim, value_dim, num_heads).eval()

    query = torch.randn(B, N_q, query_dim)
    key   = torch.randn(B, N_k, key_dim)
    value = torch.randn(B, N_k, value_dim)

    # --- Reference: original nn.MultiheadAttention on CPU ---
    with torch.no_grad():
        q_normed = ca.norm_q(query)
        k_normed = ca.norm_k(key)
        v_normed = ca.norm_v(value)
        _, ref_scores = ca.attention(
            q_normed, k_normed, v_normed, average_attn_weights=True
        )
        ref_output = einsum("b i j, b j d -> b i d", ref_scores, value)

    # --- New implementation ---
    with torch.no_grad():
        new_output, new_scores = ca(query, key, value)

    assert torch.allclose(ref_scores, new_scores, atol=1e-6), \
        f"Attention scores differ: max delta = {(ref_scores - new_scores).abs().max():.2e}"
    assert torch.allclose(ref_output, new_output, atol=1e-6), \
        f"Attention output differs: max delta = {(ref_output - new_output).abs().max():.2e}"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS device not available"
)
def test_dino_jafar_small_mps_rgb(make_napari_viewer, capsys):
    """Test dino_jafar_small on MPS with an RGB image.

    Reproduces a crash where MPS fails with:
    [MPSNDArrayDescriptor sliceDimension:withSubrange:] failed assertion
    """
    imgs_dir = os.path.join(os.path.dirname(__file__), 'test_imgs')
    im = np.array(Image.open(os.path.join(imgs_dir, '0000_img.png')))
    im_annot = np.array(Image.open(os.path.join(imgs_dir, '0000_scribbles_all_01500_w3.png')))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im)
    my_widget._on_add_annot_layer()
    my_widget.cp_model.set_params(channel_mode='rgb')

    # Select dino_jafar_small
    my_widget.qcombo_fe_type.setCurrentText('dino_jafar_small')
    assert my_widget.qcombo_fe_type.currentText() == 'dino_jafar_small'

    cp_model = my_widget.cp_model
    cp_model._param.fe_scalings = [1]
    cp_model._param.fe_order = 0
    cp_model._param.fe_name = 'dino_jafar_small'
    # cp_model._param.fe_use_gpu = True  # Force MPS
    cp_model.lock_device("gpu", "fe") # New way to force GPU constantly...
    cp_model._param.fe_use_min_features = False
    cp_model._param.tile_annotations = False
    cp_model._param.image_downsample = 1
    cp_model._param.normalize = 1
    my_widget._update_gui_from_params()
    my_widget.set_fe_btn.click()

    viewer.layers['annotations'].data[...] = im_annot
    my_widget._on_train()
    my_widget._on_predict()

    recovered = viewer.layers['segmentation'].data
    precision, recall = compute_precision_recall(
        np.array(Image.open(os.path.join(imgs_dir, '0000_ground_truth.png'))),
        recovered
    )
    assert precision > 0.7, f"Precision: {precision}, too low"
    assert recall > 0.7, f"Recall: {recall}, too low"


def test_load_model_dino(make_napari_viewer, capsys):
    # im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    # im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))
    imgs_dir = os.path.join(os.path.dirname(__file__), 'test_imgs')
    im = np.array(Image.open(os.path.join(imgs_dir, '0000_img.png')))
    im_annot = np.array(Image.open(os.path.join(imgs_dir, '0000_scribbles_all_01500_w3.png')))
    ground_truth = np.array(Image.open(os.path.join(imgs_dir, '0000_ground_truth.png')))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()

    viewer.add_image(im)
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')

    # Load the Dino model
    models_dir = os.path.join(os.path.dirname(__file__), 'model_dir')
    os.makedirs(models_dir, exist_ok=True)
    my_widget._on_load_model(save_file=os.path.join(models_dir, 'test_model_dino.pkl'))
    # Ensure the model type is set correctly after loading
    assert my_widget.qcombo_fe_type.currentText() == 'dinov2_vits14_reg'
    my_widget._on_predict()

    # recovered = viewer.layers['segmentation'].data[ground_truth==1]
    # tp = np.sum(recovered == 2)
    # fp = np.sum(recovered == 1)
    # fn = np.sum(ground_truth == 1) - tp
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    recovered = viewer.layers['segmentation'].data
    precision, recall = compute_precision_recall(ground_truth, recovered)
    assert precision > 0.8, f"Precision: {precision}, too low"
    assert recall > 0.8, f"Recall: {recall}, too low"
    

def test_save_and_load_vgg16_models(make_napari_viewer, capsys):
    # Setup synthetic data
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im)
    my_widget._on_add_annot_layer()
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')

    models_dir = os.path.join(os.path.dirname(__file__), 'model_dir')
    os.makedirs(models_dir, exist_ok=True)
    # Create and save the first model with scales [1]
    my_widget.qcombo_fe_type.setCurrentText('vgg16')
    my_widget.fe_scaling_factors.setCurrentText('[1]')
    my_widget.set_fe_btn.setEnabled(True)
    my_widget.set_fe_btn.click()
    assert my_widget.cp_model._param.fe_scalings == [1]
    viewer.layers['annotations'].data[...] = im_annot
    my_widget._on_train()
    my_widget._on_predict()
    model_path_1 = os.path.join(models_dir, 'test_model_vgg16_scale_1.pkl')
    my_widget._on_save_model(save_file=model_path_1)
    assert os.path.exists(model_path_1)

    # Create and save the second model with scales [1, 2, 3, 4]. Change in the UI:
    my_widget.fe_scaling_factors.setCurrentText('[1,2,4,8]')
    my_widget.set_fe_btn.click()
    assert my_widget.cp_model._param.fe_scalings == [1, 2, 4, 8]

    my_widget.set_fe_btn.click()
    assert my_widget.cp_model._param.fe_scalings == [1, 2, 4, 8]
    viewer.layers['annotations'].data[...] = im_annot
    my_widget._on_train()
    my_widget._on_predict()
    model_path_2 = os.path.join(models_dir, 'test_model_vgg16_scale_1248.pkl')
    my_widget._on_save_model(save_file=model_path_2)
    assert os.path.exists(model_path_2)

    # Load the second model and predict
    my_widget._on_load_model(save_file=model_path_2)
    assert my_widget.cp_model._param.fe_scalings == [1, 2, 4, 8]
    my_widget._on_predict()
    recovered = viewer.layers['segmentation'].data
    assert np.any(recovered[ground_truth == 1])  # Check if there is any prediction

    # Load the first model and predict
    my_widget._on_load_model(save_file=model_path_1)
    assert my_widget.cp_model._param.fe_scalings == [1]
    my_widget._on_predict()
    recovered = viewer.layers['segmentation'].data
    assert np.any(recovered[ground_truth == 1])  # Check if there is any prediction


# test dino model with different image sizes
def test_dino_model_with_different_image_sizes(make_napari_viewer, capsys):
    sizes = [(140, 140), (100, 100), (120, 120), (28,28)]
    for size in sizes:
        #left side: class 1, right side: class 2 (0 and 255)
        im = np.zeros(size, dtype=np.uint8)
        im[:, size[1]//2:] = 255
        im_annot = np.zeros(size, dtype=np.uint8)
        im_annot[:, size[1]//2:] = 2
        im_annot[:, :size[1]//2] = 1

        viewer = make_napari_viewer()
        my_widget = ConvpaintWidget(viewer)
        my_widget.ensure_init()
        viewer.add_image(im)
        my_widget._on_add_annot_layer()
        # my_widget.rgb_img = False # Assuming only 2D images are generated
        my_widget.cp_model.set_params(channel_mode='single') # Assuming only 2D images are generated

        # Load the Dino model
        my_widget.qcombo_fe_type.setCurrentText('dinov2_vits14_reg')
        # Set the scaling to 1
        my_widget.fe_scaling_factors.setCurrentText('[1]')
        my_widget.set_fe_btn.click()
        #set widget disable tiling annotations (should be done automatically, but just to be sure)
        my_widget.check_tile_annotations.setChecked(False)
        # my_widget._update_params_from_gui() # is done automatically
        viewer.layers['annotations'].data[...] = im_annot
        my_widget._on_train()
        my_widget._on_predict()

        recovered = viewer.layers['segmentation'].data
        
        #check that the shape of the recovered is the same as the annotation
        assert recovered.shape == im_annot.shape


def test_custom_vgg16_layers(make_napari_viewer, capsys):
    # Setup synthetic data
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im, name='sample')
    my_widget._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot
    # my_widget.rgb_img = True
    my_widget.cp_model.set_params(channel_mode='rgb')

    # Create and save the custom vgg16 model with selected layers

    my_widget.qcombo_fe_type.setCurrentText('vgg16')

    # Assuming 'self.fe_layer_selection' is your QListWidget instance
    all_tests = [[0],[0,1]]#,[0,7]]
    for indices_to_select in all_tests:
        # Iterate over the list of indices and select the corresponding items
        my_widget.fe_layer_selection.clearSelection()
        for index in indices_to_select:
            item = my_widget.fe_layer_selection.item(index)
            if item:  # Check if the item exists at that index
                item.setSelected(True)

        my_widget.set_fe_btn.setEnabled(True)
        my_widget.set_fe_btn.click()
        viewer.layers['annotations'].data[...] = im_annot
        my_widget._on_train()

        assert len(my_widget.fe_layer_selection.selectedItems()) == len(indices_to_select)

        #save the model
        models_dir = os.path.join(os.path.dirname(__file__), 'model_dir')
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f'test_model_vgg16_custom_layers_{indices_to_select}.pkl')
        my_widget._on_save_model(save_file=model_path)

    #load the models again and check if the predictions are correct
    for indices_to_select in all_tests:
        model_path = os.path.join(models_dir, f'test_model_vgg16_custom_layers_{indices_to_select}.pkl')
        my_widget._on_load_model(save_file=model_path)
        assert len(my_widget.fe_layer_selection.selectedItems()) == len(indices_to_select)

        my_widget._on_predict()
        recovered = viewer.layers['segmentation'].data
        precision, recall = compute_precision_recall(ground_truth, recovered)
        assert precision > 0.7, f"Precision: {precision}, too low"
        assert recall > 0.7, f"Recall: {recall}, too low"

    assert my_widget.qcombo_fe_type.currentText() == 'vgg16'


def test_3d_single_channel_prediction(make_napari_viewer, capsys):
    """Test train and predict on a 3D stack of single-channel images (3D_single mode)."""

    # Create a 3D stack of 3 single-channel images with a bright square
    im_dims = (100, 100)
    square_dims = (30, 30)
    num_slices = 3

    ims = []
    for i in range(num_slices):
        im, _ = generate_synthetic_square(im_dims=im_dims, square_dims=square_dims, rgb=False)
        ims.append(im)
    im_stack = np.stack(ims, axis=0)  # shape: (3, 100, 100)
    assert im_stack.shape == (num_slices, im_dims[0], im_dims[1])

    # Annotations: label two circles on the first slice
    im_annot_2d = generate_synthetic_circle_annotation(im_dims=im_dims, circle1_xy=(50, 30), circle2_xy=(50, 50))
    im_annot = np.zeros_like(im_stack, dtype=np.uint8)
    im_annot[0] = im_annot_2d

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im_stack)
    my_widget.cp_model.set_params(channel_mode='single')
    my_widget._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot

    # Train with numpy data
    my_widget._on_train()

    my_widget._on_predict()

    # Verify segmentation layer exists and has correct shape
    assert 'segmentation' in viewer.layers, "Segmentation layer not created"
    seg_data = viewer.layers['segmentation'].data
    assert seg_data.shape == im_stack.shape, (
        f"Segmentation shape {seg_data.shape} doesn't match image shape {im_stack.shape}"
    )


def test_3d_single_channel_predict_returns_array():
    """Test that _predict returns arrays (not lists) for single dask array input."""
    import dask.array as da
    from napari_convpaint.convpaint_model import ConvpaintModel

    # Create synthetic data: a single 2D plane (as would come from _get_current_plane_norm)
    im_dims = (100, 100)
    square_dims = (30, 30)
    im, _ = generate_synthetic_square(im_dims=im_dims, square_dims=square_dims, rgb=False)
    im_annot = generate_synthetic_circle_annotation(im_dims=im_dims, circle1_xy=(50, 30), circle2_xy=(50, 50))

    # Train the model with numpy data
    model = ConvpaintModel()
    model.set_params(channel_mode='single')
    model.train(im, im_annot)

    # Predict with a dask array (simulates dask-backed image layer)
    im_dask = da.from_array(im, chunks=im.shape)
    probas, seg = model._predict(im_dask, add_seg=True)

    assert isinstance(seg, np.ndarray), (
        f"_predict should return an array for single input, got {type(seg)}"
    )
    assert seg.shape == im_dims, (
        f"Segmentation shape {seg.shape} should be {im_dims}"
    )


# ---------------------------------------------------------------------------
# Parametrized test: all FE models × image types (2D / RGB)
# ---------------------------------------------------------------------------

ALL_FE_MODELS = [
    'vgg16',
    'efficient_netb0',
    'convnext',
    'dinov2_vits14_reg',
    'dino_jafar_small',
    'gaussian_features',
    'cellpose_backbone',
    'combo_dino_vgg',
    'combo_dino_gauss',
]

# Use 252×252 so dimensions are divisible by 14 (dino patch size)
_IM_DIMS = (252, 252)
_SQ_DIMS = (70, 70)

def _make_image_and_annot(image_type):
    """Return (image, annotation, channel_mode) for a given image type."""
    im_rgb, gt = generate_synthetic_square(im_dims=_IM_DIMS, square_dims=_SQ_DIMS, rgb=True)
    annot = generate_synthetic_circle_annotation(im_dims=_IM_DIMS, circle1_xy=(125, 70), circle2_xy=(125, 125))
    if image_type == '2d':
        return im_rgb[:, :, 0], annot, 'single'
    elif image_type == 'rgb':
        return im_rgb, annot, 'rgb'
    else:
        raise ValueError(f"Unknown image type: {image_type}")


def _fe_available(fe_name):
    """Check whether an FE is actually importable in this environment."""
    try:
        from napari_convpaint.convpaint_model import ConvpaintModel
        if not ConvpaintModel.FE_MODELS_TYPES_DICT:
            ConvpaintModel._init_fe_models_dict()
        return fe_name in ConvpaintModel.FE_MODELS_TYPES_DICT
    except Exception:
        return False


@pytest.mark.parametrize("fe_name", ALL_FE_MODELS)
@pytest.mark.parametrize("image_type", ['2d', 'rgb'])
def test_all_models_train_predict(make_napari_viewer, fe_name, image_type):
    """Train and predict with every FE model on 2D and RGB images."""
    if not _fe_available(fe_name):
        pytest.skip(f"{fe_name} not available in this environment")

    im, annot, channel_mode = _make_image_and_annot(image_type)

    viewer = make_napari_viewer()
    my_widget = ConvpaintWidget(viewer)
    my_widget.ensure_init()
    viewer.add_image(im, name='sample')
    my_widget._on_add_annot_layer()
    my_widget.cp_model.set_params(channel_mode=channel_mode)
    # my_widget.cp_model.lock_device("auto") # Will use mps/cuda if available, otherwise cpu

    my_widget.qcombo_fe_type.setCurrentText(fe_name)
    my_widget.set_fe_btn.click()
    my_widget.device_dropdown.setCurrentText('auto') # Will use mps/cuda if available, otherwise cpu

    viewer.layers['annotations'].data[...] = annot
    my_widget._on_train()
    my_widget._on_predict()

    assert 'segmentation' in viewer.layers, "Segmentation layer not created"
    seg = viewer.layers['segmentation'].data
    assert seg.shape == annot.shape, f"Segmentation shape {seg.shape} != annotations shape {annot.shape}"
    assert np.unique(seg).size > 1, "Segmentation is uniform — model produced no meaningful output"