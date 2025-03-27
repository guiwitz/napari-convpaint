from napari_convpaint.conv_paint_widget import ConvPaintWidget
from napari_convpaint.utils import generate_synthetic_square, generate_synthetic_circle_annotation
import numpy as np
import os
from PIL import Image

def compute_precision_recall(ground_truth, recovered):
    all_precision = []
    all_recall = []
    for class_val in np.unique(ground_truth):
            tp = np.sum((recovered == class_val) & (ground_truth == class_val))
            fp = np.sum((recovered == class_val) & (ground_truth != class_val))
            fn = np.sum((recovered != class_val) & (ground_truth == class_val))
            tn = np.sum((recovered != class_val) & (ground_truth != class_val))
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
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
    """Test that annotation and prediction layers are added correctly"""
    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.random.random((100, 100)))
    my_widget._on_add_annot_seg_layers()    

    assert 'annotations' in viewer.layers
    assert 'segmentation' in viewer.layers

def test_annotation_layer_dims(make_napari_viewer, capsys):
    """Check that dimensions of annotation layer match image layer"""

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.random.random((100, 100, 3)))
    my_widget._on_add_annot_seg_layers()
    assert viewer.layers['annotations'].data.shape == (100, 100)

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.random.random((3, 100, 100)))
    my_widget._on_add_annot_seg_layers()
    assert viewer.layers['annotations'].data.shape == (3, 100, 100)


def test_correct_model(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im, name='sample')
    my_widget._on_add_annot_seg_layers()
    viewer.layers['annotations'].data = im_annot
    my_widget._on_train()
    assert my_widget.qcombo_fe_type.currentText() == 'vgg16', "Model type not updated correctly"

    viewer.layers.clear()
    viewer.add_image(im[:,:,0], name='sample')
    my_widget._on_add_annot_seg_layers()
    viewer.layers['annotations'].data = im_annot
    my_widget._on_train()
    assert my_widget.qcombo_fe_type.currentText() == 'vgg16', "Model type not updated correctly"


def test_rgb_prediction(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture

    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget._on_add_annot_seg_layers()
    viewer.layers['annotations'].data = im_annot
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
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.moveaxis(im,2,0))
    my_widget._on_add_annot_seg_layers()
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
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget._on_add_annot_seg_layers()
    viewer.layers['annotations'].data = im_annot
    my_widget._on_train()
    my_widget._on_predict()

    os.makedirs('_tests/model_dir', exist_ok=True)
    my_widget._on_save_model(save_file='_tests/model_dir/test_model.pkl')  # Changed to .pkl
    assert os.path.exists('_tests/model_dir/test_model.pkl')  # Changed to .pkl


def test_load_model(make_napari_viewer, capsys):
    im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget._on_load_model(save_file='_tests/model_dir/test_model.pkl')  # Changed to .pkl
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
    imgs_dir = os.path.join(os.path.dirname(__file__), '_tests', 'test_imgs')
    im = np.array(Image.open(os.path.join(imgs_dir, '0000_img.png')))
    im_annot = np.array(Image.open(os.path.join(imgs_dir, '0000_scribbles_all_01500_w3.png')))
    ground_truth = np.array(Image.open(os.path.join(imgs_dir, '0000_ground_truth.png')))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget._on_add_annot_seg_layers()
    viewer.layers['annotations'].data = im_annot

    # Simulate selecting the Dino model from the dropdown
    my_widget.qcombo_fe_type.setCurrentText('dinov2_vits14_reg')
    assert my_widget.qcombo_fe_type.currentText() == 'dinov2_vits14_reg'
    
    cp_model = my_widget.cp_model
    cp_model._param.fe_scalings = [1]
    cp_model._param.fe_order = 0  # Set interpolation order to 0
    cp_model._param.fe_name = 'dinov2_vits14_reg'
    cp_model._param.fe_use_cuda = False
    cp_model._param.fe_use_min_features = False
    cp_model._param.tile_annotations = False
    cp_model._param.image_downsample = 1
    cp_model._param.normalize = 1 #no normalization (button id)
    my_widget._update_gui_from_params()
    my_widget.set_fe_btn.click()  # Load the model
    assert cp_model._param.fe_scalings == [1]
    assert cp_model._param.fe_name == 'dinov2_vits14_reg'

    my_widget._on_train()  # Update the classifier with the new parameters
    my_widget._on_predict()
    os.makedirs('_tests/model_dir', exist_ok=True)
    my_widget._on_save_model(save_file='_tests/model_dir/test_model_dino.pkl')
    assert my_widget.qcombo_fe_type.currentText() == 'dinov2_vits14_reg'
    assert os.path.exists('_tests/model_dir/test_model_dino.pkl')


def test_load_model_dino(make_napari_viewer, capsys):
    # im, ground_truth = generate_synthetic_square(im_dims=(252,252), square_dims=(70,70))
    # im_annot = generate_synthetic_circle_annotation(im_dims=(252,252), circle1_xy=(125,70), circle2_xy=(125,125))
    imgs_dir = os.path.join(os.path.dirname(__file__), '_tests', 'test_imgs')
    im = np.array(Image.open(os.path.join(imgs_dir, '0000_img.png')))
    im_annot = np.array(Image.open(os.path.join(imgs_dir, '0000_scribbles_all_01500_w3.png')))
    ground_truth = np.array(Image.open(os.path.join(imgs_dir, '0000_ground_truth.png')))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)

    viewer.add_image(im)

    # Load the Dino model
    my_widget._on_load_model(save_file='_tests/model_dir/test_model_dino.pkl')
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
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget._on_add_annot_seg_layers()
    viewer.layers['annotations'].data = im_annot

    # Create and save the first model with scales [1]
    my_widget.qcombo_fe_type.setCurrentText('vgg16')
    my_widget.fe_scaling_factors.setCurrentText('[1]')
    my_widget.set_fe_btn.setEnabled(True)
    my_widget.set_fe_btn.click()
    assert my_widget.cp_model._param.fe_scalings == [1]
    my_widget._on_train()
    my_widget._on_predict()
    model_path_1 = '_tests/model_dir/test_model_vgg16_scale_1.pkl'
    my_widget._on_save_model(save_file=model_path_1)
    assert os.path.exists(model_path_1)

    # Create and save the second model with scales [1, 2, 3, 4]. Change in the UI:
    my_widget.fe_scaling_factors.setCurrentText('[1,2,4,8]')
    my_widget.set_fe_btn.click()
    assert my_widget.cp_model._param.fe_scalings == [1, 2, 4, 8]

    my_widget.set_fe_btn.click()
    assert my_widget.cp_model._param.fe_scalings == [1, 2, 4, 8]
    my_widget._on_train()
    my_widget._on_predict()
    model_path_2 = '_tests/model_dir/test_model_vgg16_scale_1248.pkl'
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
        my_widget = ConvPaintWidget(viewer)
        viewer.add_image(im)
        my_widget._on_add_annot_seg_layers()
        viewer.layers['annotations'].data = im_annot

        # Load the Dino model
        my_widget.qcombo_fe_type.setCurrentText('dinov2_vits14_reg')
        # Set the scaling to 1
        my_widget.fe_scaling_factors.setCurrentText('[1]')
        my_widget.set_fe_btn.click()
        #set widget disable tiling annotations (should be done automatically, but just to be sure)
        my_widget.check_tile_annotations.setChecked(False)
        # my_widget._update_params_from_gui() # is done automatically
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
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget._on_add_annot_seg_layers()
    viewer.layers['annotations'].data = im_annot

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
        my_widget._on_train()

        assert len(my_widget.fe_layer_selection.selectedItems()) == len(indices_to_select)

        #save the model
        model_path = f'_tests/model_dir/test_model_vgg16_custom_layers_{indices_to_select}.pkl'
        my_widget._on_save_model(save_file=model_path)

    #load the models again and check if the predictions are correct
    for indices_to_select in all_tests:
        model_path = f'_tests/model_dir/test_model_vgg16_custom_layers_{indices_to_select}.pkl'
        my_widget._on_load_model(save_file=model_path)
        assert len(my_widget.fe_layer_selection.selectedItems()) == len(indices_to_select)

        my_widget._on_predict()
        recovered = viewer.layers['segmentation'].data
        precision, recall = compute_precision_recall(ground_truth, recovered)
        assert precision > 0.7, f"Precision: {precision}, too low"
        assert recall > 0.7, f"Recall: {recall}, too low"
    
    assert my_widget.qcombo_fe_type.currentText() == 'vgg16'
