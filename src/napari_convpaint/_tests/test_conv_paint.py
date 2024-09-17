from napari_convpaint.conv_paint import ConvPaintWidget
from napari_convpaint.utils import generate_synthetic_square, generate_synthetic_circle_annotation
import numpy as np
import os

def test_add_layers(make_napari_viewer, capsys):
    """Test that annotation and prediction layers are added correctly"""
    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.random.random((100, 100)))
    my_widget.add_annotation_layer()    

    assert 'annotations' in viewer.layers
    assert 'segmentation' in viewer.layers

def test_annotation_layer_dims(make_napari_viewer, capsys):
    """Check that dimensions of annotation layer match image layer"""

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.random.random((100, 100, 3)))
    my_widget.add_annotation_layer()
    assert viewer.layers['annotations'].data.shape == (100, 100)

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.random.random((3, 100, 100)))
    my_widget.add_annotation_layer()
    assert viewer.layers['annotations'].data.shape == (3, 100, 100)


def test_correct_model(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im, name='sample')
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data = im_annot
    my_widget.update_classifier()
    assert my_widget.qcombo_model_type.currentText() == 'single_layer_vgg16', "Model type not updated correctly"

    viewer.layers.clear()
    viewer.add_image(im[:,:,0], name='sample')
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data = im_annot
    my_widget.update_classifier()
    assert my_widget.qcombo_model_type.currentText() == 'single_layer_vgg16', "Model type not updated correctly"


def test_rgb_prediction(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture

    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data = im_annot
    my_widget.update_classifier()
    my_widget.predict()

    recovered = viewer.layers['segmentation'].data[ground_truth==1]
    precision, recall = compute_precision_recall(ground_truth, recovered)
    
    assert precision > 0.9, f"Precision: {precision}, too low"
    assert recall > 0.9, f"Recall: {recall}, too low"

def compute_precision_recall(ground_truth, recovered):
    tp = np.sum(recovered == 2)# / np.sum(ground_truth == 1)
    fp = np.sum(recovered == 1)#/ np.sum(ground_truth == 1)
    fn = np.sum(ground_truth == 1) - tp
    precision = tp /  (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

def test_multi_channel_prediction(make_napari_viewer, capsys):
    """Check that prediction is bad when disabling multi-channel training for
    image with signal in red channel and large noise in green channel"""

    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im[:,:,1] = np.random.randint(0,100,(100,100))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.moveaxis(im,2,0))
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data[1,:,:] = im_annot
    my_widget.update_classifier()
    my_widget.predict()

    recovered = viewer.layers['segmentation'].data[1][ground_truth==1]
    precision, recall = compute_precision_recall(ground_truth, recovered)
    
    assert precision < 0.9, f"Precision: {precision} is too high for non multi-channelp training"


def test_save_model(make_napari_viewer, capsys):
    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data = im_annot
    my_widget.update_classifier()
    my_widget.predict()

    os.makedirs('_tests/model_dir', exist_ok=True)
    my_widget.save_model(save_file='_tests/model_dir/test_model.pkl')  # Changed to .pkl
    assert os.path.exists('_tests/model_dir/test_model.pkl')  # Changed to .pkl


def test_load_model(make_napari_viewer, capsys):
    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget.load_model(save_file='_tests/model_dir/test_model.pkl')  # Changed to .pkl
    my_widget.predict()

    recovered = viewer.layers['segmentation'].data[ground_truth==1]
    tp = np.sum(recovered == 2)
    fp = np.sum(recovered == 1)
    fn = np.sum(ground_truth == 1) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision > 0.9, f"Precision: {precision}, too low"
    assert recall > 0.9, f"Recall: {recall}, too low"


def test_save_model_dino(make_napari_viewer, capsys):
    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data = im_annot

    # Simulate selecting the Dino model from the dropdown
    my_widget.qcombo_model_type.setCurrentText('dinov2_vits14_reg')
    assert my_widget.qcombo_model_type.currentText() == 'dinov2_vits14_reg'
    
    my_widget.check_use_custom_model.setChecked(True)
    my_widget.param.scalings = [1]
    my_widget.param.order = 0  # Set interpolation order to 0
    my_widget.param.model_name = 'dinov2_vits14_reg'
    my_widget.param.use_cuda = False
    my_widget.param.use_tile_annotations = False
    my_widget.param.use_min_features = False
    my_widget.param.image_downsample = 1
    my_widget.param.normalize = 1 #no normalization (button id)
    my_widget.update_gui_from_params()
    my_widget.create_model_btn.click()  # Load the model
    assert my_widget.param.scalings == [1]
    assert my_widget.param.model_name == 'dinov2_vits14_reg'

    my_widget.update_classifier()  # Update the classifier with the new parameters
    my_widget.predict()
    os.makedirs('_tests/model_dir', exist_ok=True)
    my_widget.save_model(save_file='_tests/model_dir/test_model_dino.pkl')
    assert my_widget.qcombo_model_type.currentText() == 'dinov2_vits14_reg'
    assert os.path.exists('_tests/model_dir/test_model_dino.pkl')


def test_load_model_dino(make_napari_viewer, capsys):
    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    my_widget.check_use_custom_model.setChecked(True)

    viewer.add_image(im)

    # Load the Dino model
    my_widget.load_model(save_file='_tests/model_dir/test_model_dino.pkl')
    # Ensure the model type is set correctly after loading
    assert my_widget.qcombo_model_type.currentText() == 'dinov2_vits14_reg'
    my_widget.predict()

    recovered = viewer.layers['segmentation'].data[ground_truth==1]
    tp = np.sum(recovered == 2)
    fp = np.sum(recovered == 1)
    fn = np.sum(ground_truth == 1) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision > 0.8, f"Precision: {precision}, too low"
    assert recall > 0.8, f"Recall: {recall}, too low"
    

def test_save_and_load_vgg16_models(make_napari_viewer, capsys):
    # Setup synthetic data
    im, ground_truth = generate_synthetic_square(im_dims=(100, 100), square_dims=(30, 30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100, 100), circle1_xy=(19, 19), circle2_xy=(56, 56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data = im_annot

    # Create and save the first model with scales [1]
    my_widget.check_use_custom_model.setChecked(True)
    my_widget.qcombo_model_type.setCurrentText('single_layer_vgg16')
    my_widget.num_scales_combo.setCurrentText('[1]')
    my_widget.update_params_from_gui()
    my_widget.create_model_btn.click()
    assert my_widget.param.scalings == [1]
    my_widget.update_classifier()
    my_widget.predict()
    model_path_1 = '_tests/model_dir/test_model_vgg16_scale_1.pkl'
    my_widget.save_model(save_file=model_path_1)
    assert os.path.exists(model_path_1)

    # Create and save the second model with scales [1, 2, 3, 4]. Change in the UI:
    my_widget.num_scales_combo.setCurrentText('[1,2,4,8]')
    my_widget.update_params_from_gui()
    my_widget.create_model_btn.click()
    assert my_widget.param.scalings == [1, 2, 4, 8]

    my_widget.update_params_from_gui()
    my_widget.create_model_btn.click()
    assert my_widget.param.scalings == [1, 2, 4, 8]
    my_widget.update_classifier()
    my_widget.predict()
    model_path_2 = '_tests/model_dir/test_model_vgg16_scale_1248.pkl'
    my_widget.save_model(save_file=model_path_2)
    assert os.path.exists(model_path_2)

    # Load the second model and predict
    my_widget.load_model(save_file=model_path_2)
    assert my_widget.param.scalings == [1, 2, 4, 8]
    my_widget.predict()
    recovered = viewer.layers['segmentation'].data[ground_truth == 1]
    assert np.any(recovered)  # Check if there is any prediction

    # Load the first model and predict
    my_widget.load_model(save_file=model_path_1)
    assert my_widget.param.scalings == [1]
    my_widget.predict()
    recovered = viewer.layers['segmentation'].data[ground_truth == 1]
    assert np.any(recovered)  # Check if there is any prediction


# test dino model with different image sizes
def test_dino_model_with_different_image_sizes(make_napari_viewer, capsys):
    sizes = [(140, 140), (100, 100), (120, 120),(14,14)]
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
        my_widget.add_annotation_layer()
        viewer.layers['annotations'].data = im_annot

        # Load the Dino model
        my_widget.check_use_custom_model.setChecked(True)
        my_widget.qcombo_model_type.setCurrentText('dinov2_vits14_reg')
        #set widget disable tiling annotations
        my_widget.check_tile_annotations.setChecked(False)
        # Set the scaling to 1
        my_widget.num_scales_combo.setCurrentText('[1]')
        my_widget.update_params_from_gui()
        my_widget.create_model_btn.click()
        my_widget.update_classifier()
        my_widget.predict()

        recovered = viewer.layers['segmentation'].data
        
        #check that the shape of the recovered is the same as the annotation
        assert recovered.shape == im_annot.shape


def test_custom_vgg16_layers(make_napari_viewer, capsys):
    # Setup synthetic data
    im, ground_truth = generate_synthetic_square(im_dims=(100, 100), square_dims=(30, 30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100, 100), circle1_xy=(19, 19), circle2_xy=(56, 56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget.add_annotation_layer()
    viewer.layers['annotations'].data = im_annot

    # Create and save the custom VGG16 model with selected layers
    my_widget.check_use_custom_model.setChecked(True)
    my_widget.qcombo_model_type.setCurrentText('vgg16')
    #select items from widget.model_output_selection = QListWidget()


    # Assuming 'self.model_output_selection' is your QListWidget instance
    all_tests = [[8]]
    for indices_to_select in all_tests:
        # Iterate over the list of indices and select the corresponding items
        for index in indices_to_select:
            item = my_widget.model_output_selection.item(index)
            if item:  # Check if the item exists at that index
                item.setSelected(True)

        my_widget.set_nnmodel_outputs_btn.click()
        my_widget.update_params_from_gui()
        my_widget.create_model_btn.click()
        my_widget.update_classifier()
        assert len(my_widget.model_output_selection.selectedItems()) == len(indices_to_select)

        #save the model
        model_path = f'_tests/model_dir/test_model_vgg16_custom_layers_{indices_to_select}.pkl'
        my_widget.save_model(save_file=model_path)

    #load the models again and check if the predictions are correct
    for indices_to_select in all_tests:
        model_path = f'_tests/model_dir/test_model_vgg16_custom_layers_{indices_to_select}.pkl'
        my_widget.load_model(save_file=model_path)
        assert len(my_widget.model_output_selection.selectedItems()) == len(indices_to_select)

        my_widget.predict()
        recovered = viewer.layers['segmentation'].data[ground_truth == 1]
        precision, recall = compute_precision_recall(ground_truth, recovered)
        assert precision > 0.9, f"Precision: {precision}, too low"
        assert recall > 0.9, f"Recall: {recall}, too low"
    
    assert my_widget.qcombo_model_type.currentText() == 'vgg16'
