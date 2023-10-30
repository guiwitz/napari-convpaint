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
    assert 'prediction' in viewer.layers

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
    assert my_widget.qcombo_model_type.currentText() == 'single_layer_vgg16_rgb', "Model type not updated correctly"

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

    recovered = viewer.layers['prediction'].data[ground_truth==1]
    tp = np.sum(recovered == 2)# / np.sum(ground_truth == 1)
    fp = np.sum(recovered == 1)#/ np.sum(ground_truth == 1)
    fn = np.sum(ground_truth == 1) - tp
    precision = tp /  (tp + fp)
    recall = tp / (tp + fn)
    assert precision > 0.9, f"Precision: {precision}, too low"
    assert recall > 0.9, f"Recall: {recall}, too low"

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
    my_widget.save_model(save_file='_tests/model_dir/test_model.joblib')
    assert os.path.exists('_tests/model_dir/test_model.joblib')
    assert os.path.getsize('_tests/model_dir/convpaint_params.yml')

def test_load_model(make_napari_viewer, capsys):

    im, ground_truth = generate_synthetic_square(im_dims=(100,100), square_dims=(30,30))
    im_annot = generate_synthetic_circle_annotation(im_dims=(100,100), circle1_xy=(19,19), circle2_xy=(56,56))

    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(im)
    my_widget.load_classifier(save_file='_tests/model_dir/test_model.joblib')
    my_widget.predict()

    recovered = viewer.layers['prediction'].data[ground_truth==1]
    tp = np.sum(recovered == 2)# / np.sum(ground_truth == 1)
    fp = np.sum(recovered == 1)#/ np.sum(ground_truth == 1)
    fn = np.sum(ground_truth == 1) - tp
    precision = tp /  (tp + fp)
    recall = tp / (tp + fn)
    assert precision > 0.9, f"Precision: {precision}, too low"
    assert recall > 0.9, f"Recall: {recall}, too low"
