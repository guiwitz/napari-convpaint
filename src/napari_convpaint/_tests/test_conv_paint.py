from napari_convpaint.conv_paint import ConvPaintWidget
import numpy as np

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    my_widget = ConvPaintWidget(viewer)
    viewer.add_image(np.random.random((100, 100)))
    my_widget.add_annotation_layer()

    # create our widget, passing in the viewer
    

    assert 'annotations' in viewer.layers
    assert 'prediction' in viewer.layers