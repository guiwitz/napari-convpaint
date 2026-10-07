"""Regression tests for the widget's tab/scroll layout."""
from qtpy.QtWidgets import QScrollArea
from napari_convpaint.convpaint_widget import ConvpaintWidget


def test_widget_opens_on_home_tab_and_tabs_scroll(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = ConvpaintWidget(viewer)
    if hasattr(widget, 'ensure_init'):
        widget.ensure_init()
    widget.show()
    qtbot.waitExposed(widget)
    # A fresh widget opens on the first (Home) tab
    assert widget.tabs.currentIndex() == 0
    # Every tab is wrapped in a scroll area (napari-guitils TabSet, scrollable=True), so that
    # content can't be cut off; tabs.widget(i) unwraps it to the page
    from qtpy.QtWidgets import QTabWidget
    for i in range(widget.tabs.count()):
        assert isinstance(QTabWidget.widget(widget.tabs, i), QScrollArea), widget.tabs.tab_names[i]
        assert widget.tabs.widget(i).layout() is not None
    # The widget has a width floor (set once shown), so the dock does not open too narrow and no
    # tab's content is wider than the scroll area's viewport
    qtbot.waitUntil(lambda: widget.minimumWidth() >= 300)
    from qtpy.QtWidgets import QTabWidget
    for i in range(widget.tabs.count()):
        page_min = widget.tabs.widget(i).minimumSizeHint().width()
        assert widget.minimumWidth() >= page_min + QTabWidget.widget(widget.tabs, i).verticalScrollBar().sizeHint().width()


def test_output_layers_do_not_steal_selection(make_napari_viewer):
    """Creating segmentation/probabilities/features layers must leave the
    active layer untouched — napari selects new layers by default, which
    would steer the user's brush into the output layer instead of the
    annotations layer they were painting on."""
    import numpy as np
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    viewer.add_image(np.random.random((64, 64)), name='img')
    w._on_add_annot_layer()
    annot = viewer.layers['annotations']
    viewer.layers.selection.active = annot

    w._check_create_segmentation_layer()
    assert viewer.layers.selection.active is annot
    w._check_create_probas_layer(2)
    assert viewer.layers.selection.active is annot
    w._check_create_features_layer(4)
    assert viewer.layers.selection.active is annot
    assert {'segmentation'} <= {l.name for l in viewer.layers}
