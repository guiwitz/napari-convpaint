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
    # The widget has a width floor (set once shown), so the dock does not open too narrow: at that
    # width every tab's content fits the viewport that is left for it. Checked on the viewport
    # itself rather than by adding the scrollbar's sizeHint, which overlay scrollbars (macOS) do
    # not take up - _set_width_floor measures the space actually taken, and adding it regardless
    # would only make the standalone widget wider than it needs to be
    qtbot.waitUntil(lambda: widget.minimumWidth() >= 300)
    widget.resize(widget.minimumWidth(), 250) # Short enough that every tab scrolls vertically
    for i in range(widget.tabs.count()):
        widget.tabs.setCurrentIndex(i)
        qtbot.wait(20)
        scroll_area = QTabWidget.widget(widget.tabs, i)
        page_min = widget.tabs.widget(i).minimumSizeHint().width()
        assert scroll_area.viewport().width() >= page_min, widget.tabs.tab_names[i]
        assert not scroll_area.horizontalScrollBar().isVisible(), widget.tabs.tab_names[i]


def test_class_count_is_capped(make_napari_viewer, tmp_path):
    """The annotation and segmentation layers are uint8, so the number of classes is capped:
    adding beyond the limit is refused, and a CSV holding too many classes is rejected before the
    current classes are replaced (it would otherwise wipe them and then fail)."""
    import pytest
    from napari_convpaint.convpaint_widget import ConvpaintWidget

    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    names = lambda: [n.text() for n in w.class_names]
    assert len(names()) == 2

    # Lowered here, so that the test does not have to create 255 class rows
    w.max_classes = 3
    w._on_add_class(text='third')
    assert len(names()) == 3
    w._on_add_class(text='fourth')
    assert len(names()) == 3 # Refused at the limit

    # A file with more classes than that leaves the current ones untouched
    before = names()
    csv_path = tmp_path / 'many.csv'
    rows = ['index,name'] + [f'{i},c{i}' for i in range(1, 6)]
    csv_path.write_text(chr(10).join(rows))
    with pytest.raises(ValueError, match='at most 3'):
        w.import_class_names_csv(str(csv_path))
    assert names() == before


def test_output_layers_do_not_steal_selection(make_napari_viewer):
    """Creating segmentation/probabilities/instances/features layers must leave the
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
    w._check_create_instances_layer()
    assert viewer.layers.selection.active is annot
    w._check_create_features_layer(4)
    assert viewer.layers.selection.active is annot
    assert {'segmentation'} <= {l.name for l in viewer.layers}
