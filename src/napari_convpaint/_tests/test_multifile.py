"""Tests for the multifile workflow in ConvpaintWidget."""
import numpy as np
import tifffile
from qtpy.QtWidgets import QFileDialog, QMessageBox

from napari_convpaint.convpaint_widget import ConvpaintWidget
from napari_convpaint.testing_data import generate_synthetic_square


def _make_widget(make_napari_viewer, channel_mode='rgb'):
    viewer = make_napari_viewer()
    widget = ConvpaintWidget(viewer)
    widget.ensure_init()
    widget.cp_model.set_params(channel_mode=channel_mode)
    return viewer, widget


def _patch_dialogs(monkeypatch, folder):
    """Make QFileDialog return `folder` and QMessageBox auto-accept."""
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        lambda *a, **k: str(folder))
    monkeypatch.setattr(QMessageBox, 'question',
                        lambda *a, **k: QMessageBox.Yes)


def _make_striped_rgb(rng):
    """128x128 RGB image with three horizontal stripes of distinct colors.

    Same color palette and layout across images so per-image normalization
    produces the same scales; small noise keeps features non-degenerate.
    Stripes: rows 0:40 red, 40:80 green, 80:128 blue.
    """
    im = rng.integers(0, 25, (128, 128, 3), dtype=np.uint8)
    im[0:40, :, 0] += 180    # red stripe
    im[40:80, :, 1] += 180   # green stripe
    im[80:128, :, 2] += 180  # blue stripe
    return im


def test_multifile_folder_populates_table_and_filters(make_napari_viewer, tmp_path, monkeypatch):
    """Opening a folder populates the table with images and skips annot/seg files."""
    # Mix of images and annotation/segmentation files
    im, _ = generate_synthetic_square((64, 64), (20, 20))
    tifffile.imwrite(tmp_path / 'img1.tif', im)
    tifffile.imwrite(tmp_path / 'img2.tif', im)
    tifffile.imwrite(tmp_path / 'img1_annotations.tif',
                     np.zeros((64, 64), dtype=np.uint8))
    tifffile.imwrite(tmp_path / 'img2_segmentation.tif',
                     np.zeros((64, 64), dtype=np.uint8))

    _patch_dialogs(monkeypatch, tmp_path)
    viewer, w = _make_widget(make_napari_viewer)

    w._select_multifile_img_folder()

    rows = [w.multifile_list.item(r, 1).text()
            for r in range(w.multifile_list.rowCount())]
    assert sorted(rows) == ['img1.tif', 'img2.tif']


def test_multifile_opens_different_image_types(make_napari_viewer, tmp_path, monkeypatch):
    """Single-channel 2D, RGB, and 3D stack all load via multifile."""
    # 2D single-channel
    gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    # 2D RGB
    rgb = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    # 3D stack (Z, H, W)
    stack = np.random.randint(0, 255, (5, 64, 64), dtype=np.uint8)

    tifffile.imwrite(tmp_path / 'a_gray.tif', gray)
    tifffile.imwrite(tmp_path / 'b_rgb.tif', rgb)
    tifffile.imwrite(tmp_path / 'c_stack.tif', stack)

    _patch_dialogs(monkeypatch, tmp_path)
    viewer, w = _make_widget(make_napari_viewer)

    w._select_multifile_img_folder()
    assert w.multifile_list.rowCount() == 3

    # Auto-opens first image; walk through the rest
    for row in range(w.multifile_list.rowCount()):
        fname = w.multifile_list.item(row, 1).text()
        w._on_multifile_open_file(row, 1)
        assert fname in viewer.layers
        assert w._current_multifile_filename == fname


def test_multifile_training_combines_classes_across_files(make_napari_viewer, tmp_path, monkeypatch, qtbot):
    """Annotate class 1 on the red stripe of img1 and class 2 on the green stripe
    of img2. After multifile training, predicting img3 must contain both classes.
    """
    rng = np.random.default_rng(0)
    im1 = _make_striped_rgb(rng)
    im2 = _make_striped_rgb(rng)
    im3 = _make_striped_rgb(rng)
    tifffile.imwrite(tmp_path / 'img1.tif', im1)
    tifffile.imwrite(tmp_path / 'img2.tif', im2)
    tifffile.imwrite(tmp_path / 'img3.tif', im3)

    _patch_dialogs(monkeypatch, tmp_path)
    viewer, w = _make_widget(make_napari_viewer, channel_mode='rgb')

    w._select_multifile_img_folder()
    qtbot.wait(200)

    # img1: class 1 annotated inside the red stripe
    annot1 = np.zeros((128, 128), dtype=np.uint8)
    annot1[10:30, 10:120] = 1
    # img2: class 2 annotated inside the green stripe
    annot2 = np.zeros((128, 128), dtype=np.uint8)
    annot2[50:70, 10:120] = 2
    w._multifile_annotations_store['img1.tif'] = annot1
    w._multifile_annotations_store['img2.tif'] = annot2

    # Train on the multifile store
    w._on_train_on_multifile()
    assert w.trained, "Multifile training did not mark model as trained"

    # Segment img3 → write to a new folder
    out_dir = tmp_path / 'seg_out'
    out_dir.mkdir()
    # "Trick" QFileDialog to return our output directory when the widget tries to ask where to save segmentations
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        lambda *a, **k: str(out_dir))

    # Select only the third row (img3)
    w.multifile_list.selectRow(2)
    w._on_segment_selected_multifile()

    # Check that segmentation TIFF was written and contains both classes in the expected regions
    seg_file = out_dir / 'img3_segmentation.tif'
    assert seg_file.exists(), "Segmentation TIFF was not written"
    seg = tifffile.imread(seg_file)

    unique = set(np.unique(seg).tolist())
    assert 1 in unique and 2 in unique, (
        f"Expected both classes 1 and 2 in prediction, got {unique}"
    )
    # Class 1 should dominate the red stripe, class 2 the green stripe
    assert (seg[0:40, :] == 1).mean() > 0.5, "Class 1 did not dominate the red stripe"
    assert (seg[40:80, :] == 2).mean() > 0.5, "Class 2 did not dominate the green stripe"

    # Segmentation store was updated and tick switched to persistent (green)
    assert w._multifile_segmentation_store.get('img3.tif') == str(seg_file)


def test_multifile_import_presaved_annotations(make_napari_viewer, tmp_path, monkeypatch):
    """Pre-saved annotations TIFFs are registered as persistent (path string, green tick)."""
    im, _ = generate_synthetic_square((64, 64), (20, 20))
    tifffile.imwrite(tmp_path / 'img1.tif', im)
    tifffile.imwrite(tmp_path / 'img2.tif', im)

    # Corresponding pre-saved annotations in a sub-folder
    annot_dir = tmp_path / 'annots'
    annot_dir.mkdir()
    pre1 = np.zeros((64, 64), dtype=np.uint8)
    pre1[10:20, 10:20] = 1
    pre2 = np.zeros((64, 64), dtype=np.uint8)
    pre2[30:40, 30:40] = 2
    tifffile.imwrite(annot_dir / 'img1_annotations.tif', pre1)
    tifffile.imwrite(annot_dir / 'img2_annotations.tif', pre2)

    _patch_dialogs(monkeypatch, tmp_path)
    viewer, w = _make_widget(make_napari_viewer)

    # Populate table from images folder
    w._select_multifile_img_folder()
    assert w.multifile_list.rowCount() == 2

    # Now import annotations from the annot_dir
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        lambda *a, **k: str(annot_dir))
    w._import_annot_and_seg()

    store = w._multifile_annotations_store
    assert set(store.keys()) == {'img1.tif', 'img2.tif'}
    # Imported entries are stored as path strings (persistent)
    assert all(isinstance(v, str) for v in store.values())
    assert store['img1.tif'].endswith('img1_annotations.tif')

    # Content on disk matches what was written
    np.testing.assert_array_equal(
        tifffile.imread(store['img1.tif']), pre1
    )
    np.testing.assert_array_equal(
        tifffile.imread(store['img2.tif']), pre2
    )

    # Training consumes the imported (path-stored) annotations and succeeds
    w._on_train_on_multifile()
    assert w.trained, "Training did not succeed with imported annotations"


def test_multifile_export_annotations_creates_tiffs(make_napari_viewer, tmp_path, monkeypatch):
    """In-memory annotations export to named TIFF files and flip to persistent state."""
    im, _ = generate_synthetic_square((64, 64), (20, 20))
    tifffile.imwrite(tmp_path / 'img1.tif', im)
    tifffile.imwrite(tmp_path / 'img2.tif', im)

    _patch_dialogs(monkeypatch, tmp_path)
    viewer, w = _make_widget(make_napari_viewer)
    w._select_multifile_img_folder()

    a1 = np.zeros((64, 64), dtype=np.uint8)
    a1[5:15, 5:15] = 1
    a2 = np.zeros((64, 64), dtype=np.uint8)
    a2[20:30, 20:30] = 2
    w._multifile_annotations_store['img1.tif'] = a1
    w._multifile_annotations_store['img2.tif'] = a2

    export_dir = tmp_path / 'exported'
    export_dir.mkdir()
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                        lambda *a, **k: str(export_dir))

    w._export_annotations()

    f1 = export_dir / 'img1_annotations.tif'
    f2 = export_dir / 'img2_annotations.tif'
    assert f1.exists(), "img1 annotations TIFF not written"
    assert f2.exists(), "img2 annotations TIFF not written"

    np.testing.assert_array_equal(tifffile.imread(f1), a1)
    np.testing.assert_array_equal(tifffile.imread(f2), a2)

    # Store now holds path strings (persistent / green tick)
    assert w._multifile_annotations_store['img1.tif'] == str(f1)
    assert w._multifile_annotations_store['img2.tif'] == str(f2)


def test_multifile_clear_annot_removes_from_store(make_napari_viewer, tmp_path, monkeypatch):
    """Selecting rows and clearing annotations drops them from the in-memory store."""
    im, _ = generate_synthetic_square((64, 64), (20, 20))
    tifffile.imwrite(tmp_path / 'img1.tif', im)
    tifffile.imwrite(tmp_path / 'img2.tif', im)

    _patch_dialogs(monkeypatch, tmp_path)
    viewer, w = _make_widget(make_napari_viewer)
    w._select_multifile_img_folder()

    w._multifile_annotations_store['img1.tif'] = np.ones((64, 64), dtype=np.uint8)
    w._multifile_annotations_store['img2.tif'] = np.ones((64, 64), dtype=np.uint8) * 2

    w.multifile_list.selectRow(0)
    w._multifile_clear_annot()

    assert 'img1.tif' not in w._multifile_annotations_store
    assert 'img2.tif' in w._multifile_annotations_store
