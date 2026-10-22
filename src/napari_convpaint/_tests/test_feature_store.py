"""Tests for the persistent feature store (storage logic; model integration is tested below)."""
import os
import shutil

import numpy as np
import pytest

from napari_convpaint.feature_store import FeatureStore, _MARKER


def _entry(rng, num_planes=1, was_torch=False):
    """An entry like the FE protocol produces: 2 levels, 2 arrays each, [F, Z, h, w]."""
    levels = []
    for h in (16, 8):
        arrays = [rng.random((5, num_planes, h, h), dtype=np.float32) for _ in range(2)]
        levels.append((arrays, (3, num_planes, h, h), (3, num_planes, h, h)))
    return {"levels": levels, "was_torch": was_torch}


def _equal(p1, p2):
    if p1["was_torch"] != p2["was_torch"] or len(p1["levels"]) != len(p2["levels"]):
        return False
    for (a1, scaled1, cropped1), (a2, scaled2, cropped2) in zip(p1["levels"], p2["levels"]):
        if tuple(scaled1) != tuple(scaled2) or tuple(cropped1) != tuple(cropped2) or len(a1) != len(a2):
            return False
        if not all(np.array_equal(x, y) for x, y in zip(a1, a2)):
            return False
    return True


def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    store = FeatureStore(tmp_path / "store")
    key = ("abc123", (("fe_name", "gaussian_features"), ("fe_scalings", (1, 2))))
    assert key not in store
    assert store.get(key) is None
    entry = _entry(rng, was_torch=True)
    store.put(key, entry)
    assert key in store and len(store) == 1
    got = store.get(key)
    assert _equal(got, entry)
    assert store.stats()["hits"] == 1 and store.stats()["misses"] == 1
    assert store.nbytes > 0
    # Memory-mapped arrays must be usable like normal arrays (incl. torch lifting, copy-on-write)
    import torch
    t = torch.from_numpy(got["levels"][0][0][0])
    assert t.shape == (5, 1, 16, 16)


def test_store_keys_and_idempotent_put(tmp_path):
    rng = np.random.default_rng(1)
    store = FeatureStore(tmp_path / "store")
    p1, p2 = _entry(rng), _entry(rng)
    key1 = ("data1", ("sig",))
    key2 = ("data1", ("other_sig",))  # same data, other FE settings -> other entry
    store.put(key1, p1)
    store.put(key2, p2)
    assert len(store) == 2
    assert _equal(store.get(key1), p1) and _equal(store.get(key2), p2)
    store.put(key1, p2)  # already present -> not overwritten
    assert _equal(store.get(key1), p1)


def test_store_clear_and_reopen(tmp_path):
    rng = np.random.default_rng(2)
    folder = tmp_path / "store"
    store = FeatureStore(folder)
    store.put(("d", ("s",)), _entry(rng))
    # Reopening an existing store finds the entries
    store2 = FeatureStore(folder)
    assert ("d", ("s",)) in store2 and len(store2) == 1
    store2.clear()
    assert len(store) == 0 and os.path.isfile(folder / _MARKER)


def test_store_size_cap(tmp_path):
    rng = np.random.default_rng(3)
    p1 = _entry(rng)
    one = sum(a.nbytes for arrays, _, _ in p1["levels"] for a in arrays)
    store = FeatureStore(tmp_path / "store", max_bytes=int(2.5 * one))
    store.put(("a", ("s",)), p1)
    store.put(("b", ("s",)), _entry(rng))
    assert len(store) == 2 and store.nbytes >= 2 * one
    with pytest.warns(UserWarning, match="size cap"):
        store.put(("c", ("s",)), _entry(rng))       # would exceed the cap -> not stored
    assert len(store) == 2 and ("c", ("s",)) not in store
    store.clear()
    assert store.nbytes == 0
    store.put(("c", ("s",)), _entry(rng))           # room again after clearing
    assert len(store) == 1
    # A reopened store knows its size
    assert FeatureStore(tmp_path / "store").nbytes == store.nbytes


def test_store_refuses_foreign_folder(tmp_path):
    (tmp_path / "somefile.txt").write_text("not a store")
    with pytest.raises(ValueError, match="not empty and not a feature store"):
        FeatureStore(tmp_path)
    # An empty folder is fine
    FeatureStore(tmp_path / "empty")
    # A folder that cannot be created is rejected with a clear message, not an OSError
    with pytest.raises(ValueError, match="Cannot create"):
        FeatureStore(tmp_path / "somefile.txt" / "sub")


def test_store_write_failure_degrades_to_not_storing(tmp_path, monkeypatch):
    """If the folder becomes unwritable, put() warns once and stops storing instead of raising
    inside the extraction (the store stays usable for reading)."""
    rng = np.random.default_rng(4)
    store = FeatureStore(tmp_path / "store")
    store.put(("a", ("s",)), _entry(rng))
    monkeypatch.setattr(np, "save", lambda *a, **k: (_ for _ in ()).throw(OSError(13, "Permission denied")))
    with pytest.warns(UserWarning, match="cannot write"):
        store.put(("b", ("s",)), _entry(rng))
    assert len(store) == 1 and ("b", ("s",)) not in store
    assert not any(d.name.endswith(".tmp") for d in (tmp_path / "store").iterdir())   # no leftovers
    assert store.get(("a", ("s",))) is not None


def test_damaged_entry_is_dropped_and_stored_again(tmp_path):
    """A damaged entry (write cut off, files removed) must not fail every later read of its plane:
    it is deleted on the next get, which simply misses, so the plane is extracted and stored again."""
    rng = np.random.default_rng(5)
    store = FeatureStore(tmp_path / "store")
    key = ("a", ("s",))
    entry = _entry(rng)
    store.put(key, entry)
    npy = os.path.join(store._entry_dir(key), "l0_0.npy")
    with open(npy, "r+b") as f:
        f.truncate(os.path.getsize(npy) // 2)   # as an interrupted write leaves it
    assert store.get(key) is None and key not in store
    assert len(store) == 0 and store.nbytes == 0
    store.put(key, entry)                        # extracted again -> stored again
    assert _equal(store.get(key), entry)
    # A truncated meta file is dropped just the same
    with open(os.path.join(store._entry_dir(key), "meta.json"), "w") as f:
        f.write('{"was_torch": fal')
    assert store.get(key) is None and len(store) == 0


def test_store_survives_a_deleted_folder(tmp_path):
    """Deleting the folder while the store is on must not raise out of training: it is recreated,
    with its marker, so that the next session still recognizes it as a feature store."""
    rng = np.random.default_rng(6)
    folder = tmp_path / "store"
    store = FeatureStore(folder)
    store.put(("a", ("s",)), _entry(rng))
    shutil.rmtree(folder)
    assert store.stats()["entries"] == 0                          # reporting does not fail either
    store.put(("b", ("s",)), _entry(rng))
    assert ("b", ("s",)) in store and len(store) == 1
    assert store.nbytes == store._scan_bytes()                    # size corrected with the recovery
    assert os.path.isfile(folder / _MARKER)
    FeatureStore(folder)                                          # opens again, not seen as foreign


def test_two_writers_do_not_delete_each_others_files(tmp_path, monkeypatch):
    """Every write uses a temporary folder of its own, so a second instance storing the same plane
    cannot remove the files of the first one; whoever renames first wins, the other drops its copy."""
    rng = np.random.default_rng(7)
    store = FeatureStore(tmp_path / "store")
    key = ("a", ("s",))
    # A write of another instance in progress (its own .tmp folder) must be left alone
    other_tmp = store._entry_dir(key) + ".999-abcdef12.tmp"
    os.makedirs(other_tmp)
    open(os.path.join(other_tmp, "l0_0.npy"), "w").close()
    store.put(key, _entry(rng))
    assert os.path.isfile(os.path.join(other_tmp, "l0_0.npy")) and key in store
    # And if the other instance renames first, this write is dropped without a warning
    monkeypatch.setattr(os, "rename", lambda *a: (_ for _ in ()).throw(FileExistsError(17, "Exists")))
    store.put(("b", ("s",)), _entry(rng))
    assert ("b", ("s",)) not in store and not store._warned_no_write
    assert not any(n.endswith(".tmp") for n in os.listdir(store.folder) if n != os.path.basename(other_tmp))


def test_store_of_another_format_or_version(tmp_path):
    """The marker is checked when opening: another entry format is refused, features of another
    convpaint version are pointed out but kept (extraction usually did not change)."""
    rng = np.random.default_rng(8)
    folder = tmp_path / "store"
    store = FeatureStore(folder)
    store.put(("a", ("s",)), _entry(rng))
    marker = folder / _MARKER
    marker.write_text('{"format": 2, "convpaint": "9.9.9"}')
    with pytest.raises(ValueError, match="another format"):
        FeatureStore(folder)
    marker.write_text('{"format": 1, "convpaint": "0.0.1"}')
    with pytest.warns(UserWarning, match="written by convpaint"):
        reopened = FeatureStore(folder)
    assert ("a", ("s",)) in reopened   # kept and reusable


# --- integration with the model --------------------------------------------

def _stack_and_annot():
    rng = np.random.RandomState(0)
    stack = rng.rand(3, 64, 64).astype(np.float32)   # [Z, H, W], single channel
    annot = np.zeros((3, 64, 64), dtype=np.uint8)
    annot[1, :10, :10] = 1
    annot[1, -10:, -10:] = 2
    return stack, annot


def test_model_store_serves_across_instances(tmp_path):
    """Features extracted with the store on are reused by another model instance with the
    same folder (no cache), bit-identical; annotation tiles are never stored."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    stack, annot = _stack_and_annot()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp_off = ConvpaintModel('gaussian')
        cp_off.set_params(tile_annotations=False, normalize=1)
        cp_off.train(stack, annot)
        seg_off = cp_off.segment(stack)

        cp1 = ConvpaintModel('gaussian')
        cp1.set_params(tile_annotations=True, normalize=1)
        store = cp1.enable_feature_store(tmp_path / "store")
        cp1.train(stack, annot)                        # annotation tiles -> nothing stored
        assert len(store) == 0
        cp1.set_params(tile_annotations=False)
        cp1.train(stack, annot)                        # the annotated plane is stored
        assert len(store) == 1
        seg1 = cp1.segment(stack)                      # 1 plane reused, 2 stored
        assert len(store) == 3 and store.stats()['hits'] == 1

        cp2 = ConvpaintModel('gaussian')               # a new model (e.g. a new session)
        cp2.set_params(tile_annotations=False, normalize=1)
        store2 = cp2.enable_feature_store(tmp_path / "store")
        cp2.train(stack, annot)
        seg2 = cp2.segment(stack)
        assert store2.stats()['hits'] == 4 and store2.stats()['misses'] == 0
        cp2.disable_feature_store()
        assert cp2._feature_store is None and len(store2) == 3   # files are kept
    assert np.array_equal(seg1, seg_off) and np.array_equal(seg2, seg_off)


def test_model_cache_and_store_together(tmp_path):
    """With both enabled, the cache is consulted first and misses are kept in both."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    stack, _ = _stack_and_annot()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel('gaussian')
        cp.set_params(normalize=1)
        fc = cp.enable_feature_cache()
        store = cp.enable_feature_store(tmp_path / "store")
        f1 = cp.get_feature_image(stack)
        assert len(fc) == 3 and len(store) == 3
        f2 = cp.get_feature_image(stack)
        assert fc.stats()['hits'] == 3 and store.stats()['hits'] == 0
        cp.disable_feature_cache()
        f3 = cp.get_feature_image(stack)
        assert store.stats()['hits'] == 3
        # Store hits are promoted into the cache (as RAM copies): the cache answers from then on
        fc2 = cp.enable_feature_cache()
        f4 = cp.get_feature_image(stack)               # from the store, promoted
        assert store.stats()['hits'] == 6 and len(fc2) == 3
        f5 = cp.get_feature_image(stack)               # from the cache
        assert store.stats()['hits'] == 6 and fc2.stats()['hits'] == 3
        assert all(not isinstance(a, np.memmap) for arrays, _, _ in next(iter(fc2._entries.values()))[0]["levels"] for a in arrays)
    assert np.array_equal(f1, f2) and np.array_equal(f1, f3)
    assert np.array_equal(f1, f4) and np.array_equal(f1, f5)


def test_store_features_prepares_a_stack(tmp_path):
    """store_features fills the store plane by plane; train and predict then only reuse."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    stack, annot = _stack_and_annot()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp_off = ConvpaintModel('gaussian')
        cp_off.set_params(tile_annotations=False)
        cp_off.train(stack, annot)
        seg_off = cp_off.segment(stack)

        cp = ConvpaintModel('gaussian')
        cp.set_params(tile_annotations=False)          # default normalization (over the stack)
        store = cp.enable_feature_store(tmp_path / "store")
        with pytest.raises(ValueError):
            ConvpaintModel('gaussian').store_features(stack)   # no store enabled
        cp.store_features(stack)
        assert len(store) == 3 and store.stats()['misses'] == 3
        cp.store_features(stack)                       # already stored -> nothing new
        assert len(store) == 3 and store.stats()['hits'] == 3
        cp.train(stack, annot)
        seg = cp.segment(stack)
        assert store.stats()['misses'] == 3            # no extraction anymore
    assert np.array_equal(seg, seg_off)


# --- widget ------------------------------------------------------------------

def test_widget_store_controls(make_napari_viewer, tmp_path, monkeypatch):
    """Checkbox connects/disconnects the store on the model; train+predict fill it;
    delete (confirmed) empties it but keeps it active; unchecking keeps the files."""
    from qtpy.QtWidgets import QMessageBox
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    from napari_convpaint.testing_data import generate_synthetic_square, generate_synthetic_circle_annotation
    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    assert w.cp_model._feature_store is None and not w.check_use_store.isChecked()
    assert w.store_folder.endswith('feature_store')            # default folder in the user cache dir

    w.store_folder = str(tmp_path / 'store')
    w.check_use_store.setChecked(True)
    fs = w.cp_model._feature_store
    assert fs is not None and fs.folder == w.store_folder and w.btn_store_delete.isEnabled()

    im, _ = generate_synthetic_square(im_dims=(252, 252), square_dims=(70, 70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252, 252), circle1_xy=(125, 70), circle2_xy=(125, 125))
    viewer.add_image(im)
    w._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot
    w.cp_model.set_params(channel_mode='rgb')
    w._on_train()
    w._on_predict()
    assert len(fs) >= 1 and 'Stored features: 1 planes' in w.store_size_label.text()

    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
    w._on_delete_stored_features()
    assert len(fs) == 0 and w.cp_model._feature_store is fs   # emptied, still active

    w.check_use_store.setChecked(False)
    assert w.cp_model._feature_store is None and not w.btn_store_delete.isEnabled()
    assert (tmp_path / 'store' / 'convpaint_feature_store.json').is_file()

    # A folder that cannot be used: reported, checkbox stays off, the previous folder is kept,
    # and the store can be enabled again right away (nothing sticky)
    (tmp_path / 'dogs').mkdir(); (tmp_path / 'dogs' / 'dog1.png').write_bytes(b'x')
    w.store_folder = str(tmp_path / 'dogs')
    w.check_use_store.setChecked(True)
    assert w.cp_model._feature_store is None and not w.check_use_store.isChecked()
    assert w.store_folder == str(tmp_path / 'store') and not w.btn_store_features.isEnabled()
    w.check_use_store.setChecked(True)
    assert w.cp_model._feature_store is not None and w.cp_model._feature_store.folder == str(tmp_path / 'store')


def test_widget_choosing_a_folder_uses_it(make_napari_viewer, tmp_path, monkeypatch):
    """Choosing a folder enables the store on it, as in the API where the folder comes with
    enabling - and an unusable folder is reported right away instead of only when the box is
    ticked, leaving the settings as they were."""
    import napari_convpaint.convpaint_widget as cw
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    messages = []
    monkeypatch.setattr(cw, 'show_info', messages.append)
    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    assert not w.check_use_store.isChecked()

    # Store off: choosing a usable folder switches it on
    good = tmp_path / 'good'
    monkeypatch.setattr(cw.QFileDialog, 'getExistingDirectory', staticmethod(lambda *a, **k: str(good)))
    w._on_choose_store_folder()
    assert w.check_use_store.isChecked() and w.store_enabled
    assert w.cp_model._feature_store is not None and w.store_folder == str(good)

    # Store off again: an unusable folder is refused when it is chosen, nothing changes
    w.check_use_store.setChecked(False)
    bad = tmp_path / 'dogs'
    bad.mkdir()
    (bad / 'dog1.png').write_bytes(b'x')
    messages.clear()
    monkeypatch.setattr(cw.QFileDialog, 'getExistingDirectory', staticmethod(lambda *a, **k: str(bad)))
    w._on_choose_store_folder()
    assert any('not enabled' in m for m in messages)
    assert not w.check_use_store.isChecked() and not w.store_enabled
    assert w.cp_model._feature_store is None and w.store_folder == str(good)

    # Cancelling the dialog leaves everything alone
    monkeypatch.setattr(cw.QFileDialog, 'getExistingDirectory', staticmethod(lambda *a, **k: ''))
    w._on_choose_store_folder()
    assert w.store_folder == str(good) and not w.check_use_store.isChecked()


def test_widget_bad_folder_keeps_the_running_store(make_napari_viewer, tmp_path):
    """Choosing an unusable folder while the store runs must not switch it off: the model keeps the
    store it has (the new folder is only taken once it works), so the widget keeps showing it."""
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    w.store_folder = str(tmp_path / 'good')
    w.check_use_store.setChecked(True)
    running = w.cp_model._feature_store
    assert running is not None

    (tmp_path / 'dogs').mkdir()
    (tmp_path / 'dogs' / 'dog1.png').write_bytes(b'x')
    w.store_folder = str(tmp_path / 'dogs')
    w._apply_feature_store(recreate=True) # As _on_choose_store_folder does

    assert w.cp_model._feature_store is running # Same store, still on the folder that works
    assert w.check_use_store.isChecked() and w.store_enabled
    assert w.store_folder == str(tmp_path / 'good')
    assert w.btn_store_delete.isEnabled() and w.btn_store_features.isEnabled()


def test_widget_reset_restores_store_settings(make_napari_viewer, tmp_path):
    """Resetting Convpaint puts the store settings back to their defaults: the checkbox, the folder
    and the model's store, which would otherwise keep running on the old folder."""
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    default_folder = w.store_folder

    w.store_folder = str(tmp_path / 'store')
    w.check_use_store.setChecked(True)
    assert w.cp_model._feature_store is not None

    w._on_reset_convpaint()

    assert not w.check_use_store.isChecked() and not w.store_enabled
    assert w.store_folder == default_folder and w.cp_model._feature_store is None
    assert '(store off)' in w.store_size_label.text()


def test_widget_store_features_of_stack(make_napari_viewer, tmp_path):
    """The 'Store features' action stores every plane of the selected stack; a following
    stack prediction only reuses."""
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    from napari_convpaint.testing_data import generate_synthetic_square, generate_synthetic_circle_annotation
    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    w.store_folder = str(tmp_path / 'store')
    w.check_use_store.setChecked(True)
    fs = w.cp_model._feature_store

    im, _ = generate_synthetic_square(im_dims=(252, 252), square_dims=(70, 70))
    stack = np.stack([im, im[::-1], im[:, ::-1]])           # 3 different planes [Z, H, W, 3] (RGB)
    viewer.add_image(stack, rgb=True)
    w._on_select_layer()                                    # (the changed-signal handler is QTimer-delayed)
    assert w.cp_model.get_param('channel_mode') == 'rgb'
    w._on_store_features()
    assert len(fs) == 3 and fs.stats()['misses'] == 3

    im_annot = generate_synthetic_circle_annotation(im_dims=(252, 252), circle1_xy=(125, 70), circle2_xy=(125, 125))
    w._on_add_annot_layer()
    viewer.layers['annotations'].data[0] = im_annot
    w.check_tile_annotations.setChecked(False)
    w._on_train()
    assert len(fs) == 3 and fs.stats()['misses'] == 3, fs.stats()   # training reused the stored plane
    w._on_predict_all()
    assert len(fs) == 3 and fs.stats()['misses'] == 3, fs.stats()   # nothing extracted anymore


def test_store_hit_is_served_from_the_cached_copy(tmp_path):
    """A store hit is copied into the cache, and that copy is also what is returned: reconstructing
    from the memory-mapped files instead would read all of it a second time."""
    from napari_convpaint.convpaint_model import ConvpaintModel
    cp = ConvpaintModel('gaussian')
    store = cp.enable_feature_store(tmp_path / "store")
    cp.enable_feature_cache()
    key = ("data", ("sig",))
    store.put(key, _entry(np.random.default_rng(9)))
    native = cp._reuse_native(key)
    assert all(not isinstance(a, np.memmap) for arrays, _, _ in native["levels"] for a in arrays)
    assert native is cp._feature_cache.get(key) # The very copy that was cached, not the memory maps


def test_cached_features_are_stored_when_the_store_is_on(tmp_path):
    """Features extracted before the store was enabled are only in RAM; using them afterwards (here
    through store_features) keeps them on disk as well, instead of silently storing nothing."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.default_rng(12)
    stack = rng.random((3, 32, 32), dtype=np.float32)
    cp = ConvpaintModel('gaussian')
    cp.enable_feature_cache()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp.get_feature_image(stack)                      # cache only, no store yet
        assert len(cp._feature_cache) == 3
        store = cp.enable_feature_store(tmp_path / "store")
        assert len(store) == 0
        cp.store_features(stack)                         # served from the cache, kept on disk
    assert len(store) == 3


def test_store_features_announces_the_image(tmp_path):
    """store_features takes a whole image, so it announces it like train and predict do - once for
    the image, not once per plane, which would make every plane count as another image."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.default_rng(11)
    stack_a = rng.random((3, 32, 32), dtype=np.float32)
    stack_b = rng.random((3, 32, 32), dtype=np.float32)
    cp = ConvpaintModel('gaussian')
    cp.enable_feature_store(tmp_path / "store")
    cp.enable_feature_cache()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp.store_features(stack_a)
        first = cp._image_scope
        assert first is not None
        cp.store_features(stack_b)
    assert cp._image_scope is not None and cp._image_scope != first


def test_clear_feature_store_keeps_the_store(tmp_path):
    """clear_feature_store() deletes the stored features but leaves the store enabled."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel('gaussian')
        fs = cp.enable_feature_store(tmp_path / "store")
        cp.get_feature_image(img)
        assert len(fs) == 1
        cp.clear_feature_store()
        assert len(fs) == 0 and cp._feature_store is fs
        cp.get_feature_image(img)                      # still usable afterwards
        assert len(fs) == 1
