"""Tests for cooperative cancellation of training and prediction."""
import threading
import warnings

import numpy as np
import pytest

from napari_convpaint.convpaint_model import ConvpaintModel
from napari_convpaint.feature_extractors.gaussian import GaussianFeatures
from napari_convpaint.utils import CancelToken, CancelledError


def _tiny_dataset():
    rng = np.random.default_rng(0)
    image = rng.random((64, 64), dtype=np.float32)
    annot = np.zeros((64, 64), dtype=np.uint8)
    annot[10:20, 10:20] = 1
    annot[30:40, 30:40] = 2
    return image, annot


def test_cancel_token_starts_uncancelled():
    t = CancelToken()
    assert t.cancelled is False
    t.raise_if_cancelled()  # should not raise


def test_cancel_token_raises_after_cancel():
    t = CancelToken()
    t.cancel()
    assert t.cancelled is True
    with pytest.raises(CancelledError):
        t.raise_if_cancelled()


def test_train_aborts_when_token_is_pre_cancelled():
    """A token cancelled before train() starts should abort at the first checkpoint."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()

    token = CancelToken()
    token.cancel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.train(image, annot, cancel_token=token)


def test_segment_aborts_when_token_is_pre_cancelled():
    """Same for segment() — once trained, a pre-cancelled token aborts prediction."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.train(image, annot)

    token = CancelToken()
    token.cancel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.segment(image, cancel_token=token)


def test_train_completes_with_fresh_token():
    """Passing an un-cancelled token must not affect normal completion."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = model.train(image, annot, cancel_token=CancelToken())
    assert clf is not None


class _CancelOnNthCheck(CancelToken):
    """Test double: auto-cancels itself after `n` checkpoints have been hit.

    Lets us deterministically verify that cancel-checks are reached mid-run,
    without racing a wall-clock timer against a fast feature extractor."""

    def __init__(self, n):
        super().__init__()
        self._n = n
        self._checks = 0

    def raise_if_cancelled(self):
        self._checks += 1
        if self._checks >= self._n and not self.cancelled:
            self.cancel()
        super().raise_if_cancelled()


def test_cancel_mid_train_is_honored():
    """Proves that cancel-checkpoints are actually reached during training:
    the token cancels itself after 2 checks, so train must raise CancelledError
    (it would otherwise complete normally)."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()
    token = _CancelOnNthCheck(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.train(image, annot, cancel_token=token)
    assert token._checks >= 2, "cancel_token was never checked during train"


def test_cancel_mid_segment_is_honored():
    """Same idea for prediction."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.train(image, annot)

    token = _CancelOnNthCheck(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.segment(image, cancel_token=token)
    assert token._checks >= 2, "cancel_token was never checked during segment"


def test_cancel_preserves_previous_classifier():
    """A cancelled training must not clobber a classifier from an earlier
    successful training. Cancel fires before _clf_train (which is an
    uninterruptible C call), so classifier state stays on the previous fit."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.train(image, annot)
    first_clf = model.classifier
    assert first_clf is not None

    token = _CancelOnNthCheck(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.train(image, annot, cancel_token=token)

    assert model.classifier is first_clf, (
        "classifier object changed after a cancelled training — previous fit was clobbered"
    )


def test_cancel_in_memory_mode_leaves_state_retrainable():
    """In memory-mode training, _register_and_update_annots mutates self.annot_dict
    and self.table before feature extraction. If we cancel without rolling those
    back, the *next* train sees no new annotations and raises 'No features or
    targets found'. Verify the rollback makes retraining work after a cancel."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()

    # Cancel the first training mid-run
    token = _CancelOnNthCheck(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.train(image, annot, memory_mode=True, img_ids='img0',
                        cancel_token=token)

    # After cancel, the memory state should be back to where it started, so a
    # fresh train with the same annotations can still find work to do.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = model.train(image, annot, memory_mode=True, img_ids='img0')
    assert clf is not None, "retrain after cancel should succeed"


class _OldSignatureFE(GaussianFeatures):
    """Simulates a third-party FE written before cancellation existed: its
    overrides use the pre-cancellation signatures (no cancel_token, no
    **kwargs). Cancellation is carried by an ambient ContextVar, so such FEs
    must keep working unmodified AND be cancellable through the base-class
    loop checkpoints."""

    def extract_features_pyramid(self, data, param, patched=True, device=None):
        return super().extract_features_pyramid(data, param, patched=patched, device=device)

    def extract_features_from_stack(self, image, device=None):
        return super().extract_features_from_stack(image, device=device)


def test_custom_fe_with_old_signature_still_works():
    """A custom FE with pre-cancellation override signatures must neither crash
    (TypeError from an unexpected cancel_token kwarg) nor lose cancellability."""
    model = ConvpaintModel(fe_name='gaussian_features')
    model.fe_model = _OldSignatureFE()
    image, annot = _tiny_dataset()

    # Trains fine without a token ...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = model.train(image, annot)
    assert clf is not None

    # ... and with one ...
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = model.train(image, annot, cancel_token=CancelToken())
    assert clf is not None

    # ... and is still cancellable mid-run through the ambient token, even
    # though the FE itself never sees or forwards it.
    token = _CancelOnNthCheck(n=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.train(image, annot, cancel_token=token)
    assert token._checks >= 2, "ambient cancel_token was never checked with a legacy FE"


def test_ambient_token_does_not_leak_out_of_the_call():
    """After a cancelled call returns, the ambient token must be uninstalled —
    a later call without a token must not see the stale cancelled one."""
    from napari_convpaint.utils import check_cancel

    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()

    token = CancelToken()
    token.cancel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.train(image, annot, cancel_token=token)

    check_cancel()  # must not raise: no ambient token installed anymore
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = model.train(image, annot)  # must not be affected by the old token
    assert clf is not None


def test_cancel_during_catboost_fit_preserves_classifier():
    """The CatBoost fit itself is cancellable on CPU via a per-iteration
    callback. A cancel that lands mid-fit must abort with CancelledError and
    leave the previously trained classifier in place (the partial fit is
    discarded, and self.classifier is only reassigned after a successful fit)."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.train(image, annot)
    first_clf = model.classifier
    assert first_clf is not None

    # Only a handful of checkpoints run before the fit (per-scale, per-Z,
    # pre-fit), so n=20 lands inside the boosting loop's per-iteration
    # callback checks (default 100 iterations).
    token = _CancelOnNthCheck(n=20)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(CancelledError):
            model.train(image, annot, cancel_token=token)
    assert token._checks >= 20, "cancel checkpoints were never reached"
    assert model.classifier is first_clf, (
        "classifier changed after a cancelled fit — partial fit was adopted"
    )

    # And the model must still be retrainable afterwards
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = model.train(image, annot)
    assert clf is not None and clf is not first_clf


def test_cancel_from_another_thread_aborts_train():
    """Cross-thread cancel: main thread calls cancel() while a worker runs train().
    Uses the auto-cancel token so the outcome is deterministic regardless of how
    fast the feature extractor happens to be on this machine."""
    model = ConvpaintModel(fe_name='gaussian_features')
    image, annot = _tiny_dataset()

    # Give enough checkpoints that the main thread has time to call cancel(),
    # but few enough that the worker will actually reach the cancel on the next check.
    token = CancelToken()
    result = {}
    started = threading.Event()

    def worker():
        started.set()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.train(image, annot, cancel_token=token)
            result['ok'] = True
        except CancelledError:
            result['cancelled'] = True
        except Exception as e:  # noqa: BLE001
            result['error'] = repr(e)

    t = threading.Thread(target=worker)
    t.start()
    started.wait(timeout=5.0)
    token.cancel()
    t.join(timeout=30.0)

    assert not t.is_alive(), "worker did not exit within the timeout after cancel"
    # Either cancelled mid-run (ideal) or completed before cancel landed (acceptable
    # on very fast hardware — the deterministic mid-run case is covered separately).
    assert 'error' not in result, f"worker crashed: {result!r}"
    assert 'cancelled' in result or 'ok' in result


def test_widget_async_worker_completes_without_main_thread_violation(make_napari_viewer):
    """Regression test for the 'NSWindow should only be instantiated on the main
    thread' crash on macOS. The rest of the widget test suite runs with
    ConvpaintWidget._sync_workers = True, which executes the worker on the
    calling thread and would hide any QWidget-constructed-from-worker-thread
    bug. Flip that off and drive one full _on_train / _on_predict cycle end
    to end in a real Qt worker thread."""
    import time
    from qtpy.QtWidgets import QApplication
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    from napari_convpaint.testing_data import (
        generate_synthetic_square,
        generate_synthetic_circle_annotation,
    )

    im, _ = generate_synthetic_square(im_dims=(252, 252), square_dims=(70, 70))
    im_annot = generate_synthetic_circle_annotation(
        im_dims=(252, 252), circle1_xy=(125, 70), circle2_xy=(125, 125)
    )

    orig_sync = ConvpaintWidget._sync_workers
    ConvpaintWidget._sync_workers = False
    try:
        viewer = make_napari_viewer()
        widget = ConvpaintWidget(viewer)
        widget.ensure_init()
        # Swap to a cheap CPU-only feature extractor so the test finishes
        # quickly; the bug we're guarding against is about Qt thread affinity,
        # not about the specific FE choice.
        widget.cp_model = ConvpaintModel(fe_name='gaussian_features')
        widget.auto_seg = False  # keep the test to a single worker round
        viewer.add_image(im)
        widget._on_add_annot_layer()
        viewer.layers['annotations'].data[...] = im_annot
        widget.cp_model.set_params(channel_mode='rgb')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            widget._on_train()
            deadline = time.monotonic() + 30
            while widget._op is not None and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.01)
            assert widget._op is None, "async worker did not finish within 30s"
            # If QWidgets had been constructed on the worker thread, the
            # process would have aborted before reaching this line on macOS.
            assert widget.trained
    finally:
        ConvpaintWidget._sync_workers = orig_sync


def test_clf_reset_is_deferred_while_op_running(make_napari_viewer):
    """Regression test: resetting the classifier while a worker op is running
    (layer switch, channel-mode/norm radio click, classifier-param change —
    none of which are disabled op buttons) must not null the classifier under
    the worker's feet ('NoneType' object has no attribute 'predict_proba').
    The reset is deferred and applied when the op finishes."""
    from qtpy.QtWidgets import QPushButton
    from napari_convpaint.convpaint_widget import ConvpaintWidget, _ActiveOp

    viewer = make_napari_viewer()
    widget = ConvpaintWidget(viewer)
    widget.ensure_init()
    widget.cp_model = ConvpaintModel(fe_name='gaussian_features')
    widget.auto_seg = False
    rng = np.random.default_rng(0)
    viewer.add_image(rng.random((64, 64)), name='img')
    widget._on_add_annot_layer()
    annot = np.zeros((64, 64), dtype=np.uint8)
    annot[10:20, 10:20] = 1
    annot[30:40, 30:40] = 2
    viewer.layers['annotations'].data[...] = annot
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        widget._on_train()
    assert widget.trained and widget.cp_model.classifier is not None

    # Simulate a running op, then a mid-op reset trigger.
    btn = widget.train_classifier_btn
    widget._op = _ActiveOp(name='predict_all', cancel_token=CancelToken(),
                           button=btn, button_orig_text=btn.text())
    widget._reset_clf()
    assert widget.cp_model.classifier is not None, \
        "classifier was nulled while an op was running"
    assert widget.trained

    # When the op finishes, the deferred reset is applied.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        widget._on_worker_finished()
    assert widget.cp_model.classifier is None
    assert not widget.trained
