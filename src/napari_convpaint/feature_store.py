"""Persistent feature store for stacks and movies (the disk counterpart of feature_cache.py).

Keeps the native (pre-rescale) features of every processed plane as .npy files in a
user-chosen folder, without eviction: extract a stack once, then train and predict from
the stored features (also across sessions). Entries use the same key as the RAM cache
(content hash of the prepared plane + extraction signature), so the store is self-invalidating:
a changed image or setting simply misses. Hits are memory-mapped, so only the planes
actually used are read from disk.
"""
import hashlib
import json
import os
import shutil
import time
import uuid
import warnings

import numpy as np

try:
    from ._version import version as _VERSION
except ImportError:
    _VERSION = "unknown"


def _release(version):
    """The release part of a version string (1.2.1.dev5+g5f88654 -> 1.2.1), so that stored features
    are only questioned across releases and not across every development build."""
    return str(version).split('+')[0].split('.dev')[0]


_MARKER = "convpaint_feature_store.json" # Marks a folder as a feature store (only such folders are cleared)
_FORMAT = 1 # Layout of the entries; a folder written in another format is not reused
_META = "meta.json" # Per-entry description (shapes, native form)
_DISK_HEADROOM_BYTES = 2 * 1024 ** 3 # Never fill the disk below this much free space


# Default cap, see default_max_bytes().
_DEFAULT_MAX_BYTES = 20 * 10**9  # 20 GB


def default_max_bytes() -> int:
    """The cap used when none is given: moderate, since a store may be forgotten on disk. With
    prune_when_full_days it works as a rolling budget, otherwise storing simply stops once it is
    reached (the disk is protected by _DISK_HEADROOM_BYTES in any case)."""
    return _DEFAULT_MAX_BYTES


def _entry_dirs(folder, include_tmp=False):
    """Folders of the entries (sub-folders with a meta file; optionally also leftover .tmp folders).
    Empty if the folder is gone, so that reading and reporting never fail on it."""
    try:
        dirs = [os.path.join(folder, n) for n in os.listdir(folder)]
    except OSError:
        return []
    return [d for d in dirs if os.path.isfile(os.path.join(d, _META))
            or (include_tmp and d.endswith('.tmp') and os.path.isdir(d))]


def prune_folder(folder, max_age_days):
    """Delete the entries of a feature store folder that have not been used for `max_age_days`
    days and return how many were deleted (an entry is touched whenever it is read, see
    FeatureStore.get). Takes a folder rather than a store, so that one can be tidied up without
    being opened - and a folder that is not there is left alone instead of being created.
    None skips it, 0 days drops every entry, i.e. keeps nothing beyond the session that stored it."""
    if max_age_days is None:
        return 0
    if not os.path.isfile(os.path.join(folder, _MARKER)):
        return 0 # Only ever a folder Convpaint marked as a store of its own, never anything else
    cutoff = time.time() - max_age_days * 86400
    deleted = 0
    for d in _entry_dirs(folder):
        try:
            if os.path.getmtime(d) >= cutoff:
                continue
        except OSError: # Gone in the meantime
            continue
        shutil.rmtree(d, ignore_errors=True) # Stays if locked -> tried again next time
        deleted += 1
    return deleted


class FeatureStore:
    """Feature store in a folder: one sub-folder per entry (plane), no eviction.

    Parameters
    ----------
    folder : str or Path
        Folder to store the features in. Created if it does not exist; an existing
        folder must be empty or a feature store (marked by a marker file).
    max_bytes : int, optional
        Size cap of the store; once reached, nothing more is stored (with a warning).
        None → `default_max_bytes()`. The size is tracked per session, so entries another
        instance adds in the meantime are not counted.
    prune_when_full_days : int, optional
        Make room when the cap is reached by dropping the entries that have not been used
        for that many days (once per time it fills up, see prune). None (default) = keep
        everything and stop storing instead, which is what a folder chosen for keeping wants.
        What this session has written or read is never dropped, however long it runs, so the
        span only says how far beyond this session to reach back; 0 keeps exactly this session's.
    """

    def __init__(self, folder, max_bytes=None, prune_when_full_days=None):
        self.folder = str(folder)
        self.max_bytes = default_max_bytes() if max_bytes is None else max_bytes
        self.prune_when_full_days = prune_when_full_days
        self._session_start = time.time() # Nothing used since then can be dropped to make room
        self.hits = 0
        self.misses = 0
        self.is_full = False # Nothing fits anymore (cap or disk); extracting to store is then pointless
        self._warned_no_write = False
        self._ensure_folder()
        self._nbytes = self._scan_bytes() # Size on disk, kept up to date by put/clear

    # -- the folder ---------------------------------------------------------

    def _ensure_folder(self):
        """Create and check the folder (creatable, a feature store or empty, writable), with one
        clear message per case, so that a bad folder is rejected here and not during extraction.
        Called again if the folder disappeared while the store was in use, hence the marker is
        (re)written every time: a folder without it is not recognized as a store anymore."""
        try:
            os.makedirs(self.folder, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create the folder '{self.folder}' ({e.strerror}).")
        marker = os.path.join(self.folder, _MARKER)
        if os.path.isfile(marker):
            self._check_marker(marker)
        elif os.listdir(self.folder):
            raise ValueError(f"'{self.folder}' is not empty and not a feature store. "
                             "Choose an empty (or not yet existing) folder.")
        try:
            with open(marker, 'w') as f:
                json.dump({"format": _FORMAT, "convpaint": _VERSION}, f)
        except OSError as e:
            raise ValueError(f"No write access to '{self.folder}' ({e.strerror}).")

    def _check_marker(self, marker):
        """Refuse a store of another entry format, and point out features written by another
        convpaint release (they are kept and reused, as extraction usually did not change)."""
        try:
            with open(marker) as f:
                info = json.load(f)
        except (OSError, ValueError): # Unreadable marker: take the folder as a store of this version
            return
        if info.get("format", _FORMAT) != _FORMAT:
            raise ValueError(f"'{self.folder}' is a feature store of another format "
                             f"({info.get('format')} instead of {_FORMAT}). Choose another folder.")
        stored = info.get("convpaint", _VERSION)
        if _release(stored) != _release(_VERSION) and self._entry_dirs():
            warnings.warn(f"Feature store '{self.folder}' was written by convpaint {stored}, "
                          f"running {_VERSION} now. Its features are reused; delete them if "
                          "feature extraction changed in between.")

    def _folder_ready(self):
        """Whether the folder is (still) there, recreating it if it was deleted while the store was
        in use. Warns and stops storing instead of raising in the middle of an extraction."""
        if os.path.isdir(self.folder):
            return True
        try:
            self._ensure_folder()
        except ValueError as e:
            self._warn_no_write(str(e))
            return False
        self._nbytes = self._scan_bytes() # The entries went with the folder
        return True

    def _warn_no_write(self, reason):
        """Warn once that nothing is stored anymore (one warning per plane would flood)."""
        if not self._warned_no_write:
            warnings.warn(f"Feature store '{self.folder}': {reason}")
            self._warned_no_write = True

    def _scan_bytes(self):
        """Size of the stored files, read from the folder itself."""
        total = 0
        for d in self._entry_dirs():
            for n in os.listdir(d):
                try:
                    total += os.path.getsize(os.path.join(d, n))
                except OSError: # Removed in the meantime (another instance)
                    pass
        return total

    # -- keys ---------------------------------------------------------------

    @staticmethod
    def _entry_name(key):
        """Folder name of an entry: <data signature>_<hash of the extraction signature>."""
        data_sig, extraction_sig = key
        extraction_hash = hashlib.blake2b(repr(extraction_sig).encode(), digest_size=8).hexdigest()
        return f"{data_sig}_{extraction_hash}"

    def _entry_dir(self, key):
        return os.path.join(self.folder, self._entry_name(key))

    def _entry_dirs(self, include_tmp=False):
        """Folders of the entries of this store (see the module function of the same name)."""
        return _entry_dirs(self.folder, include_tmp)

    # -- public API ---------------------------------------------------------

    def __contains__(self, key):
        return os.path.isfile(os.path.join(self._entry_dir(key), _META))

    def get(self, key):
        """Return the stored entry for `key` (arrays memory-mapped, copy-on-write), or None.
        A damaged entry (write cut off by a crash, files removed) is deleted and counted as a miss,
        so that its plane is extracted again instead of failing on every later read."""
        entry_dir = self._entry_dir(key)
        meta_path = os.path.join(entry_dir, _META)
        if not os.path.isfile(meta_path):
            self.misses += 1
            return None
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            levels = []
            for i, level_meta in enumerate(meta["levels"]):
                arrays = [np.load(os.path.join(entry_dir, f"l{i}_{j}.npy"), mmap_mode='c')
                          for j in range(level_meta["n_arrays"])]
                levels.append((arrays, tuple(level_meta["scaled_shape"]), tuple(level_meta["cropped_shape"])))
            entry = {"levels": levels, "was_torch": meta["was_torch"]}
        except (OSError, ValueError, KeyError, TypeError): # Truncated, incomplete or foreign files
            shutil.rmtree(entry_dir, ignore_errors=True) # Stays if locked -> skipped again next time
            self._nbytes = self._scan_bytes()
            self.misses += 1
            return None
        try:
            os.utime(entry_dir) # When it was last used, for prune(); read-only folders keep their times
        except OSError:
            pass
        self.hits += 1
        return entry

    def put(self, key, entry):
        """Store `entry` under `key` (no-op if present or if the disk would get too full)."""
        if key in self:
            return
        nbytes = sum(a.nbytes for arrays, _, _ in entry["levels"] for a in arrays)
        if not self._folder_ready():
            return
        if self._nbytes + nbytes > self.max_bytes:
            # Make room once with what went unused, reaching back at least to the start of this
            # session (entries are touched when read), so that a run can never drop what it wrote
            # or read itself, however long it takes and however short the span is
            if self.prune_when_full_days is not None and not self.is_full:
                session_days = (time.time() - self._session_start) / 86400
                self.prune(max(self.prune_when_full_days, session_days))
        if self._nbytes + nbytes > self.max_bytes:
            self.is_full = True
            self._warn_no_write(f"size cap ({self.max_bytes / 1e9:.1f} GB) reached, features are not "
                                "stored anymore (delete stored features or raise the cap).")
            return
        if shutil.disk_usage(self.folder).free - nbytes < _DISK_HEADROOM_BYTES:
            self.is_full = True
            self._warn_no_write("disk (almost) full, features are not stored anymore.")
            return
        # Write into a temporary folder of this write alone and rename it, so that readers never see
        # a partial entry and two instances storing the same plane cannot delete each other's files
        entry_dir = self._entry_dir(key)
        tmp_dir = f"{entry_dir}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp"
        try:
            os.makedirs(tmp_dir)
            meta = {"was_torch": bool(entry["was_torch"]), "levels": []}
            for i, (arrays, scaled_shape, cropped_shape) in enumerate(entry["levels"]):
                for j, a in enumerate(arrays):
                    np.save(os.path.join(tmp_dir, f"l{i}_{j}.npy"), np.ascontiguousarray(a))
                meta["levels"].append({"n_arrays": len(arrays),
                                       "scaled_shape": list(scaled_shape),
                                       "cropped_shape": list(cropped_shape)})
            with open(os.path.join(tmp_dir, _META), 'w') as f:
                json.dump(meta, f)
            try:
                os.rename(tmp_dir, entry_dir)
            except OSError: # Another instance stored the same plane first (identical content)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return
        except OSError as e: # E.g. the folder became unwritable (removed, locked by a sync client)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            self._warn_no_write(f"cannot write ({e.strerror}), features are not stored anymore.")
            return
        try:
            self._nbytes += sum(os.path.getsize(os.path.join(entry_dir, n)) for n in os.listdir(entry_dir))
        except OSError: # Entry removed again (another instance clearing the store)
            self._nbytes = self._scan_bytes()

    def clear(self):
        """Delete all entries (keeps the folder, its marker and anything that is not an entry)."""
        for d in self._entry_dirs(include_tmp=True):
            shutil.rmtree(d, ignore_errors=True)
        self._nbytes = self._scan_bytes() # Not 0: deleting can fail, e.g. while a file is memory-mapped
        self.is_full = self._nbytes >= self.max_bytes
        self._warned_no_write = False

    def prune(self, max_age_days):
        """Delete the entries of this store that have not been used for `max_age_days` days and
        return how many (see prune_folder). Also how the store makes room when it is full, if
        prune_when_full_days is given (see put); otherwise it never does this by itself."""
        deleted = prune_folder(self.folder, max_age_days)
        if deleted:
            self._nbytes = self._scan_bytes()
            self.is_full = self._nbytes >= self.max_bytes
        return deleted

    def set_max_bytes(self, max_bytes):
        """Change the cap in place; nothing is deleted by it, and storing resumes if there is
        room again. A cap below what is stored is full right away, while room that may still be
        too small for the next entry is left to be found out by trying."""
        self.max_bytes = int(max_bytes)
        self.is_full = self._nbytes >= self.max_bytes
        self._warned_no_write = False

    def __len__(self):
        return len(self._entry_dirs())

    @property
    def nbytes(self):
        """Total size of the stored features on disk."""
        return self._nbytes

    def stats(self):
        return {
            "entries": len(self),
            "bytes": self.nbytes,
            "hits": self.hits,
            "misses": self.misses,
            "folder": self.folder,
        }
