"""Bounded, FE-pluggable feature cache for the interactive annotate→predict loop.

The expensive step in interactive segmentation is feature extraction; when the
same image (or z-slice / movie frame) is segmented repeatedly while refining
scribbles, its features can be reused instead of recomputed. This module holds a
generic, feature-extractor-agnostic cache: it stores an *opaque entry* defined
by each FE (e.g. DINO patch tokens — tiny and lossless to upsample), keyed by
`(data signature, extraction signature)`, and owns everything storage-related — LRU
eviction and, crucially, a memory budget so caching a 100-slice stack or a
300-frame movie can never grow unbounded and crash the kernel.

Before storing an entry, its size is checked against the configured cap. If it
does not fit, entries are evicted (see FeatureCache for the order); if it still does not
fit (a single entry larger than the cap), it is simply not cached and the
caller recomputes.

FE-specific behaviour lives in the FeatureExtractor (see the
`extract_native` / `reconstruct_from_native` / `native_to_numpy` / `native_from_numpy` /
`supports_feature_reuse` protocol on the base class): the cache never needs to know what is inside
an entry.
"""
from __future__ import annotations

from collections import OrderedDict


# Default cap (same as the widget default of 2048 MB).
_DEFAULT_MAX_BYTES = 2048 * 10**6  # 2 GB


def _nbytes(entry) -> int:
    """Best-effort byte size of an opaque entry (array, list/tuple of arrays,
    or anything exposing .nbytes, recursing into lists/tuples/dicts).
    Unknown → 0 (treated as free)."""
    if entry is None:
        return 0
    if hasattr(entry, "nbytes"):
        return int(entry.nbytes)
    if isinstance(entry, (list, tuple)):
        return sum(_nbytes(p) for p in entry)
    if isinstance(entry, dict):
        return sum(_nbytes(v) for v in entry.values())
    return 0


class FeatureCache:
    """LRU feature cache bounded by a memory budget.

    Entries of images worked on earlier are evicted first (see new_scope), then, within the
    current image, the ones that were never used, then the least recently used: when a stack
    larger than the cache is processed plane by plane, the first planes stay resident instead
    of being pushed out one by one, so a second pass over the stack reuses them (with plain
    LRU, every plane would evict the next one it needs -> no reuse at all).

    Parameters
    ----------
    max_bytes : int or None
        Hard cap on the cache's size. None → `_DEFAULT_MAX_BYTES` (2 GB).
    """

    def __init__(self, max_bytes: int | None = None):
        self._entries: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (entry, nbytes, used, scope)
        self._total_bytes = 0
        if max_bytes is None:
            max_bytes = _DEFAULT_MAX_BYTES
        self._max_bytes = int(max_bytes)
        self._scope = 0
        self.hits = 0
        self.misses = 0

    # -- budget helpers ----------------------------------------------------

    def _fits(self, nbytes: int) -> bool:
        """Whether adding `nbytes` keeps the cache under its cap."""
        return self._total_bytes + nbytes <= self._max_bytes

    # -- public API --------------------------------------------------------

    def get(self, key):
        """Return the cached entry for `key`, or None."""
        item = self._entries.get(key)
        if item is not None:
            self._entries[key] = (item[0], item[1], True, self._scope) # Used, and part of the current image
            self._entries.move_to_end(key)  # most-recently-used
            self.hits += 1
            return item[0]
        self.misses += 1
        return None

    def put(self, key, entry):
        """Store `entry` under `key` if it fits the budget; else evict (see _evict_one)
        and retry. An entry that can never fit is not cached (the caller recomputes)."""
        if entry is None:
            return
        nbytes = _nbytes(entry)
        # Overwrite of an existing key: drop the old size first.
        if key in self._entries:
            self._total_bytes -= self._entries.pop(key)[1]
        # A single entry larger than the whole cap can never be held.
        if nbytes > self._max_bytes:
            return
        # Evict least-recently-used until the new entry fits.
        while self._entries and not self._fits(nbytes):
            self._evict_one()
        # The key is absent at this point (popped above if present), so
        # assignment appends at the MRU end.
        self._entries[key] = (entry, nbytes, False, self._scope) # (entry, size, used since stored, image)
        self._total_bytes += nbytes

    def _evict_one(self):
        """Evict the oldest entry of an image worked on earlier; failing that, the newest
        never-used entry of the current one, else its least recently used (plain LRU would
        cycle through a stack larger than the cache: every plane evicts the one needed next,
        so nothing is ever reused; see the class docstring)."""
        key = next((k for k in self._entries if self._entries[k][3] < self._scope), None)
        if key is None:
            key = next((k for k in reversed(self._entries) if not self._entries[k][2]), next(iter(self._entries)))
        self._total_bytes -= self._entries.pop(key)[1]

    # -- invalidation / limits --------------------------------------------

    def new_scope(self):
        """Note that the work moved on to another image, whose entries are then kept over the
        earlier ones (which would otherwise stay resident for good once they have been used).
        Ignored while the current image still has entries that were never used: a stack is
        processed plane by plane, i.e. announced as one image after another, and has to stay
        one scope (its planes would otherwise evict each other instead of the earlier image)."""
        if any(not e[2] and e[3] == self._scope for e in self._entries.values()):
            return
        self._scope += 1

    def clear(self):
        """Drop all entries. Entries are content-addressed and never go stale;
        clearing only frees memory."""
        self._entries.clear()
        self._total_bytes = 0

    def set_max_bytes(self, max_bytes: int):
        """Change the cap in place, evicting (see _evict_one) while over it."""
        self._max_bytes = int(max_bytes)
        while self._entries and self._total_bytes > self._max_bytes:
            self._evict_one()

    @property
    def nbytes(self) -> int:
        return self._total_bytes

    def __len__(self):
        return len(self._entries)

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "bytes": self._total_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "max_bytes": self._max_bytes,
        }
