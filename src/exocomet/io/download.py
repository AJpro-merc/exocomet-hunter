# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Light-curve retrieval from MAST, with a stream-and-discard option.

The archive is the source of truth. Kepler and TESS photometry is public and
permanent, so keeping a private copy of it is storage spent on something already
guaranteed to exist. For large searches this module can therefore fetch a
target, hand back the arrays, and delete the downloaded FITS immediately — the
working set stays at a few megabytes no matter how many stars are processed,
which is what makes an all-sky-scale search possible on a laptop.

Only this module knows about ``lightkurve``. Everything downstream operates on
:class:`~exocomet.core.types.LightCurveData`, so the science code neither knows
nor cares where the photometry came from.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np

from exocomet.core.types import LightCurveData

logger = logging.getLogger(__name__)

__all__ = [
    "MastLightCurveSource",
    "MastTimeoutError",
    "fetch_light_curve",
    "iter_light_curves",
]

DEFAULT_CACHE_DIR = Path("data/raw")

# -- Network bounds -----------------------------------------------------------
#
# Every MAST call in this module is bounded. Before 2026-09-15 none of them
# were: a fetch for KIC 3542116 sat at zero CPU for nine minutes and had to be
# killed, twice. The 17 FITS files in the cache were all complete and all
# written inside one minute, so the bytes had arrived -- the process was
# blocked on a socket that never returned and had no deadline to hit. These
# constants exist so that a stall fails loudly instead of silently forever.

#: Wall-clock ceiling for the MAST catalogue query. It returns a small table;
#: a minute is already an order of magnitude more than the healthy case.
DEFAULT_SEARCH_TIMEOUT = 60.0

#: Wall-clock ceiling for the bulk FITS fetch. The observed healthy download of
#: 17 Kepler quarters completed in under a minute, so 5 minutes is roughly a
#: 5x margin for a slow link without tolerating an indefinite stall.
DEFAULT_DOWNLOAD_TIMEOUT = 300.0

#: Total attempts per network call (so at most two retries). Bounded on
#: purpose: an unattended batch must not spin forever on one sick target.
DEFAULT_MAX_ATTEMPTS = 3

#: Base backoff in seconds between attempts; doubles each time (5s, 10s).
DEFAULT_RETRY_BACKOFF = 5.0

#: Hard per-target ceiling used by :func:`iter_light_curves`. Trumps the inner
#: budgets above: worst case they sum to ~18 minutes across three attempts of
#: both calls, and no single star in a batch deserves that much of the run.
DEFAULT_TARGET_BUDGET = 900.0

#: Socket-level timeout handed to astroquery for each individual HTTP request.
#: astroquery's own default (600s) applies per request, which for a bulk
#: download means an effectively unbounded total.
ASTROQUERY_TIMEOUT = 120.0

#: astroquery's default page size is 50000 rows; smaller pages mean each
#: request is short enough for ``ASTROQUERY_TIMEOUT`` to be a meaningful bound.
ASTROQUERY_PAGESIZE = 5000

_T = TypeVar("_T")


class MastTimeoutError(TimeoutError):
    """A MAST network call exceeded its wall-clock budget.

    Subclasses :class:`TimeoutError`, so existing ``except OSError`` /
    ``except Exception`` handlers (including ``skip_errors`` in
    :func:`iter_light_curves`) keep working unchanged.
    """


def _configure_astroquery(
    timeout: float = ASTROQUERY_TIMEOUT, pagesize: int = ASTROQUERY_PAGESIZE
) -> None:
    """Pin astroquery's MAST timeouts to bounded values.

    Lazy and local, matching the ``import lightkurve as lk`` style below:
    astroquery is a heavy import and is only needed on the network path.
    Best-effort -- the private ``_portal_api_connection`` /
    ``_service_api_connection`` attributes are the only place the per-request
    timeout actually lives, and their names have moved between astroquery
    releases. Failing to tune them must never fail a download, because the
    thread-based budget below is the real backstop.
    """
    try:
        from astroquery.mast import Observations, conf

        conf.timeout = timeout
        conf.pagesize = pagesize
        for connection in (
            getattr(Observations, "_portal_api_connection", None),
            getattr(Observations, "_service_api_connection", None),
        ):
            if connection is None:
                continue
            if hasattr(connection, "TIMEOUT"):
                connection.TIMEOUT = timeout
            if hasattr(connection, "PAGESIZE"):
                connection.PAGESIZE = pagesize
    except Exception as exc:  # pragma: no cover - depends on astroquery internals
        logger.debug("could not configure astroquery timeouts: %s", exc)


def _call_with_timeout(func: Callable[[], _T], timeout: float, description: str) -> _T:
    """Run ``func`` on a daemon thread and give up waiting after ``timeout``.

    A daemon thread rather than ``concurrent.futures`` on purpose: a stalled
    MAST socket cannot be cancelled from outside, so the worker may still be
    alive when we stop waiting. Daemon threads do not block interpreter exit,
    whereas ``ThreadPoolExecutor`` joins its workers at shutdown and would turn
    a bounded call back into a hang at the end of the process.

    The honest limitation: this bounds *how long the caller waits*, and the
    orphaned thread keeps holding its socket until astroquery's own
    ``ASTROQUERY_TIMEOUT`` trips. That is why both layers are set.
    """
    box: dict[str, object] = {}

    def runner() -> None:
        try:
            box["value"] = func()
        except BaseException as exc:  # re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=runner, name=f"mast-{description}", daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise MastTimeoutError(
            f"{description} exceeded its {timeout:.0f}s budget and was abandoned "
            "(the underlying request may still be stalled in the background)"
        )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]  # type: ignore[return-value]


def _call_with_retries(
    func: Callable[[], _T],
    timeout: float,
    description: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff: float = DEFAULT_RETRY_BACKOFF,
    cleanup: Callable[[], None] | None = None,
) -> _T:
    """Bounded retry around :func:`_call_with_timeout`: never infinite.

    ``cleanup``, if given, runs after a failed attempt and before the next
    one (never after the last, exhausted attempt). Found necessary
    2026-09-15 by direct observation: a download interrupted partway leaves a
    truncated FITS file cached on disk, and simply calling ``download_all``
    again does not fix it -- ``astropy``/``lightkurve`` finds the file already
    present at its cache path and does not know it is broken, so every retry
    fails identically on the same corrupt file instead of re-fetching it. A
    retry with no cleanup is not actually a retry in that case, just the same
    failure repeated on a timer. See ``_discard_cached`` for what the caller
    passes at the ``download_all`` call site.
    """
    attempts = max(1, int(max_attempts))
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _call_with_timeout(func, timeout, description)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            delay = backoff * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s -- retrying in %.0fs",
                description,
                attempt,
                attempts,
                exc,
                delay,
            )
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:  # noqa: BLE001 - cleanup failing must not mask last_exc
                    logger.warning("cleanup before retry of %s raised", description, exc_info=True)
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _to_light_curve_data(
    lc: object, target_id: str, mission: str, extra_meta: dict[str, object] | None = None
) -> LightCurveData:
    """Convert a stitched ``lightkurve`` object into our own container.

    Cadences with non-finite time or flux are dropped: they carry no
    information and would otherwise propagate NaNs through every downstream
    statistic.
    """
    time = np.asarray(lc.time.value, dtype=np.float64)  # type: ignore[attr-defined]
    flux = np.asarray(lc.flux.value, dtype=np.float64)  # type: ignore[attr-defined]
    flux_err = np.asarray(lc.flux_err.value, dtype=np.float64)  # type: ignore[attr-defined]

    good = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    time, flux, flux_err = time[good], flux[good], flux_err[good]

    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    # Guard against duplicate timestamps from overlapping quarters/sectors,
    # which would break the strictly-increasing invariant of LightCurveData.
    # Logged rather than silently dropped -- a large count here is itself a
    # sign that mismatched products (see A2 below) got stitched together.
    unique = np.concatenate(([True], np.diff(time) > 0))
    n_duplicate = int(np.count_nonzero(~unique))
    if n_duplicate:
        logger.info("dropped %d duplicate-timestamp cadence(s) for %s", n_duplicate, target_id)
    time, flux, flux_err = time[unique], flux[unique], flux_err[unique]

    median = float(np.median(flux))
    if median <= 0 or not np.isfinite(median):
        raise ValueError(f"{target_id}: cannot normalise, median flux is {median}")

    meta: dict[str, object] = {
        "n_dropped_nonfinite": int(np.count_nonzero(~good)),
        "n_dropped_duplicate_timestamp": n_duplicate,
        "median_raw_flux": median,
        "source": "MAST",
    }
    if extra_meta:
        meta.update(extra_meta)

    return LightCurveData(
        target_id=target_id,
        mission=mission,
        time=time,
        flux=flux / median,
        flux_err=flux_err / median,
        meta=meta,
    )


def _extract_sectors_or_quarters(search: object) -> list[int]:
    """Best-effort extraction of the sector/quarter numbers actually used.

    Defensive: lightkurve's search-result table schema has shifted across
    versions, and this is provenance-only metadata -- never worth failing a
    download over.
    """
    import re

    try:
        table = search.table  # type: ignore[attr-defined]
        if "sequence_number" in table.colnames:
            return sorted({int(v) for v in table["sequence_number"] if v is not None})
        numbers: set[int] = set()
        for entry in table["mission"]:
            match = re.search(r"(\d+)\s*$", str(entry))
            if match:
                numbers.add(int(match.group(1)))
        return sorted(numbers)
    except Exception:
        return []


def fetch_light_curve(
    target_id: str,
    mission: str = "Kepler",
    cadence: str = "long",
    author: str | None = None,
    exptime: int | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    discard_after_read: bool = False,
    quality_bitmask: str = "default",
    search_timeout: float = DEFAULT_SEARCH_TIMEOUT,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> LightCurveData:
    """Download, stitch and normalise all available photometry for one target.

    Parameters
    ----------
    target_id
        Catalogue identifier, e.g. ``"KIC 3542116"`` or ``"TIC 260128333"``.
    mission
        ``"Kepler"``, ``"K2"`` or ``"TESS"``.
    cadence
        ``"long"`` or ``"short"`` for Kepler/K2; ignored for TESS (see
        ``exptime`` instead).
    author
        Pin to a single pipeline. Defaults to ``"SPOC"`` for TESS and
        ``"Kepler"`` for Kepler/K2 when not given -- see the module docstring
        section on A2 below for why a pin matters.
    exptime
        Pin to a single exposure time in seconds, TESS only. Defaults to
        ``120`` (SPOC's standard 2-minute cadence) when not given.
    cache_dir
        Directory for downloaded FITS files.
    discard_after_read
        When ``True``, delete this target's cached files once the arrays have
        been extracted. Use for large batches, where re-downloading later is
        cheaper than storing terabytes locally.
    quality_bitmask
        Passed to ``lightkurve``; the default mask removes cadences the mission
        pipeline flagged as bad.
    search_timeout, download_timeout
        Wall-clock ceilings in seconds for the catalogue query and the bulk
        FITS fetch respectively. Exceeding one raises :class:`MastTimeoutError`
        rather than blocking forever.
    max_attempts, retry_backoff
        Bounded retry policy applied to each of those two calls: at most
        ``max_attempts`` tries, sleeping ``retry_backoff * 2**(n-1)`` seconds
        between them.

    Returns
    -------
    LightCurveData
        Normalised, strictly time-ordered photometry with non-finite cadences
        removed. ``meta`` records ``author``, ``exptime`` (``None`` for
        non-TESS), and ``sectors_or_quarters``.

    Raises
    ------
    FileNotFoundError
        If the archive returns no products for this target, mission, author
        and exptime combination.
    MastTimeoutError
        If the catalogue query or the bulk download exceeds its budget on
        every attempt.

    Notes
    -----
    **Why pin author/exptime at all (Next Steps A2).** A plain
    ``lk.search_lightcurve(target, mission="TESS")`` can return dozens of
    products from different pipelines (QLP, SPOC, TARS, TASOC, TESS-SPOC,
    TGLC, ...) at cadences from 20s to 1800s, and blindly stitching all of
    them together causes two distinct real problems, both found this session:
    (1) a detrending/baseline window sized for one cadence is a wildly wrong
    physical duration at another (see A1), and (2) at least one HLSP pipeline
    (TARS) reports flux as an already zero-centered residual rather than raw
    counts, so stitching it in can divide by a near-zero median and corrupt
    the normalisation outright, not just misalign cadences. Pinning to a
    single ``author``/``exptime`` avoids both at once, rather than only
    separating by cadence.
    """
    import lightkurve as lk

    _configure_astroquery()

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    search_kwargs: dict[str, object] = {"mission": mission}
    if mission == "TESS":
        resolved_author = author or "SPOC"
        resolved_exptime = exptime if exptime is not None else 120
        search_kwargs["author"] = resolved_author
        search_kwargs["exptime"] = resolved_exptime
    else:
        resolved_author = author or "Kepler"
        resolved_exptime = None
        search_kwargs["cadence"] = cadence
        search_kwargs["author"] = resolved_author

    search = _call_with_retries(
        lambda: lk.search_lightcurve(target_id, **search_kwargs),
        timeout=search_timeout,
        description=f"search_lightcurve({target_id!r})",
        max_attempts=max_attempts,
        backoff=retry_backoff,
    )
    if len(search) == 0:
        raise FileNotFoundError(
            f"no {mission} light curves found for {target_id} "
            f"(author={resolved_author!r}, exptime={resolved_exptime!r})"
        )

    logger.info(
        "downloading %d products for %s (%s, author=%s, exptime=%s)",
        len(search),
        target_id,
        mission,
        resolved_author,
        resolved_exptime,
    )
    collection = _call_with_retries(
        lambda: search.download_all(download_dir=str(cache), quality_bitmask=quality_bitmask),
        timeout=download_timeout,
        description=f"download_all({target_id!r})",
        max_attempts=max_attempts,
        backoff=retry_backoff,
        cleanup=lambda: _discard_cached(cache, target_id),
    )
    # .stitch() is local CPU work on already-downloaded arrays -- no network,
    # so deliberately left unbounded.
    stitched = collection.stitch()

    data = _to_light_curve_data(
        stitched,
        target_id,
        mission,
        extra_meta={
            "n_products": len(search),
            "author": resolved_author,
            "exptime": resolved_exptime,
            "sectors_or_quarters": _extract_sectors_or_quarters(search),
        },
    )

    if discard_after_read:
        _discard_cached(cache, target_id)

    return data


def _discard_cached(cache: Path, target_id: str) -> None:
    """Remove cached download files, freeing disk immediately.

    Wipes the entire ``mastDownload`` tree under ``cache`` rather than trying
    to match subdirectories against ``target_id`` -- worth explaining why.
    ``lightkurve`` names each product's cache directory after its own
    pipeline-internal identifier, never the catalog string a caller searched
    with: a Kepler product is ``kplr001161345_lc_...`` (Kepler's own
    zero-padded ID, no ``"kic"`` anywhere), and a TESS HLSP product embeds a
    zero-padded TIC number mid-filename (``hlsp_tars_tess_ffi_s0003-0000000054003409_...``)
    -- which is unrelated to a resolved common name like ``"beta Pic"``, itself
    carrying no digits at all. An earlier version of this function matched a
    normalised slug of ``target_id`` against directory names and matched
    nothing, for any target, Kepler or TESS: ``discard_after_read`` had
    silently been a no-op since it was written (found 2026-09-14, when 166 MB
    of "already discarded" downloads had actually accumulated on disk).

    Wiping the whole tree is correct as long as targets are processed one at a
    time -- true of every current caller (:func:`iter_light_curves`,
    ``scripts/monitor_new_data.py``, :class:`MastLightCurveSource`). A future
    parallel-worker batch (Next Steps E4) will need per-target cache
    subdirectories instead of this blunter approach.
    """
    tree = cache / "mastDownload"
    if not tree.exists():
        return
    removed = sum(1 for _ in tree.iterdir())
    shutil.rmtree(tree, ignore_errors=True)
    if removed:
        logger.debug("discarded cached mastDownload tree (%d entries) after %s", removed, target_id)


def iter_light_curves(
    target_ids: list[str],
    mission: str = "Kepler",
    cadence: str = "long",
    author: str | None = None,
    exptime: int | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    discard_after_read: bool = True,
    skip_errors: bool = True,
    per_target_timeout: float | None = DEFAULT_TARGET_BUDGET,
    search_timeout: float = DEFAULT_SEARCH_TIMEOUT,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Iterator[LightCurveData]:
    """Yield light curves one at a time, discarding each after it is consumed.

    This is the memory- and disk-safe way to process a large target list: at any
    moment only one star's photometry exists locally.

    Parameters
    ----------
    author, exptime
        Passed through to :func:`fetch_light_curve` -- see its docstring for
        the A2 pinning defaults (SPOC/120s for TESS).
    skip_errors
        When ``True``, log and skip targets that fail to download rather than
        aborting the whole batch — one unavailable star should not end a run
        over thousands.
    per_target_timeout
        Hard wall-clock budget in seconds for one target, covering the whole
        of :func:`fetch_light_curve` including its internal retries. ``None``
        disables the outer budget and leaves only the per-call ones. This is
        what stops a single hung star from stalling an entire batch: before
        2026-09-15 ``skip_errors`` could only skip a target that *failed*, and
        a target that simply never returned held the loop forever.

    Notes
    -----
    A target abandoned at ``per_target_timeout`` leaves its worker thread
    running. If that zombie later completes with ``discard_after_read=True``
    it will wipe the shared ``mastDownload`` tree, possibly underneath the
    next target's download. The blast radius is a redundant re-download rather
    than corrupt science, but per-target cache subdirectories (Next Steps E4)
    would remove it entirely.
    """
    for target_id in target_ids:

        def _fetch(target_id: str = target_id) -> LightCurveData:
            return fetch_light_curve(
                target_id,
                mission=mission,
                cadence=cadence,
                author=author,
                exptime=exptime,
                cache_dir=cache_dir,
                discard_after_read=discard_after_read,
                search_timeout=search_timeout,
                download_timeout=download_timeout,
                max_attempts=max_attempts,
                retry_backoff=retry_backoff,
            )

        try:
            if per_target_timeout is None:
                yield _fetch()
            else:
                yield _call_with_timeout(
                    _fetch, per_target_timeout, f"fetch_light_curve({target_id!r})"
                )
        except Exception as exc:
            if not skip_errors:
                raise
            logger.warning("skipping %s: %s", target_id, exc)


class MastLightCurveSource:
    """MAST-backed :class:`~exocomet.core.interfaces.LightCurveSource`."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        cadence: str = "long",
        author: str | None = None,
        exptime: int | None = None,
        discard_after_read: bool = False,
        search_timeout: float = DEFAULT_SEARCH_TIMEOUT,
        download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cadence = cadence
        self.author = author
        self.exptime = exptime
        self.discard_after_read = discard_after_read
        self.search_timeout = search_timeout
        self.download_timeout = download_timeout
        self.max_attempts = max_attempts

    def fetch(self, target_id: str, mission: str) -> LightCurveData:
        """Retrieve one target's photometry from the archive."""
        return fetch_light_curve(
            target_id,
            mission=mission,
            cadence=self.cadence,
            author=self.author,
            exptime=self.exptime,
            cache_dir=self.cache_dir,
            discard_after_read=self.discard_after_read,
            search_timeout=self.search_timeout,
            download_timeout=self.download_timeout,
            max_attempts=self.max_attempts,
        )
