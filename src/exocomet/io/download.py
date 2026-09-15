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
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from exocomet.core.types import LightCurveData

logger = logging.getLogger(__name__)

__all__ = ["MastLightCurveSource", "fetch_light_curve", "iter_light_curves"]

DEFAULT_CACHE_DIR = Path("data/raw")


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

    search = lk.search_lightcurve(target_id, **search_kwargs)
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
    collection = search.download_all(download_dir=str(cache), quality_bitmask=quality_bitmask)
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
    """
    for target_id in target_ids:
        try:
            yield fetch_light_curve(
                target_id,
                mission=mission,
                cadence=cadence,
                author=author,
                exptime=exptime,
                cache_dir=cache_dir,
                discard_after_read=discard_after_read,
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
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cadence = cadence
        self.author = author
        self.exptime = exptime
        self.discard_after_read = discard_after_read

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
        )
