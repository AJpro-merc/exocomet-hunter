# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Best-effort sky-coordinate resolution, for fetching a real sky image.

Nothing in the real pipeline extracts or stores RA/Dec today -- see
``exocomet.io.download``'s ``fetch_light_curve``, which only reads
``sequence_number``/``mission`` columns from the search result table. This
module is additive and dashboard-only: it runs a second, lightweight
catalogue query (no FITS download) to read ``s_ra``/``s_dec`` from the same
``lightkurve`` search-result table, mirroring the defensive
try/except style already used for
``exocomet.io.download._extract_sectors_or_quarters`` -- coordinates are
cosmetic (they drive a picture, not a science result), so a failure here
must never fail a run.
"""

from __future__ import annotations

import logging

from dashboard import storage

logger = logging.getLogger("dashboard.coordinates")


def resolve_coordinates(target_id: str, mission: str) -> tuple[float, float] | None:
    """Return ``(ra_deg, dec_deg)`` for a target, or ``None`` if unavailable.

    Cached in the dashboard's own SQLite DB (``coordinate_cache`` table) so
    the same star is only searched for once, not once per run it appears in.
    """
    cached = storage.get_cached_coordinates(target_id, mission)
    if cached is not None:
        return cached

    ra: float | None = None
    dec: float | None = None
    try:
        import lightkurve as lk

        search_kwargs: dict[str, object] = {"mission": mission}
        if mission == "TESS":
            search_kwargs["author"] = "SPOC"
            search_kwargs["exptime"] = 120
        else:
            search_kwargs["author"] = "Kepler"
            search_kwargs["cadence"] = "long"

        search = lk.search_lightcurve(target_id, **search_kwargs)
        if len(search) > 0:
            table = search.table
            if "s_ra" in table.colnames and "s_dec" in table.colnames:
                ra = float(table["s_ra"][0])
                dec = float(table["s_dec"][0])
    except Exception as exc:  # noqa: BLE001 - coordinates are cosmetic, never fatal
        logger.debug("could not resolve coordinates for %s: %s", target_id, exc)

    storage.cache_coordinates(target_id, mission, ra, dec)
    if ra is None or dec is None:
        return None
    return ra, dec
