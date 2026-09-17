# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Real sky-survey cutout images for whatever star is currently being searched.

Uses ``astroquery.skyview.SkyView`` to fetch a real Digitized Sky Survey (DSS)
image by coordinates -- genuine telescope imagery of the star field, not a
light-curve plot and not a Kepler/TESS pixel postage stamp. Cached to disk by
target, so the same star's image is only fetched once.
"""

from __future__ import annotations

import logging
from pathlib import Path

from dashboard import DASHBOARD_VAR_DIR

logger = logging.getLogger("dashboard.images")

IMAGE_CACHE_DIR = DASHBOARD_VAR_DIR / "image_cache"


def _sanitize(target_id: str) -> str:
    return target_id.replace(" ", "_").replace("/", "_")


def cached_image_path(target_id: str) -> Path:
    return IMAGE_CACHE_DIR / f"{_sanitize(target_id)}.png"


def fetch_sky_image(target_id: str, ra: float, dec: float) -> Path | None:
    """Fetch (or return the cached) real DSS cutout for one target.

    Returns ``None`` on any failure -- an image is a nice-to-have for the
    live view, never something a run should fail over.
    """
    out_path = cached_image_path(target_id)
    if out_path.exists():
        return out_path

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        from astroquery.skyview import SkyView

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        images = SkyView.get_images(position=coord, survey=["DSS"], pixels="400", radius=0.15 * u.deg)
        if not images:
            return None
        data = np.asarray(images[0][0].data, dtype=float)

        # Robust contrast stretch -- raw DSS counts are heavily skewed.
        lo, hi = np.nanpercentile(data, [1.0, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        clipped = np.clip((data - lo) / (hi - lo), 0.0, 1.0)

        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(clipped, cmap="gray", origin="lower")
        ax.set_axis_off()
        fig.patch.set_facecolor("black")
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0, facecolor="black")
        plt.close(fig)
        return out_path
    except Exception as exc:  # noqa: BLE001 - an image is cosmetic, never fatal
        logger.debug("could not fetch sky image for %s: %s", target_id, exc)
        return None
