# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Empirically vet candidate hosts for the C1 training list (Next Steps C1/C6).

Found 2026-09-14: the hand-curated ``KEPLER_HOST_STARS`` list was never
checked against a real catalog, and KIC 1026032 turned out to have a real,
strong (~8% deep, ~8.4-day period) signal that survives detrending -- almost
certainly a real eclipsing binary. Catalog lookups (NASA Exoplanet Archive,
Kepler EB catalog) can miss stars nobody ever flagged; this script instead
runs the actual pipeline and asks the only question that really matters for
this use case: *does our own detector find anything suspicious on this star
with no injection at all?* That is a more directly relevant test than "is
this in a catalog," though slower (one real download per candidate).

Usage
-----
    python scripts/vet_host_list.py KIC 1234567 KIC 2345678 ...
    python scripts/vet_host_list.py --check-current-list
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kepler_host_list import EXCLUDED_HOLD_OUT_TARGETS, KEPLER_HOST_STARS, KNOWN_BAD_HOSTS

from exocomet.core.config import PipelineConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detrend.detrend import detrend_savgol
from exocomet.io.download import fetch_light_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vet_host_list")

#: A quiet host's detrended noise floor should be well under this; a
#: detection above it on unmodified data is a real signal, not noise.
SUSPECT_DEPTH_PPM = 3000.0
#: A genuinely quiet star should have very few >4-sigma excursions across a
#: multi-year baseline; a large count suggests real, recurring structure.
SUSPECT_CANDIDATE_COUNT = 15


def vet_one(target_id: str, mission: str = "Kepler") -> dict[str, object]:
    """Fetch, detrend, and check one target for signs it is not actually quiet."""
    config = PipelineConfig()
    lc = detrend_savgol(
        fetch_light_curve(target_id, mission=mission, discard_after_read=True), config.detrend
    )
    records = AsymmetricDipDetector(config).run(lc)
    max_depth = max((r.event.depth_ppm for r in records), default=0.0)
    is_suspect = max_depth > SUSPECT_DEPTH_PPM or len(records) > SUSPECT_CANDIDATE_COUNT
    verdict = "suspect" if is_suspect else "quiet"
    return {
        "target_id": target_id,
        "n_candidates": len(records),
        "max_depth_ppm": max_depth,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="target IDs to vet, e.g. 'KIC 1234567'")
    parser.add_argument(
        "--check-current-list",
        action="store_true",
        help="vet every star already in KEPLER_HOST_STARS (slow: one real download each)",
    )
    args = parser.parse_args()

    targets = list(args.targets)
    if args.check_current_list:
        targets = list(KEPLER_HOST_STARS)

    if not targets:
        parser.error("give target IDs, or pass --check-current-list")

    for target in targets:
        if target in EXCLUDED_HOLD_OUT_TARGETS:
            logger.warning("%s is a hold-out target -- never vet or use it as a host", target)
            continue
        try:
            result = vet_one(target)
        except Exception as exc:
            logger.warning("%s: could not vet (%s)", target, exc)
            continue
        flag = " <-- already in KNOWN_BAD_HOSTS" if target in KNOWN_BAD_HOSTS else ""
        logger.info(
            "%s: %s (n_candidates=%d, max_depth_ppm=%.0f)%s",
            result["target_id"],
            result["verdict"],
            result["n_candidates"],
            result["max_depth_ppm"],
            flag,
        )


if __name__ == "__main__":
    main()
