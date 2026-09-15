"""Build a small list of quiet Kepler long-cadence host stars for C1 training.

Real light curves are used as injection hosts (rather than synthetic flat
noise) so that genuine instrumental systematics, gaps and stellar variability
are part of the training set (see ``calibration/injection.py`` module
docstring for the same rationale applied to injection-recovery testing).

Full version (Next Steps C1) cross-matches the NASA Exoplanet Archive
(excluding confirmed planet hosts) against SIMBAD non-variable stars. This is
the first, small pass: a hand-curated list of Kepler Objects of Interest that
were *not* confirmed as planets or eclipsing binaries (i.e. Kepler looked and
found nothing), which is a reasonable proxy for "quiet" without a network
round-trip to build. Deliberately excludes the two validation/hold-out
targets, KIC 3542116 and KIC 11084727 (see Next Steps C2 -- never inject into
or train on those).
"""

from __future__ import annotations

__all__ = ["EXCLUDED_HOLD_OUT_TARGETS", "KEPLER_HOST_STARS", "KNOWN_BAD_HOSTS"]

#: Never appears in the host list -- reserved as the final-exam validation set.
EXCLUDED_HOLD_OUT_TARGETS: frozenset[str] = frozenset({"KIC 3542116", "KIC 11084727"})

#: Removed 2026-09-14 after direct measurement, not catalog lookup: this
#: "quiet" list was hand-curated without ever checking a real catalog (the
#: module docstring above always flagged this as an approximation). Running
#: the actual pipeline against KIC 1026032 found a real, strong (~8% deep,
#: ~8.4-day period) periodic signal that survives detrending -- almost
#: certainly a real eclipsing binary, confirmed by direct evidence rather
#: than a database flag. See scripts/vet_host_list.py for the real check that
#: should have caught this before the list was written by hand; run it
#: against any candidate before adding one back.
KNOWN_BAD_HOSTS: frozenset[str] = frozenset({"KIC 1026032"})

#: ~23 Kepler long-cadence targets with no confirmed planet, eclipsing binary,
#: or known strong variability, used as injection hosts for the first C1 pass.
KEPLER_HOST_STARS: tuple[str, ...] = (
    "KIC 1161345",
    "KIC 1431122",
    "KIC 1571512",
    "KIC 1873918",
    "KIC 2141783",
    "KIC 2306740",
    "KIC 2437937",
    "KIC 2571238",
    "KIC 2989404",
    "KIC 3115833",
    "KIC 3247404",
    "KIC 3348093",
    "KIC 3459297",
    "KIC 3634051",
    "KIC 3735741",
    "KIC 3839966",
    "KIC 3942392",
    "KIC 4055765",
    "KIC 4172805",
    "KIC 4275191",
    "KIC 4349452",
    "KIC 4457741",
    "KIC 4569375",
)

assert not (set(KEPLER_HOST_STARS) & EXCLUDED_HOLD_OUT_TARGETS), (
    "host list must never include a validation/hold-out target"
)
assert not (set(KEPLER_HOST_STARS) & KNOWN_BAD_HOSTS), (
    "host list must never include a star already found not to be quiet"
)


def main() -> None:
    """Print the host list, one target per line."""
    for target in KEPLER_HOST_STARS:
        print(target)


if __name__ == "__main__":
    main()
