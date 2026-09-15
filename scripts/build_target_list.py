# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Build a catalogue-driven target list of debris-disc stars, plus a control group.

Why this script exists
-----------------------
``config/watchlist.txt`` and ``config/tess_training_hosts.txt`` are currently
33 hand-picked stars total. An hourly ingestion pipeline exhausts a list that
small in under two runs. This script queries real astronomical catalogues
instead, so the target list can grow to the size a recurring pipeline needs.

An exocomet needs a reservoir of icy bodies to come from, so the physically
motivated place to look is stars known to host a **debris disc** (the
leftover-planetesimal belt a comet would be dislodged from). This script:

1. Pulls a published, already-vetted debris-disc catalogue from VizieR.
2. Cross-matches each disc star's sky position against the TESS Input
   Catalog (TIC) via ``astroquery.mast.Catalogs`` -- which, for stars in the
   original Kepler field, *also* carries the star's KIC number as a
   cross-identification column, giving a Kepler ID with no separate lookup.
3. Confirms real, downloadable Kepler long-cadence or TESS SPOC 120s
   photometry exists for the resolved id (an unobservable target is useless
   to an ingestion pipeline), via the same ``lightkurve`` search machinery
   used elsewhere in this repo (see ``exocomet/io/download.py``).
4. For every disc star kept, picks a **matched control star** nearby on sky
   with a similar TESS magnitude (and, when known, a similar distance) and
   no known disc -- for Next Steps F4 ("candidate rate for debris-disc stars
   vs a matched control group").
5. Re-enforces, loudly (with ``assert``, not just a comment -- see
   ``enforce_hold_out_exclusion`` below), that neither of the two validation
   hold-out stars nor the one known-bad host from
   ``scripts/build_kepler_host_list.py`` can end up in the output, in either
   mission's list, whether reached directly or through a TIC->KIC
   cross-match. Those three stars are the final exam for the whole pipeline;
   training on any of them silently invalidates every result.

Catalogue choice
-----------------
The debris-disc source is Cotten & Song (2016), ApJS 225, 15, VizieR id
``J/ApJS/225/15``, "A Uniformly Vetted Compilation of Debris Disk
Candidates". It was picked over a general infrared-excess survey because it
is *already* a curated, literature-cross-checked list of confirmed/candidate
disc hosts spanning a wide range of spectral types and magnitudes (rather
than a catalogue of raw photometric excesses that would need its own
vetting pass first) -- including several stars already in
``config/watchlist.txt`` (e.g. beta Pic, Fomalhaut, HR 4796), which is a
reasonable sanity check that the catalogue is pulling the right kind of
star. Override with ``--vizier-catalog`` to try a different one (e.g. a
Chen et al. Spitzer IRS disc survey) without touching the code.

Control-group matching rule (documented per the task, not sophisticated by
design -- see :func:`select_control_star`)
--------------------------------------------------------------------------
For each disc star, candidate controls are every other star the TIC cone
search around the same sky position returns. A candidate is accepted if:

* it is not the disc star itself, not a disc-catalogue member already
  resolved earlier in this run, and not a hold-out/known-bad KIC target;
* its TESS magnitude is within ``--mag-tolerance`` (default 0.5 mag) of the
  disc star's -- a same-brightness-bin proxy for comparable photometric
  noise floor, since detectability of a comet-depth dip is magnitude-driven;
* when both stars have a TIC parallax-derived distance, the candidate's
  distance is within ``--dist-tolerance-frac`` (default 30%) of the disc
  star's -- same distance regime, so an actual disc (if the control secretly
  had one) would be roughly as resolvable/relevant.

Among survivors, the control closest in magnitude to the disc star is kept
(ties broken by angular separation). This is a documented, defensible rule,
not a rigorous spectral-type match -- good enough for a population-level
comparison, not for a paired case study.

Caching
-------
Every VizieR/MAST/lightkurve call result is cached as JSON under
``data/target_list_cache.json`` (``data/`` is already gitignored -- see
``.gitignore``, "light curves are re-downloadable... never committed"; the
same logic applies to catalogue lookups). Each cache section has its own
staleness TTL (catalogue membership barely changes; photometry availability
can grow as new sectors/quarters land):

* debris-disc catalogue rows: 30 days
* TIC cross-match / control-candidate cone searches: 14 days
* photometry-availability checks: 7 days

Pass ``--refresh-cache`` to ignore all of the above and re-query everything.

Output format
--------------
Two plain-text target lists, one target per line, same convention as
``config/watchlist.txt`` (``#``-comments and blank lines ignored, per
``generate_training_labels.load_host_list``'s ``raw.split("#", 1)[0].strip()``
parsing):

* ``config/target_list_kepler.txt``
* ``config/target_list_tess.txt``

Each line is ``TARGET_ID  # role=... pair=... host=... catalog=...`` -- the
id is the only thing the loader reads; everything after ``#`` is there for a
human skimming the file. ``role`` is ``disc`` or ``control``; two lines
sharing the same ``pair`` id are the disc star and its matched control for
Next Steps F4's population comparison. Full numeric metadata (magnitude,
temperature, distance, separation) for every row lives alongside in
``config/target_list_metadata.csv`` -- open that one in any spreadsheet app
to see the actual matching numbers behind each pair.

Usage
-----
    python scripts/build_target_list.py --max-targets 20
    python scripts/build_target_list.py --max-targets 5 --dry-run
    python scripts/build_target_list.py --max-targets 20 --refresh-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kepler_host_list import EXCLUDED_HOLD_OUT_TARGETS, KNOWN_BAD_HOSTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_target_list")

__all__ = [
    "DEFAULT_CACHE_PATH",
    "DEFAULT_OUT_DIR",
    "DEFAULT_VIZIER_CATALOG",
    "EXCLUDED_HOLD_OUT_KIC_NUMBERS",
    "KNOWN_BAD_KIC_NUMBERS",
    "DiscCatalogEntry",
    "TargetRow",
    "TicMatch",
    "build_target_list",
    "enforce_hold_out_exclusion",
    "is_forbidden_kic",
    "kic_number",
    "select_control_star",
    "write_target_lists",
]

DEFAULT_VIZIER_CATALOG = "J/ApJS/225/15"
DEFAULT_CACHE_PATH = Path("data/target_list_cache.json")
DEFAULT_OUT_DIR = Path("config")

#: Staleness budgets, in days, per cache section. See module docstring.
CACHE_TTL_DAYS: dict[str, float] = {
    "disc_catalog": 30.0,
    "tic_crossmatch": 14.0,
    "control_candidates": 14.0,
    "photometry": 7.0,
}

# -- Hold-out / known-bad exclusion -------------------------------------------
#
# Re-derived (not re-typed) from build_kepler_host_list.py so there is exactly
# one place these identifiers are spelled out. Converted to bare KIC numbers
# because the TIC cross-match below returns a numeric KIC column, not the
# "KIC 1234567" string form -- comparing numbers is what lets this catch a
# star reached via its TIC/coordinate match, not just a literal name match.


#: Matches "KIC 1234567", "kic1234567", "KIC1234567" -- deliberately requires
#: the "KIC" prefix (rather than grabbing any trailing digit run from any
#: string) so a *TIC* id string is never misparsed as a KIC number just
#: because its digits happen to coincide with a forbidden KIC number. A bare
#: int is still accepted directly -- that is how a real KIC number arrives
#: from the TIC catalogue's own ``KIC`` cross-id column.
_KIC_STRING_PATTERN = re.compile(r"^\s*kic\s*(\d+)\s*$", re.IGNORECASE)


def kic_number(identifier: str | int | None) -> int | None:
    """Parse a KIC identifier to its bare integer, or ``None`` if not KIC-shaped.

    Accepts ``"KIC 3542116"``, ``"kic3542116"``, the bare int ``3542116``, or
    ``None``. A string that is not ``"KIC <digits>"`` -- including a ``"TIC
    ..."`` id string, even one whose digits happen to coincide with a
    forbidden KIC number -- returns ``None``: only a genuine KIC identifier
    should ever compare equal to a KIC exclusion.
    """
    if identifier is None:
        return None
    if isinstance(identifier, bool):  # bool is an int subclass; guard explicitly
        return None
    if isinstance(identifier, int):
        return identifier
    match = _KIC_STRING_PATTERN.match(str(identifier))
    return int(match.group(1)) if match else None


#: The two validation/hold-out stars, as bare KIC numbers.
EXCLUDED_HOLD_OUT_KIC_NUMBERS: frozenset[int] = frozenset(
    n for n in (kic_number(t) for t in EXCLUDED_HOLD_OUT_TARGETS) if n is not None
)
#: The one known-bad host, as a bare KIC number.
KNOWN_BAD_KIC_NUMBERS: frozenset[int] = frozenset(
    n for n in (kic_number(t) for t in KNOWN_BAD_HOSTS) if n is not None
)

assert len(EXCLUDED_HOLD_OUT_KIC_NUMBERS) == len(EXCLUDED_HOLD_OUT_TARGETS), (
    "every EXCLUDED_HOLD_OUT_TARGETS entry must parse to a KIC number"
)
assert len(KNOWN_BAD_KIC_NUMBERS) == len(KNOWN_BAD_HOSTS), (
    "every KNOWN_BAD_HOSTS entry must parse to a KIC number"
)


def is_forbidden_kic(identifier: str | int | None) -> bool:
    """Report whether ``identifier`` (any KIC-shaped form) is hold-out or known-bad."""
    n = kic_number(identifier)
    if n is None:
        return False
    return n in EXCLUDED_HOLD_OUT_KIC_NUMBERS or n in KNOWN_BAD_KIC_NUMBERS


# -- Data model ----------------------------------------------------------------


@dataclass(frozen=True)
class DiscCatalogEntry:
    """One row resolved from the debris-disc VizieR catalogue."""

    name: str
    ra_deg: float
    dec_deg: float
    sptype: str | None = None
    catalog_ref: str = DEFAULT_VIZIER_CATALOG


@dataclass(frozen=True)
class TicMatch:
    """One TIC row, from either a single-star cross-match or a cone search."""

    tic_id: int
    ra_deg: float
    dec_deg: float
    tmag: float | None = None
    teff: float | None = None
    plx_mas: float | None = None
    kic_id: int | None = None
    separation_arcsec: float | None = None

    @property
    def distance_pc(self) -> float | None:
        """Parallax-derived distance, or ``None`` if the parallax is unusable."""
        if self.plx_mas is None or self.plx_mas <= 0:
            return None
        return 1000.0 / self.plx_mas


@dataclass(frozen=True)
class TargetRow:
    """One line of the final output: a single mission-specific target id."""

    target_id: str
    mission: str  # "Kepler" or "TESS"
    role: str  # "disc" or "control"
    pair_id: str
    host_name: str
    catalog_ref: str
    tmag: float | None = None
    teff: float | None = None
    distance_pc: float | None = None
    separation_arcsec: float | None = None


@dataclass(frozen=True)
class ResolvedTarget:
    """A disc-catalogue entry, resolved to its TIC (and possibly KIC) identity."""

    disc_entry: DiscCatalogEntry
    tic: TicMatch


def is_forbidden_target(resolved: ResolvedTarget) -> bool:
    """Report whether this resolved disc star is the hold-out or known-bad host."""
    return is_forbidden_kic(resolved.tic.kic_id)


# -- Cache ----------------------------------------------------------------------


def _load_cache(path: Path) -> dict[str, Any]:
    """Read the JSON cache, tolerating a missing or corrupt file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cache at %s unreadable (%s); starting fresh", path, exc)
        return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    """Persist the JSON cache, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _cache_get(
    cache: dict[str, Any], section: str, key: str, ttl_days: float, refresh: bool
) -> Any | None:
    """Return a cached value if present and not stale, else ``None``."""
    if refresh:
        return None
    entry = cache.get(section, {}).get(key)
    if entry is None:
        return None
    try:
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
    except (KeyError, ValueError):
        return None
    if datetime.now(UTC) - fetched_at > timedelta(days=ttl_days):
        return None
    return entry["data"]


def _cache_set(cache: dict[str, Any], section: str, key: str, data: Any) -> None:
    """Write one cache entry with the current UTC timestamp."""
    cache.setdefault(section, {})[key] = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "data": data,
    }


# -- VizieR: debris-disc catalogue ----------------------------------------------

#: Column-name candidates, in priority order, for each logical field. VizieR
#: table schemas vary catalogue to catalogue; trying several plausible names
#: rather than hardcoding one is what keeps this working if ``--vizier-catalog``
#: points at a differently-organised table.
_NAME_COLS = ("Name", "SimbadName", "MAIN_ID", "HD", "TYC", "_2MASS", "recno")
_RA_COLS = ("RAJ2000", "_RAJ2000", "RA_ICRS", "RAdeg", "ra")
_DEC_COLS = ("DEJ2000", "_DEJ2000", "DE_ICRS", "DEdeg", "dec")
_SPTYPE_COLS = ("SpType", "SpT", "Sp")


def _first_present(colnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    """Return the first of ``candidates`` that is actually in ``colnames``."""
    for name in candidates:
        if name in colnames:
            return name
    return None


def _safe_float(value: Any) -> float | None:
    """Coerce a table cell to ``float``, treating masked/blank/NaN as missing."""
    if value is None:
        return None
    if bool(getattr(value, "mask", False)):  # astropy masked scalar
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # filter NaN without importing math/np here


def _safe_int(value: Any) -> int | None:
    """Coerce a table cell to ``int``, treating masked/blank/NaN as missing."""
    f = _safe_float(value)
    return int(f) if f is not None else None


def _parse_angle(value: Any, *, hour_angle: bool) -> float | None:
    """Coerce a table cell to decimal degrees, accepting sexagesimal strings.

    Most VizieR catalogues publish ``RAJ2000``/``DEJ2000`` as plain decimal
    degrees (what :func:`_safe_float` handles directly), but some -- Cotten &
    Song 2016 (``J/ApJS/225/15``), the debris-disc catalogue this script
    defaults to, among them -- publish them as sexagesimal strings instead
    (``"00 04 20.33"`` / ``"-29 16 07.7"``). ``_safe_float`` returns ``None``
    for those, which without this fallback silently dropped *every* row of
    the default catalogue as having "no usable RA/Dec" -- caught by actually
    running the query rather than trusting the column names. Tries a plain
    float first (cheap, and correct for the common case), then falls back to
    :class:`astropy.coordinates.Angle` sexagesimal parsing (hour angle for
    RA, degrees for Dec).
    """
    f = _safe_float(value)
    if f is not None:
        return f
    if value is None or bool(getattr(value, "mask", False)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        import astropy.units as u
        from astropy.coordinates import Angle

        angle = Angle(text, unit=u.hourangle if hour_angle else u.deg)
        deg = float(angle.degree)
    except Exception:
        return None
    return deg if deg == deg else None  # filter NaN


def parse_vizier_disc_table(table: Any, catalog_ref: str) -> list[DiscCatalogEntry]:
    """Turn an astropy ``Table`` from the disc catalogue into entries.

    Pure (no network): given any object with a ``colnames`` sequence and
    row-indexable access to those columns (an astropy ``Table`` in
    production, a small hand-built fixture in tests), returns one
    :class:`DiscCatalogEntry` per row that has a usable position. Rows
    without a resolvable RA/Dec are dropped and counted in a log line --
    every entry here is about to be searched at MAST by coordinates, so a
    row with no position is not a candidate, it is unusable.
    """
    colnames = list(table.colnames)
    name_col = _first_present(colnames, _NAME_COLS)
    ra_col = _first_present(colnames, _RA_COLS)
    dec_col = _first_present(colnames, _DEC_COLS)
    sptype_col = _first_present(colnames, _SPTYPE_COLS)

    if ra_col is None or dec_col is None:
        raise ValueError(
            f"could not find RA/Dec columns in {catalog_ref}; saw columns {colnames}"
        )

    entries: list[DiscCatalogEntry] = []
    n_dropped = 0
    for i, row in enumerate(table):
        ra = _parse_angle(row[ra_col], hour_angle=True)
        dec = _parse_angle(row[dec_col], hour_angle=False)
        if ra is None or dec is None:
            n_dropped += 1
            continue
        name = str(row[name_col]).strip() if name_col is not None else f"{catalog_ref}#{i}"
        sptype = str(row[sptype_col]).strip() if sptype_col is not None else None
        entries.append(
            DiscCatalogEntry(
                name=name or f"{catalog_ref}#{i}",
                ra_deg=ra,
                dec_deg=dec,
                sptype=sptype or None,
                catalog_ref=catalog_ref,
            )
        )
    if n_dropped:
        logger.warning("%s: dropped %d row(s) with no usable RA/Dec", catalog_ref, n_dropped)
    return entries


def _query_vizier_disc_catalog(catalog_id: str, row_limit: int) -> list[DiscCatalogEntry]:
    """Network call: fetch and parse the debris-disc catalogue from VizieR."""
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=["**"], row_limit=row_limit)
    tables = vizier.get_catalogs(catalog_id)
    if len(tables) == 0:
        raise ValueError(f"VizieR returned no tables for catalogue {catalog_id!r}")
    if len(tables) > 1:
        logger.info(
            "%s: VizieR returned %d tables; using the first (%s)",
            catalog_id,
            len(tables),
            tables[0].meta.get("name", "?"),
        )
    return parse_vizier_disc_table(tables[0], catalog_id)


def fetch_disc_catalog(
    cache: dict[str, Any],
    cache_path: Path,
    catalog_id: str = DEFAULT_VIZIER_CATALOG,
    row_limit: int = -1,
    refresh: bool = False,
) -> list[DiscCatalogEntry]:
    """Cache and return :func:`_query_vizier_disc_catalog`'s result."""
    cached = _cache_get(cache, "disc_catalog", catalog_id, CACHE_TTL_DAYS["disc_catalog"], refresh)
    if cached is not None:
        return [DiscCatalogEntry(**row) for row in cached]
    entries = _query_vizier_disc_catalog(catalog_id, row_limit)
    _cache_set(cache, "disc_catalog", catalog_id, [vars(e) for e in entries])
    _save_cache(cache_path, cache)
    return entries


# -- MAST TIC cross-match --------------------------------------------------------

_TIC_ID_COLS = ("ID", "TIC", "TICID")
_TIC_RA_COLS = ("ra", "RA", "RAJ2000")
_TIC_DEC_COLS = ("dec", "DEC", "DEJ2000")
_TIC_TMAG_COLS = ("Tmag", "TESSmag")
_TIC_TEFF_COLS = ("Teff", "teff")
_TIC_PLX_COLS = ("plx", "Plx", "parallax")
_TIC_KIC_COLS = ("KIC", "kic")
_TIC_SEP_COLS = ("dstArcSec", "distance", "dist")


def _parse_tic_table(table: Any) -> list[TicMatch]:
    """Pure parsing of a TIC query result (any Vizier/MAST cross-match) into rows."""
    colnames = list(table.colnames)
    id_col = _first_present(colnames, _TIC_ID_COLS)
    ra_col = _first_present(colnames, _TIC_RA_COLS)
    dec_col = _first_present(colnames, _TIC_DEC_COLS)
    if id_col is None or ra_col is None or dec_col is None:
        raise ValueError(f"TIC result missing id/ra/dec columns; saw {colnames}")
    tmag_col = _first_present(colnames, _TIC_TMAG_COLS)
    teff_col = _first_present(colnames, _TIC_TEFF_COLS)
    plx_col = _first_present(colnames, _TIC_PLX_COLS)
    kic_col = _first_present(colnames, _TIC_KIC_COLS)
    sep_col = _first_present(colnames, _TIC_SEP_COLS)

    matches: list[TicMatch] = []
    for row in table:
        tic_id = _safe_int(row[id_col])
        ra = _safe_float(row[ra_col])
        dec = _safe_float(row[dec_col])
        if tic_id is None or ra is None or dec is None:
            continue
        matches.append(
            TicMatch(
                tic_id=tic_id,
                ra_deg=ra,
                dec_deg=dec,
                tmag=_safe_float(row[tmag_col]) if tmag_col else None,
                teff=_safe_float(row[teff_col]) if teff_col else None,
                plx_mas=_safe_float(row[plx_col]) if plx_col else None,
                kic_id=_safe_int(row[kic_col]) if kic_col else None,
                separation_arcsec=_safe_float(row[sep_col]) if sep_col else None,
            )
        )
    return matches


def _query_tic_region(ra_deg: float, dec_deg: float, radius_deg: float) -> list[TicMatch]:
    """Network call: cone-search the TESS Input Catalog around a sky position."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs

    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    table = Catalogs.query_region(coord, radius=radius_deg * u.deg, catalog="TIC")
    return _parse_tic_table(table)


def fetch_tic_region(
    cache: dict[str, Any],
    cache_path: Path,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    section: str,
    refresh: bool,
) -> list[TicMatch]:
    """Cache and return :func:`_query_tic_region`'s result.

    ``section`` selects the cache bucket/TTL (``"tic_crossmatch"`` for a
    tight single-star match, ``"control_candidates"`` for the wider cone
    search) -- same underlying call, different cached call-sites so a
    tight-radius and a wide-radius query for the same star don't collide.
    """
    key = f"{ra_deg:.5f},{dec_deg:.5f},{radius_deg:.4f}"
    cached = _cache_get(cache, section, key, CACHE_TTL_DAYS[section], refresh)
    if cached is not None:
        return [TicMatch(**row) for row in cached]
    matches = _query_tic_region(ra_deg, dec_deg, radius_deg)
    _cache_set(cache, section, key, [vars(m) for m in matches])
    _save_cache(cache_path, cache)
    return matches


def best_tic_match(matches: Sequence[TicMatch]) -> TicMatch | None:
    """Pure selection of the nearest cross-match from a cone-search result."""
    if not matches:
        return None

    def _sep(m: TicMatch) -> float:
        return m.separation_arcsec if m.separation_arcsec is not None else 0.0

    return min(matches, key=_sep)


# -- Photometry availability -----------------------------------------------------


def _query_photometry_available(target_id: str, mission: str) -> int:
    """Network call: how many light-curve products the archive has for this id.

    Reuses ``exocomet.io.download``'s bounded search machinery (its module
    docstring explains why an unbounded MAST call is dangerous: a real hang
    was found and fixed there 2026-09-14) rather than duplicating a second
    ad hoc timeout/retry implementation for what is otherwise the same call.
    Only searches -- never downloads -- so this is cheap even at scale.
    """
    import lightkurve as lk

    from exocomet.io import download as _download

    _download._configure_astroquery()
    search_kwargs: dict[str, object] = {"mission": mission}
    if mission == "TESS":
        search_kwargs["author"] = "SPOC"
        search_kwargs["exptime"] = 120
    else:
        search_kwargs["author"] = "Kepler"
        search_kwargs["cadence"] = "long"

    search = _download._call_with_retries(
        lambda: lk.search_lightcurve(target_id, **search_kwargs),
        timeout=_download.DEFAULT_SEARCH_TIMEOUT,
        description=f"search_lightcurve({target_id!r})",
        max_attempts=_download.DEFAULT_MAX_ATTEMPTS,
        backoff=_download.DEFAULT_RETRY_BACKOFF,
    )
    return len(search)


def confirm_photometry(
    cache: dict[str, Any],
    cache_path: Path,
    target_id: str,
    mission: str,
    refresh: bool,
) -> bool:
    """Cache and return whether ``target_id`` has any downloadable photometry."""
    key = f"{target_id}|{mission}"
    cached = _cache_get(cache, "photometry", key, CACHE_TTL_DAYS["photometry"], refresh)
    if cached is not None:
        return bool(cached["n_products"] > 0)
    try:
        n = _query_photometry_available(target_id, mission)
    except Exception as exc:  # pragma: no cover - network/archive dependent
        logger.warning("photometry check failed for %s (%s): %s", target_id, mission, exc)
        n = 0
    _cache_set(cache, "photometry", key, {"n_products": n})
    _save_cache(cache_path, cache)
    return n > 0


# -- Control-group matching ------------------------------------------------------


def select_control_star(
    target: TicMatch,
    candidates: Sequence[TicMatch],
    excluded_tic_ids: AbstractSet[int] = frozenset(),
    mag_tolerance: float = 0.5,
    dist_tolerance_frac: float = 0.3,
) -> TicMatch | None:
    """Pick a magnitude- (and, if known, distance-) matched control star.

    See the module docstring's "Control-group matching rule" section for the
    full rationale. Pure function: takes a plain list of candidate
    :class:`TicMatch` rows (a cone search in production, a hand-built
    fixture in tests) and returns the best surviving match, or ``None`` if
    nothing qualifies. Never returns the target itself, anything named in
    ``excluded_tic_ids`` (disc-catalogue members already resolved this run),
    or a hold-out/known-bad KIC target -- this last check is why control
    selection is exercised by the same hold-out test as the main list.
    """
    best: TicMatch | None = None
    best_key: tuple[float, float] | None = None
    for cand in candidates:
        if cand.tic_id == target.tic_id:
            continue
        if cand.tic_id in excluded_tic_ids:
            continue
        if is_forbidden_kic(cand.kic_id):
            continue
        if target.tmag is None or cand.tmag is None:
            continue
        if abs(cand.tmag - target.tmag) > mag_tolerance:
            continue
        if target.distance_pc is not None and cand.distance_pc is not None:
            dist_diff = abs(cand.distance_pc - target.distance_pc)
            if dist_diff > dist_tolerance_frac * target.distance_pc:
                continue
        key = (abs(cand.tmag - target.tmag), cand.separation_arcsec or 0.0)
        if best_key is None or key < best_key:
            best_key = key
            best = cand
    return best


# -- Exclusion enforcement (the important part) ----------------------------------


def enforce_hold_out_exclusion(rows: Sequence[TargetRow]) -> list[TargetRow]:
    """Drop, log, and then *assert* that no hold-out/known-bad KIC survives.

    This is the single most important correctness check in this script (see
    the module docstring). It is deliberately redundant with the earlier,
    per-star ``is_forbidden_target``/``is_forbidden_kic`` checks made while
    resolving candidates -- this function is the final gate every row must
    pass immediately before being written or returned, so a bug anywhere
    upstream (a new call-site that forgets the check, a future refactor)
    fails loudly here instead of silently producing a compromised list.
    """
    kept = [row for row in rows if not is_forbidden_kic(row.target_id)]
    n_dropped = len(rows) - len(kept)
    if n_dropped:
        logger.warning(
            "enforce_hold_out_exclusion: dropped %d row(s) that were hold-out/known-bad", n_dropped
        )
    assert not any(is_forbidden_kic(row.target_id) for row in kept), (
        "a hold-out validation target or known-bad host leaked into the generated "
        "target list -- this must never happen (see EXCLUDED_HOLD_OUT_KIC_NUMBERS "
        "/ KNOWN_BAD_KIC_NUMBERS)"
    )
    return kept


# -- Orchestration -----------------------------------------------------------------


def _slugify(name: str) -> str:
    """Turn a host name into a short, filesystem/CSV-safe pair id."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name.strip()).strip("-").lower()
    return slug or "target"


def build_target_list(
    max_targets: int,
    vizier_catalog: str = DEFAULT_VIZIER_CATALOG,
    cache_path: Path = DEFAULT_CACHE_PATH,
    refresh_cache: bool = False,
    max_candidates: int | None = None,
    tic_match_radius_arcsec: float = 5.0,
    control_search_radius_deg: float = 1.0,
    mag_tolerance: float = 0.5,
    dist_tolerance_frac: float = 0.3,
    row_limit: int = -1,
) -> list[TargetRow]:
    """Run the full pipeline and return the accepted, exclusion-checked rows.

    Parameters
    ----------
    max_targets
        Stop once this many disc stars have been *accepted* (resolved to a
        real, downloadable target and not a hold-out/known-bad star). Each
        accepted disc star may also contribute one control row per mission.
    max_candidates
        Upper bound on how many disc-catalogue rows are even considered
        (cross-matched at MAST) before giving up, regardless of how many
        were accepted -- bounds worst-case network calls when the catalogue
        has many entries with no usable photometry. Defaults to
        ``max(5 * max_targets, 20)``.
    row_limit
        Row cap passed to the VizieR query itself (``-1`` = no limit).

    Notes
    -----
    Known limitation of the control-group exclusion: a control star is only
    checked against disc-catalogue stars *already resolved earlier in this
    run*, not the full catalogue -- resolving every catalogue row up front
    just to build a complete exclusion set would multiply the number of MAST
    calls for no real benefit at this list size. Documented here rather than
    silently assumed.
    """
    cache = _load_cache(cache_path)
    disc_entries = fetch_disc_catalog(cache, cache_path, vizier_catalog, row_limit, refresh_cache)
    logger.info(
        "%s: %d debris-disc catalogue rows with usable positions", vizier_catalog, len(disc_entries)
    )

    if max_candidates is None:
        max_candidates = max(5 * max_targets, 20)

    rows: list[TargetRow] = []
    known_disc_tic_ids: set[int] = set()
    n_accepted = 0
    n_considered = 0

    for entry in disc_entries:
        if n_accepted >= max_targets or n_considered >= max_candidates:
            break
        n_considered += 1

        tic_matches = fetch_tic_region(
            cache, cache_path, entry.ra_deg, entry.dec_deg,
            tic_match_radius_arcsec / 3600.0, "tic_crossmatch", refresh_cache,
        )
        tic = best_tic_match(tic_matches)
        if tic is None:
            logger.info(
                "%s: no TIC cross-match within %.1f arcsec, skipping",
                entry.name, tic_match_radius_arcsec,
            )
            continue

        resolved = ResolvedTarget(disc_entry=entry, tic=tic)
        if is_forbidden_target(resolved):
            logger.warning(
                "%s: resolves to hold-out/known-bad KIC %s -- excluded, not just skipped",
                entry.name, tic.kic_id,
            )
            continue

        kepler_ok = tic.kic_id is not None and confirm_photometry(
            cache, cache_path, f"KIC {tic.kic_id}", "Kepler", refresh_cache
        )
        tess_ok = confirm_photometry(cache, cache_path, f"TIC {tic.tic_id}", "TESS", refresh_cache)
        if not kepler_ok and not tess_ok:
            logger.info(
                "%s (TIC %d): no downloadable Kepler or TESS photometry, skipping",
                entry.name, tic.tic_id,
            )
            continue

        known_disc_tic_ids.add(tic.tic_id)
        pair_id = _slugify(entry.name)

        if kepler_ok and tic.kic_id is not None:
            rows.append(
                TargetRow(
                    target_id=f"KIC {tic.kic_id}",
                    mission="Kepler",
                    role="disc",
                    pair_id=pair_id,
                    host_name=entry.name,
                    catalog_ref=vizier_catalog,
                    tmag=tic.tmag,
                    teff=tic.teff,
                    distance_pc=tic.distance_pc,
                )
            )
        if tess_ok:
            rows.append(
                TargetRow(
                    target_id=f"TIC {tic.tic_id}",
                    mission="TESS",
                    role="disc",
                    pair_id=pair_id,
                    host_name=entry.name,
                    catalog_ref=vizier_catalog,
                    tmag=tic.tmag,
                    teff=tic.teff,
                    distance_pc=tic.distance_pc,
                )
            )

        control_candidates = fetch_tic_region(
            cache, cache_path, entry.ra_deg, entry.dec_deg,
            control_search_radius_deg, "control_candidates", refresh_cache,
        )
        control = select_control_star(
            tic, control_candidates, known_disc_tic_ids, mag_tolerance, dist_tolerance_frac
        )
        if control is not None:
            control_kepler_ok = control.kic_id is not None and confirm_photometry(
                cache, cache_path, f"KIC {control.kic_id}", "Kepler", refresh_cache
            )
            control_tess_ok = confirm_photometry(
                cache, cache_path, f"TIC {control.tic_id}", "TESS", refresh_cache
            )
            if control_kepler_ok and control.kic_id is not None:
                rows.append(
                    TargetRow(
                        target_id=f"KIC {control.kic_id}",
                        mission="Kepler",
                        role="control",
                        pair_id=pair_id,
                        host_name=entry.name,
                        catalog_ref=vizier_catalog,
                        tmag=control.tmag,
                        teff=control.teff,
                        distance_pc=control.distance_pc,
                        separation_arcsec=control.separation_arcsec,
                    )
                )
            if control_tess_ok:
                rows.append(
                    TargetRow(
                        target_id=f"TIC {control.tic_id}",
                        mission="TESS",
                        role="control",
                        pair_id=pair_id,
                        host_name=entry.name,
                        catalog_ref=vizier_catalog,
                        tmag=control.tmag,
                        teff=control.teff,
                        distance_pc=control.distance_pc,
                        separation_arcsec=control.separation_arcsec,
                    )
                )
        else:
            logger.info("%s (TIC %d): no matched control star found", entry.name, tic.tic_id)

        n_accepted += 1

    logger.info(
        "considered %d/%d catalogue rows, accepted %d disc host(s), %d total rows before exclusion",
        n_considered, len(disc_entries), n_accepted, len(rows),
    )
    return enforce_hold_out_exclusion(rows)


# -- Output --------------------------------------------------------------------


def _format_line(row: TargetRow) -> str:
    """One output-file line: id first (what the loader parses), then a comment."""
    return (
        f"{row.target_id}  # role={row.role} pair={row.pair_id} "
        f"host={row.host_name} catalog={row.catalog_ref}"
    )


_HEADER = """\
# Generated by scripts/build_target_list.py -- do not hand-edit, re-run the
# script instead (it will re-derive this file from cache or a fresh query).
#
# One target per line. Blank lines and '#' comments ignored (same convention
# as config/watchlist.txt). Each line's trailing comment records: which
# debris-disc host this target is paired with ('pair'), whether this row IS
# the disc host or its matched control ('role'), the host's common name, and
# the source catalogue. Full numeric metadata (magnitude, temperature,
# distance) for every row is in config/target_list_metadata.csv -- open that
# file to see the actual numbers behind each disc/control pairing.
#
# Generated: {timestamp}
"""


def write_target_lists(rows: Sequence[TargetRow], out_dir: Path, dry_run: bool = False) -> None:
    """Write the two mission target lists plus the metadata CSV.

    Re-runs :func:`enforce_hold_out_exclusion` one more time immediately
    before writing -- the last possible point of failure, and cheap enough
    that there is no reason not to.
    """
    rows = enforce_hold_out_exclusion(rows)
    timestamp = datetime.now(UTC).isoformat()

    by_mission: dict[str, list[TargetRow]] = {"Kepler": [], "TESS": []}
    for row in rows:
        by_mission.setdefault(row.mission, []).append(row)

    file_map = {
        "Kepler": out_dir / "target_list_kepler.txt",
        "TESS": out_dir / "target_list_tess.txt",
    }

    if dry_run:
        for mission, path in file_map.items():
            mission_rows = by_mission.get(mission, [])
            print(f"--- would write {path} ({len(mission_rows)} targets) ---")
            for row in mission_rows:
                print(_format_line(row))
        print(f"--- would write {out_dir / 'target_list_metadata.csv'} ({len(rows)} rows) ---")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for mission, path in file_map.items():
        mission_rows = by_mission.get(mission, [])
        lines = [_HEADER.format(timestamp=timestamp)]
        lines.extend(_format_line(row) for row in mission_rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("wrote %d targets to %s", len(mission_rows), path)

    import csv

    meta_path = out_dir / "target_list_metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["target_id", "mission", "role", "pair_id", "host_name", "catalog_ref",
             "tmag", "teff", "distance_pc", "separation_arcsec"]
        )
        for row in rows:
            writer.writerow(
                [row.target_id, row.mission, row.role, row.pair_id, row.host_name, row.catalog_ref,
                 row.tmag, row.teff, row.distance_pc, row.separation_arcsec]
            )
    logger.info("wrote %d metadata rows to %s", len(rows), meta_path)


# -- CLI --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-targets", type=int, default=50,
        help="stop once this many disc host stars have been accepted",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, don't write files")
    parser.add_argument("--refresh-cache", action="store_true", help="ignore cached query results")
    parser.add_argument("--vizier-catalog", default=DEFAULT_VIZIER_CATALOG)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--tic-match-radius-arcsec", type=float, default=5.0)
    parser.add_argument("--control-search-radius-deg", type=float, default=1.0)
    parser.add_argument("--mag-tolerance", type=float, default=0.5)
    parser.add_argument("--dist-tolerance-frac", type=float, default=0.3)
    parser.add_argument("--row-limit", type=int, default=-1, help="VizieR row cap, -1 = unlimited")
    args = parser.parse_args()

    rows = build_target_list(
        max_targets=args.max_targets,
        vizier_catalog=args.vizier_catalog,
        cache_path=args.cache_path,
        refresh_cache=args.refresh_cache,
        max_candidates=args.max_candidates,
        tic_match_radius_arcsec=args.tic_match_radius_arcsec,
        control_search_radius_deg=args.control_search_radius_deg,
        mag_tolerance=args.mag_tolerance,
        dist_tolerance_frac=args.dist_tolerance_frac,
        row_limit=args.row_limit,
    )
    write_target_lists(rows, args.out_dir, dry_run=args.dry_run)

    n_disc = sum(1 for r in rows if r.role == "disc")
    n_control = sum(1 for r in rows if r.role == "control")
    print(f"{len(rows)} total rows ({n_disc} disc, {n_control} control) across Kepler + TESS")


if __name__ == "__main__":
    main()
