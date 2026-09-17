# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained HTML reports, built from a run's saved extracts.

Regenerates plots from the extract parquet already saved during the run
(``exocomet.io.extract.load_extract`` -> ``detrend_savgol`` ->
``AsymmetricDipDetector`` -- the same, unmodified pipeline steps, run again
locally with no network involved) rather than storing full plot objects in
SQLite. Images are embedded as base64 so the resulting HTML file is portable
-- open it anywhere, nothing else to ship alongside it.

Every report generated for an injected run carries a loud, unmissable
"INJECTED DATA" banner -- see ``build_run_report``/``build_target_report``'s
``injected`` parameter, threaded in from the run's own stored flag
(``storage.get_run(run_id)["injected"]``), never left to the caller to
remember.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path
from typing import Any

from dashboard import DASHBOARD_VAR_DIR
from dashboard import storage
from exocomet.core.config import PipelineConfig
from exocomet.detect.comet_detector import AsymmetricDipDetector
from exocomet.detrend.detrend import detrend_savgol
from exocomet.io import extract as extract_mod
from exocomet.viz import plots as viz_plots

REPORTS_DIR = DASHBOARD_VAR_DIR / "reports"


def _run_extract_dir(run_id: str) -> Path:
    return DASHBOARD_VAR_DIR / "runs" / run_id / "extract"


def _fig_to_base64(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _regenerate(run_id: str, target_id: str, mission: str, config: PipelineConfig) -> tuple[Any, list[Any]]:
    """Reload the saved extract and re-run detect (no network) for real plots."""
    ex_path = extract_mod.extract_path(_run_extract_dir(run_id), mission, target_id)
    lc = extract_mod.load_extract(ex_path)
    detrended = detrend_savgol(lc, config.detrend)
    records = AsymmetricDipDetector(config).run(detrended)
    return lc, records


_HTML_SHELL = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ background:#0b0e17; color:#e6e8ef; font-family: -apple-system, system-ui, sans-serif; padding: 2rem; }}
  h1, h2 {{ color:#9fd2ff; }}
  .banner {{ background:#5a1f1f; color:#ffb3b3; border:2px solid #ff4d4d; padding:0.75rem 1rem;
             border-radius:6px; font-weight:bold; margin-bottom:1.5rem; letter-spacing:0.03em; }}
  table {{ border-collapse: collapse; width:100%; margin-bottom:1.5rem; }}
  th, td {{ border:1px solid #2a2f3d; padding:0.4rem 0.6rem; text-align:left; font-size:0.9rem; }}
  th {{ background:#161b28; }}
  .flagged {{ color:#7CFC9B; font-weight:bold; }}
  img {{ max-width:100%; border-radius:6px; margin:0.5rem 0 1.5rem 0; border:1px solid #2a2f3d; }}
  .meta {{ color:#8892a6; font-size:0.85rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def build_run_report(run_id: str) -> Path:
    """End-of-run summary report: every target, every candidate, key charts."""
    run = storage.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run {run_id!r}")
    injected = bool(run["injected"])
    config = PipelineConfig.from_dict(json.loads(run["config_json"]))
    targets = storage.get_run_targets(run_id)
    candidates = storage.get_candidates(run_id)

    parts = [f"<h1>Exocomet Hunter -- Run Report</h1>"]
    if injected:
        parts.append('<div class="banner">&#9888; INJECTED DATA -- every result below comes from a synthetic transit injected into real photometry. Not a real detection.</div>')
    parts.append(
        f'<p class="meta">Run <code>{run_id}</code> &middot; mode {run["mode"]} &middot; '
        f'mission {run["mission"]} &middot; started {run["started_at"]} &middot; status {run["status"]}</p>'
    )

    parts.append("<h2>Targets processed</h2><table><tr><th>Target</th><th>Status</th>"
                  "<th>Candidates</th><th>Flagged</th><th>Error</th></tr>")
    for t in targets:
        parts.append(
            f'<tr><td>{t["target_id"]}</td><td>{t["status"]}</td>'
            f'<td>{t["n_candidates"] if t["n_candidates"] is not None else "-"}</td>'
            f'<td class="flagged">{t["n_flagged"] if t["n_flagged"] is not None else "-"}</td>'
            f'<td>{t["error"] or ""}</td></tr>'
        )
    parts.append("</table>")

    flagged = [c for c in candidates if c["flagged"]]
    parts.append(f"<h2>Flagged candidates ({len(flagged)})</h2>")
    if flagged:
        parts.append("<table><tr><th>Target</th><th>t_min</th><th>Depth (ppm)</th>"
                      "<th>&Delta;BIC</th><th>&tau;/&sigma;</th></tr>")
        for c in flagged:
            parts.append(
                f'<tr><td>{c["target_id"]}</td><td>{c["t_min"]:.3f}</td>'
                f'<td>{c["depth_ppm"]:.1f}</td><td>{c["delta_bic"]:.1f}</td>'
                f'<td>{c["tau_over_sigma"]:.2f}</td></tr>'
            )
        parts.append("</table>")
    else:
        parts.append("<p>No events crossed the flagging thresholds in this run.</p>")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    title = f"INJECTED -- Run {run_id}" if injected else f"Run {run_id}"
    filename = f"INJECTED_{run_id}.html" if injected else f"{run_id}.html"
    out_path = REPORTS_DIR / filename
    out_path.write_text(_HTML_SHELL.format(title=title, body="\n".join(parts)), encoding="utf-8")
    return out_path


def build_target_report(run_id: str, target_id: str) -> Path:
    """On-demand per-star report with the real light-curve and event plots."""
    run = storage.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run {run_id!r}")
    injected = bool(run["injected"])
    config = PipelineConfig.from_dict(json.loads(run["config_json"]))

    lc, records = _regenerate(run_id, target_id, run["mission"], config)

    tmp_dir = REPORTS_DIR / "_tmp"
    lc_plot_path = viz_plots.plot_light_curve(
        lc, events=[r.event for r in records], path=tmp_dir / f"{run_id}_{target_id}_lc.png",
        title=f"{target_id} ({'INJECTED' if injected else 'real'})",
    )

    parts = [f"<h1>{target_id} -- detail</h1>"]
    if injected:
        parts.append('<div class="banner">&#9888; INJECTED DATA -- this star\'s results come from a synthetic transit injected into real photometry. Not a real detection.</div>')

    lc_b64 = _fig_to_base64(lc_plot_path)
    if lc_b64:
        parts.append(f'<img src="data:image/png;base64,{lc_b64}" alt="light curve">')

    parts.append(f"<h2>Detected events ({len(records)})</h2>")
    for i, record in enumerate(records):
        event_path = viz_plots.plot_event(
            lc, record, path=tmp_dir / f"{run_id}_{target_id}_event{i}.png"
        )
        b64 = _fig_to_base64(event_path)
        flagged_tag = ' <span class="flagged">FLAGGED</span>' if record.score.flagged else ""
        parts.append(
            f"<h3>Event {i + 1}{flagged_tag}</h3>"
            f'<p class="meta">depth {record.event.depth_ppm:.1f} ppm &middot; '
            f"&Delta;BIC {record.score.delta_bic:.1f} &middot; "
            f"&tau;/&sigma; {record.score.tau_over_sigma:.2f}</p>"
        )
        if b64:
            parts.append(f'<img src="data:image/png;base64,{b64}" alt="event {i + 1}">')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_target = extract_mod.sanitize_target_id(target_id)
    title = f"INJECTED -- {target_id} -- {run_id}" if injected else f"{target_id} -- {run_id}"
    filename = f"INJECTED_{run_id}_{safe_target}.html" if injected else f"{run_id}_{safe_target}.html"
    out_path = REPORTS_DIR / filename
    out_path.write_text(
        _HTML_SHELL.format(title=title, body="\n".join(parts)), encoding="utf-8"
    )
    return out_path


def _injected_label(injected: bool) -> str:
    """Redundant, text-form INJECTED marker.

    Not just a boolean column: a boolean is silent once anyone filters, greps,
    pastes a screenshot, or copies three rows into a chat message without the
    header. This is deliberately loud and repeated -- the word itself, not a
    flag someone has to know to look for -- because a shared fragment of this
    data must announce what it is even stripped of all other context.
    """
    return "INJECTED-SYNTHETIC-NOT-REAL" if injected else "real"


def export_csv(run_id: str) -> str:
    run = storage.get_run(run_id)
    injected = bool(run["injected"]) if run else False
    rows = storage.get_candidates(run_id)
    buf = io.StringIO()
    if injected:
        buf.write(
            "# WARNING -- INJECTED SYNTHETIC DATA -- every row below comes from a "
            "synthetic transit injected into real photometry for calibration/demo "
            "purposes. NOT a real detection. INJECTED. INJECTED. INJECTED.\n"
        )
    if rows:
        fieldnames = [*rows[0].keys(), "data_source"]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "target_id": f"{row['target_id']} [{_injected_label(injected)}]",
                              "data_source": _injected_label(injected)})
    return buf.getvalue()


def export_json(run_id: str) -> str:
    run = storage.get_run(run_id)
    injected = bool(run["injected"]) if run else False
    candidates = storage.get_candidates(run_id)
    for row in candidates:
        row["data_source"] = _injected_label(injected)
        row["target_id"] = f"{row['target_id']} [{_injected_label(injected)}]"
    payload: dict[str, Any] = {
        "run": run,
        "targets": storage.get_run_targets(run_id),
        "candidates": candidates,
    }
    if injected:
        payload = {
            "WARNING": "INJECTED SYNTHETIC DATA -- every candidate below comes from a "
            "synthetic transit injected into real photometry. NOT a real detection. "
            "INJECTED. INJECTED. INJECTED.",
            **payload,
        }
    return json.dumps(payload, indent=2, default=str)
