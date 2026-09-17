// Copyright (c) 2026 Atharva Joshi
// SPDX-License-Identifier: BSD-3-Clause

// Boot sequence: an emblem + a REAL system-health checklist (backed by
// GET /api/health) runs on every load. If anything is missing, a terminal
// panel lists it, streams a real repair via GET /api/health/repair (git
// restore for tracked files, local regeneration for generated state), then
// re-checks. On the very first visit only (localStorage-gated, unskippable),
// a longer cinematic tail plays after the checklist before the dashboard
// appears. Independent of Alpine/app.js -- this only touches #boot-* nodes.

(function () {
  "use strict";

  const overlay = document.getElementById("boot-overlay");
  if (!overlay) return;

  const emblemStage = document.getElementById("boot-emblem-stage");
  const checklistEl = document.getElementById("boot-checklist");
  const terminalEl = document.getElementById("boot-terminal");
  const terminalLog = document.getElementById("boot-terminal-log");
  const explainerEl = document.getElementById("boot-explainer");
  const montageEl = document.getElementById("boot-montage");
  const crosshairEl = document.getElementById("boot-crosshair");
  const targetAcquiredEl = document.getElementById("boot-target-acquired");
  const nominalEl = document.getElementById("boot-nominal");

  const LS_KEY = "exocomet_boot_seen";
  let firstVisit = true;
  try {
    firstVisit = !window.localStorage.getItem(LS_KEY);
  } catch (err) {
    firstVisit = true; // storage blocked (private mode etc.) -- just replay, harmless
  }

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function finishBoot() {
    try {
      window.localStorage.setItem(LS_KEY, "1");
    } catch (err) {
      /* storage blocked -- cinematic will just replay next time */
    }
    overlay.classList.add("boot-fading");
    setTimeout(() => overlay.classList.add("boot-hidden"), 450);
  }

  function revealChecklist(checks) {
    return new Promise((resolve) => {
      if (checks.length === 0) {
        resolve();
        return;
      }
      checks.forEach((c, i) => {
        setTimeout(() => {
          const row = document.createElement("div");
          row.className = "boot-line" + (c.ok ? "" : " fail");

          const label = document.createElement("span");
          label.className = "boot-line-label";
          label.textContent = c.label;

          const status = document.createElement("span");
          status.className = "boot-line-status";
          status.textContent = c.ok ? "OK" : "FAIL";

          row.appendChild(label);
          row.appendChild(status);
          checklistEl.appendChild(row);
          void row.offsetWidth; // force reflow so the transition plays
          row.classList.add("show");

          if (i === checks.length - 1) setTimeout(resolve, 260);
        }, i * 130);
      });
    });
  }

  function printTerminalLine(text, cls) {
    const row = document.createElement("div");
    row.className = "boot-term-line" + (cls ? " " + cls : "");
    row.textContent = text;
    terminalLog.appendChild(row);
    terminalLog.scrollTop = terminalLog.scrollHeight;
    return row;
  }

  function streamRepair() {
    return new Promise((resolve) => {
      const rows = {};
      let es;
      try {
        es = new EventSource("/api/health/repair");
      } catch (err) {
        resolve();
        return;
      }
      es.onmessage = (ev) => {
        let data;
        try {
          data = JSON.parse(ev.data);
        } catch (err) {
          return;
        }
        if (data.type === "repair_start") {
          rows[data.id] = printTerminalLine(`  ${data.path || data.id} ... RESTORING`, "dim");
        } else if (data.type === "repair_done") {
          const text = `  ${data.ok ? "RESTORED" : "FAILED"}: ${data.id}${data.detail ? " (" + data.detail + ")" : ""}`;
          const row = rows[data.id];
          if (row) {
            row.textContent = text;
            row.className = "boot-term-line " + (data.ok ? "ok" : "danger");
          } else {
            printTerminalLine(text, data.ok ? "ok" : "danger");
          }
        } else if (data.type === "repair_complete") {
          es.close();
          resolve();
        }
      };
      es.onerror = () => {
        es.close();
        resolve();
      };
    });
  }

  async function fetchHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) throw new Error("HTTP " + res.status);
      return await res.json();
    } catch (err) {
      return {
        all_ok: false,
        checks: [{ id: "api", label: "BACKEND API", ok: false, fix: null, detail: String(err) }],
      };
    }
  }

  async function runRepairFlow(checks) {
    const failing = checks.filter((c) => !c.ok);
    const fixable = failing.filter((c) => c.fix);
    const unfixable = failing.filter((c) => !c.fix);

    terminalEl.hidden = false;
    printTerminalLine(`MISSING COMPONENTS DETECTED (${failing.length})`, "warn");
    failing.forEach((c) => printTerminalLine(`  ${c.path || c.id}`, "dim"));

    if (fixable.length > 0) {
      printTerminalLine("AUTO-DOWNLOADING...", "accent");
      await streamRepair();
    }
    unfixable.forEach((c) => printTerminalLine(`  UNRESOLVED: ${c.label}`, "danger"));

    const recheck = await fetchHealth();
    printTerminalLine(recheck.all_ok ? "ALL SYSTEMS NOMINAL" : "SOME SYSTEMS DEGRADED — CONTINUING", recheck.all_ok ? "ok" : "warn");
    await sleep(600);
    terminalEl.hidden = true;
    return recheck.all_ok;
  }

  function showNominal(ok) {
    nominalEl.textContent = ok ? "ALL SYSTEMS NOMINAL" : "SYSTEMS NOMINAL — DEGRADED MODE";
    if (!ok) nominalEl.style.color = "var(--inject)";
    nominalEl.classList.add("show");
    return sleep(700);
  }

  async function playCinematicTail() {
    explainerEl.classList.add("active");
    const lines = explainerEl.querySelectorAll(".boot-explainer-line");
    for (const line of lines) {
      line.classList.add("show");
      await sleep(950);
    }
    await sleep(300);

    explainerEl.style.transition = "opacity 0.4s ease";
    explainerEl.style.opacity = "0";
    montageEl.classList.add("active");
    crosshairEl.classList.add("active");
    await sleep(400);

    const frames = montageEl.querySelectorAll(".boot-frame");
    const positions = ["4%", "34%", "63%", "88%"];
    for (let i = 0; i < frames.length; i++) {
      frames.forEach((f) => f.classList.remove("show"));
      frames[i].classList.add("show");
      crosshairEl.style.left = positions[i];
      await sleep(950);
    }

    crosshairEl.classList.add("locked");
    targetAcquiredEl.classList.add("show");
    await sleep(900);

    montageEl.style.transition = "opacity 0.4s ease";
    montageEl.style.opacity = "0";
    crosshairEl.style.opacity = "0";
    await sleep(400);
  }

  async function boot() {
    await sleep(2600); // emblem zoom-in hold

    emblemStage.classList.add("shrink");
    await sleep(500);

    const health = await fetchHealth();
    await revealChecklist(health.checks);

    let allOk = health.all_ok;
    if (!allOk) {
      allOk = await runRepairFlow(health.checks);
    }

    await showNominal(allOk);

    if (firstVisit) {
      await playCinematicTail();
    }

    finishBoot();
  }

  boot();
})();
