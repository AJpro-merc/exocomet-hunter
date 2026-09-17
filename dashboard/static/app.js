// Copyright (c) 2026 Atharva Joshi
// SPDX-License-Identifier: BSD-3-Clause

function degToHMS(deg) {
  const totalHours = ((deg % 360) + 360) % 360 / 15;
  const h = Math.floor(totalHours);
  const mFull = (totalHours - h) * 60;
  const m = Math.floor(mFull);
  const s = ((mFull - m) * 60).toFixed(1);
  return `${String(h).padStart(2, "0")}h${String(m).padStart(2, "0")}m${s.padStart(4, "0")}s`;
}

function degToDMS(deg) {
  const sign = deg < 0 ? "−" : "+";
  const abs = Math.abs(deg);
  const d = Math.floor(abs);
  const mFull = (abs - d) * 60;
  const m = Math.floor(mFull);
  const s = Math.round((mFull - m) * 60);
  return `${sign}${String(d).padStart(2, "0")}°${String(m).padStart(2, "0")}′${String(s).padStart(2, "0")}″`;
}

function dashboardApp() {
  return {
    tab: "run",
    mode: "normal",
    mission: "Kepler",
    targetSource: "list",
    targetLists: {},
    targetFilter: "",
    selectedListTargets: [],
    manualTargetsText: "",
    maxTargets: 5,
    injected: false,
    injectionDepth: 0.001,
    injectionIngress: 0.3,
    injectionEgress: 1.0,
    startingRun: false,

    activeRuns: {},
    eventSources: {},

    history: [],
    selectedRun: null,

    configForm: null,
    advancedOpen: false,
    pendingInlineConfig: null,

    now: new Date(),
    timezone: 0, // UTC offset in whole hours
    timezoneList: Array.from({ length: 27 }, (_, i) => i - 12), // -12..+14

    sessionStats: { searched: 0, candidates: 0, flagged: 0 },
    tweenedStats: { searched: 0, candidates: 0, flagged: 0 },
    missionLog: [],

    fxMuted: window.ExoSound ? window.ExoSound.isFxMuted() : false,
    ambientOn: window.ExoSound ? window.ExoSound.isAmbientOn() : false,

    toggleFx() {
      this.fxMuted = window.ExoSound.toggleFx();
    },
    toggleAmbient() {
      this.ambientOn = window.ExoSound.toggleAmbient();
    },

    forceStopConfirm: null, // runId pending a force-stop confirmation, or null

    // Small, deliberately not a full curation system (see VISUAL_REDESIGN.md) --
    // just a short list of targets with a known caveat found during testing.
    targetCaveats: {
      "KIC 1161345": "unusually high flag rate in earlier testing — unverified host",
    },

    async init() {
      await this.fetchTargetLists();
      await this.refreshHistory();
      this.$watch("mode", async (val) => {
        if (val === "research" && !this.configForm) {
          await this.fetchConfig();
        }
      });

      setInterval(() => {
        this.now = new Date();
      }, 1000);
    },

    tzLabel(offset) {
      if (offset === 0) return "UTC";
      return offset > 0 ? `UTC+${offset}` : `UTC${offset}`;
    },

    clockDisplay() {
      const shifted = new Date(this.now.getTime() + this.timezone * 3600000);
      const hh = String(shifted.getUTCHours()).padStart(2, "0");
      const mm = String(shifted.getUTCMinutes()).padStart(2, "0");
      const ss = String(shifted.getUTCSeconds()).padStart(2, "0");
      return `${hh}:${mm}:${ss}`;
    },

    elapsedDisplay() {
      const runs = Object.values(this.activeRuns);
      if (runs.length === 0) return "T+ --:--:--";
      const startedAt = Math.min(...runs.map((r) => r.startedAt));
      const totalSec = Math.max(0, Math.floor((this.now.getTime() - startedAt) / 1000));
      const hh = String(Math.floor(totalSec / 3600)).padStart(2, "0");
      const mm = String(Math.floor((totalSec % 3600) / 60)).padStart(2, "0");
      const ss = String(totalSec % 60).padStart(2, "0");
      return `T+ ${hh}:${mm}:${ss}`;
    },

    systemStatus() {
      const runs = Object.values(this.activeRuns);
      return runs.some((r) => r.streamOk === false) ? "degraded" : "nominal";
    },

    tweenStat(key, target) {
      const start = this.tweenedStats[key];
      if (start === target) return;
      const startTime = performance.now();
      const duration = 500;
      const step = (t) => {
        const p = Math.min(1, (t - startTime) / duration);
        this.tweenedStats[key] = Math.round(start + (target - start) * p);
        if (p < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    },

    pushMissionLog(text) {
      this.missionLog.push(text);
      if (this.missionLog.length > 40) this.missionLog.shift();
    },

    statusLightClass(status) {
      if (status === "running") return "go-light amber";
      if (status === "completed") return "go-light green";
      return "go-light dim"; // stopped / force_stopped
    },

    async fetchTargetLists() {
      const res = await fetch("/api/target-lists");
      this.targetLists = await res.json();
    },

    async fetchConfig() {
      const res = await fetch("/api/config");
      this.configForm = await res.json();
    },

    async saveConfig() {
      const sections = {};
      for (const section of Object.keys(this.configForm.values)) {
        sections[section] = this.configForm.values[section];
      }
      const res = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sections }),
      });
      this.configForm = await res.json();
    },

    async resetConfig() {
      const res = await fetch("/api/config/reset", { method: "POST" });
      this.configForm = await res.json();
    },

    useConfigForNextRunOnly() {
      this.pendingInlineConfig = JSON.parse(JSON.stringify(this.configForm.values));
      this.tab = "run";
    },

    filteredTargets() {
      const list = this.targetLists[this.mission] || [];
      if (!this.targetFilter.trim()) return list;
      const q = this.targetFilter.trim().toLowerCase();
      return list.filter((t) => t.toLowerCase().includes(q));
    },

    caveatFor(targetId) {
      return this.targetCaveats[targetId] || null;
    },

    selectAllTargets() {
      this.selectedListTargets = [...this.filteredTargets()];
    },

    currentTargets() {
      if (this.targetSource === "manual") {
        return this.manualTargetsText
          .split("\n")
          .map((s) => s.trim())
          .filter((s) => s.length > 0);
      }
      return this.selectedListTargets.length > 0 ? this.selectedListTargets : null;
    },

    async startRun() {
      this.startingRun = true;
      try {
        const body = {
          mode: this.mode,
          mission: this.mission,
          targets: this.currentTargets(),
          max_targets: this.maxTargets || null,
          injected: this.mode === "research" ? this.injected : false,
          injection_depth: this.injectionDepth,
          injection_ingress_days: this.injectionIngress,
          injection_egress_days: this.injectionEgress,
          inline_config: this.mode === "research" ? this.pendingInlineConfig : null,
        };
        const res = await fetch("/api/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const err = await res.json();
          alert("Could not start run: " + (err.detail || res.statusText));
          return;
        }
        const { run_id } = await res.json();
        this.pendingInlineConfig = null;
        this.trackRun(run_id, body.injected);
      } finally {
        this.startingRun = false;
      }
    },

    trackRun(runId, injected) {
      this.activeRuns[runId] = {
        status: "running",
        currentTarget: null,
        imageUrl: null,
        ra: null,
        dec: null,
        processed: 0,
        total: 0,
        injected: injected,
        stopping: false,
        log: [],
        startedAt: Date.now(),
        streamOk: true,
      };
      this.pushMissionLog(`[${runId}] run started`);
      if (window.ExoSound) window.ExoSound.runStart();

      const es = new EventSource("/api/stream/" + runId);
      this.eventSources[runId] = es;

      es.addEventListener("target_start", (ev) => {
        const data = JSON.parse(ev.data);
        const run = this.activeRuns[runId];
        if (!run) return;
        run.currentTarget = data.target_id;
        run.ra = null;
        run.dec = null;
        run.total = data.total;
        run.log.unshift({ text: `Searching ${data.target_id}...`, cls: "" });
        this.pushMissionLog(`[${runId}] searching ${data.target_id}`);
      });

      es.addEventListener("image", (ev) => {
        const data = JSON.parse(ev.data);
        const run = this.activeRuns[runId];
        if (!run) return;
        run.imageUrl = "/api/images/" + encodeURIComponent(data.target_id) + "?t=" + Date.now();
        if (typeof data.ra_deg === "number" && typeof data.dec_deg === "number") {
          run.ra = "RA " + degToHMS(data.ra_deg);
          run.dec = "DEC " + degToDMS(data.dec_deg);
        }
      });

      es.addEventListener("target_done", (ev) => {
        const data = JSON.parse(ev.data);
        const run = this.activeRuns[runId];
        if (!run) return;
        run.processed = data.index + 1;
        run.total = data.total;
        const tag = data.error ? `FAILED (${data.error})` : `${data.n_candidates} event(s), ${data.n_flagged} flagged`;
        const cls = data.error ? "failed" : data.n_flagged > 0 ? "flagged" : "success";
        run.log.unshift({ text: `${data.target_id}: ${tag}`, cls });

        this.sessionStats.searched += 1;
        this.sessionStats.candidates += data.n_candidates || 0;
        this.sessionStats.flagged += data.n_flagged || 0;
        this.tweenStat("searched", this.sessionStats.searched);
        this.tweenStat("candidates", this.sessionStats.candidates);
        this.tweenStat("flagged", this.sessionStats.flagged);

        this.pushMissionLog(`[${runId}] ${data.target_id}: ${tag}`);
        if (window.ExoSound && data.n_flagged > 0) window.ExoSound.candidateFlagged();
      });

      es.addEventListener("run_complete", (ev) => {
        const data = JSON.parse(ev.data);
        const run = this.activeRuns[runId];
        if (run) {
          run.status = data.status;
          run.log.unshift({ text: `Run ${data.status}.`, cls: "" });
        }
        this.pushMissionLog(`[${runId}] run ${data.status}`);
        if (window.ExoSound) window.ExoSound.runComplete();
        es.close();
        delete this.eventSources[runId];
        this.refreshHistory();
        setTimeout(() => {
          delete this.activeRuns[runId];
        }, 4000);
      });

      es.onerror = () => {
        const run = this.activeRuns[runId];
        if (run) {
          run.log.unshift("(stream disconnected)");
          run.streamOk = false;
        }
      };
    },

    async stopRun(runId, force) {
      const run = this.activeRuns[runId];
      if (run) run.stopping = true;
      await fetch(`/api/runs/${runId}/stop?force=${force}`, { method: "POST" });
    },

    requestForceStop(runId) {
      this.forceStopConfirm = runId;
    },

    cancelForceStop() {
      this.forceStopConfirm = null;
    },

    async confirmForceStop() {
      const runId = this.forceStopConfirm;
      this.forceStopConfirm = null;
      if (runId) await this.stopRun(runId, true);
    },

    async refreshHistory() {
      const res = await fetch("/api/runs");
      this.history = await res.json();
    },

    async openHistory(runId) {
      const res = await fetch("/api/runs/" + runId);
      this.selectedRun = await res.json();
    },
  };
}
