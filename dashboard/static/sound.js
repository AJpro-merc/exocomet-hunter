// Copyright (c) 2026 Atharva Joshi
// SPDX-License-Identifier: BSD-3-Clause

// Confirmation tones + optional ambient drone, synthesized with the Web
// Audio API (no external audio files). Both toggleable, both persisted
// per-browser via localStorage. Exposes window.ExoSound for app.js.

(function () {
  "use strict";

  const LS_FX_MUTED = "exocomet_sound_muted";
  const LS_AMBIENT_ON = "exocomet_ambient_on"; // ambient defaults OFF

  let ctx = null;
  function ensureCtx() {
    if (!ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      ctx = new Ctx();
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  function isFxMuted() {
    try {
      return localStorage.getItem(LS_FX_MUTED) === "1";
    } catch (err) {
      return false;
    }
  }

  function isAmbientOn() {
    try {
      return localStorage.getItem(LS_AMBIENT_ON) === "1";
    } catch (err) {
      return false;
    }
  }

  function tone(freq, duration, type, gainPeak) {
    if (isFxMuted()) return;
    const c = ensureCtx();
    if (!c) return;
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = type || "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0, c.currentTime);
    gain.gain.linearRampToValueAtTime(gainPeak || 0.07, c.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, c.currentTime + duration);
    osc.connect(gain).connect(c.destination);
    osc.start();
    osc.stop(c.currentTime + duration + 0.05);
  }

  let ambientNodes = null;
  function startAmbient() {
    if (ambientNodes) return;
    const c = ensureCtx();
    if (!c) return;
    const osc1 = c.createOscillator();
    const osc2 = c.createOscillator();
    const gain = c.createGain();
    osc1.type = "sine";
    osc1.frequency.value = 55;
    osc2.type = "sine";
    osc2.frequency.value = 82.5;
    gain.gain.value = 0;
    gain.gain.linearRampToValueAtTime(0.018, c.currentTime + 2);
    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(c.destination);
    osc1.start();
    osc2.start();
    ambientNodes = { osc1, osc2, gain };
  }

  function stopAmbient() {
    if (!ambientNodes || !ctx) return;
    const { osc1, osc2, gain } = ambientNodes;
    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 1);
    setTimeout(() => {
      osc1.stop();
      osc2.stop();
    }, 1100);
    ambientNodes = null;
  }

  window.ExoSound = {
    runStart() {
      tone(440, 0.16, "sine", 0.06);
      setTimeout(() => tone(660, 0.14, "sine", 0.05), 90);
    },
    runComplete() {
      tone(523, 0.14, "sine", 0.06);
      setTimeout(() => tone(659, 0.14, "sine", 0.06), 100);
      setTimeout(() => tone(880, 0.22, "sine", 0.07), 200);
    },
    candidateFlagged() {
      tone(740, 0.1, "square", 0.035);
    },
    isFxMuted,
    isAmbientOn,
    toggleFx() {
      const next = !isFxMuted();
      try {
        localStorage.setItem(LS_FX_MUTED, next ? "1" : "0");
      } catch (err) {
        /* storage blocked -- toggle still works for this tab */
      }
      return next;
    },
    toggleAmbient() {
      const next = !isAmbientOn();
      try {
        localStorage.setItem(LS_AMBIENT_ON, next ? "1" : "0");
      } catch (err) {
        /* storage blocked */
      }
      if (next) startAmbient();
      else stopAmbient();
      return next;
    },
    initAmbientIfEnabled() {
      if (isAmbientOn()) startAmbient();
    },
  };

  // Browsers require a user gesture before audio can play; catch the first
  // click anywhere and use it to resume the context / start ambient if it's
  // enabled but couldn't start yet (e.g. it was on before this page loaded).
  document.addEventListener(
    "click",
    () => {
      ensureCtx();
      if (isAmbientOn()) startAmbient();
    },
    { once: true }
  );
})();
