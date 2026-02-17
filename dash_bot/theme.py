# dash_bot/theme.py
from __future__ import annotations


def get_tron_css() -> str:
    return """
/* ═══════════════════════════════════════════════════════════════
   TRON LEGACY — NEON DASHBOARD THEME
   ═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');

:root {
  --tron-bg:         #060b14;
  --tron-surface:    #0a1628;
  --tron-surface2:   #0d1f3c;
  --tron-cyan:       #00f0ff;
  --tron-cyan-dim:   #007a82;
  --tron-cyan-glow:  0 0 8px rgba(0,240,255,0.4), 0 0 20px rgba(0,240,255,0.15);
  --tron-cyan-glow-strong: 0 0 10px rgba(0,240,255,0.6), 0 0 30px rgba(0,240,255,0.3), 0 0 60px rgba(0,240,255,0.1);
  --tron-magenta:    #ff2eed;
  --tron-magenta-dim:#8a1a80;
  --tron-orange:     #ff6f1a;
  --tron-green:      #00ff88;
  --tron-red:        #ff3355;
  --tron-yellow:     #ffe01a;
  --tron-text:       #c8dce8;
  --tron-text-dim:   #5a7a90;
  --tron-border:     rgba(0,240,255,0.12);
  --tron-border-active: rgba(0,240,255,0.5);
  --font-display:    'Orbitron', sans-serif;
  --font-mono:       'Share Tech Mono', monospace;
  --font-body:       'Rajdhani', sans-serif;
}

* { box-sizing: border-box; }

body {
  background: var(--tron-bg) !important;
  color: var(--tron-text) !important;
  font-family: var(--font-body) !important;
  font-weight: 400;
  overflow-x: hidden;
}

/* Background grid effect */
body::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    linear-gradient(rgba(0,240,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,240,255,0.015) 1px, transparent 1px);
  background-size: 60px 60px;
  pointer-events: none;
  z-index: 0;
}

/* Top scanline sweep animation */
body::after {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--tron-cyan), transparent);
  opacity: 0.3;
  animation: scanline 6s linear infinite;
  z-index: 9999;
  pointer-events: none;
}

@keyframes scanline {
  0%   { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

@keyframes pulse-glow {
  0%, 100% { opacity: 0.7; }
  50%      { opacity: 1; }
}

@keyframes border-pulse {
  0%, 100% { border-color: rgba(0,240,255,0.15); }
  50%      { border-color: rgba(0,240,255,0.4); }
}

/* ── HEADER ─────────────────────────────────────────────────── */
.tron-title {
  font-family: var(--font-display) !important;
  font-weight: 700;
  font-size: 1.6rem;
  letter-spacing: 4px;
  text-transform: uppercase;
  color: var(--tron-cyan) !important;
  text-shadow: var(--tron-cyan-glow-strong);
  text-align: center;
  margin: 18px 0 2px 0;
  padding: 0;
}
.tron-subtitle {
  text-align: center;
  color: var(--tron-text-dim) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem;
  letter-spacing: 1px;
  margin-bottom: 18px;
}

/* ── CARDS ──────────────────────────────────────────────────── */
.card, .card.bg-dark {
  background: var(--tron-surface) !important;
  border: 1px solid var(--tron-border) !important;
  border-radius: 4px !important;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
  border-color: var(--tron-border-active) !important;
  box-shadow: var(--tron-cyan-glow);
}
.card-header {
  background: linear-gradient(135deg, rgba(0,240,255,0.08), rgba(0,240,255,0.02)) !important;
  border-bottom: 1px solid var(--tron-border) !important;
  font-family: var(--font-display) !important;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--tron-cyan) !important;
  text-shadow: 0 0 6px rgba(0,240,255,0.3);
  padding: 12px 16px;
}
.card-body {
  background: transparent !important;
  padding: 14px 16px;
}

/* ── METRIC CARDS ──────────────────────────────────────────── */
.metric-card {
  background: linear-gradient(180deg, var(--tron-surface2), var(--tron-surface)) !important;
  border: 1px solid var(--tron-border) !important;
  border-radius: 4px !important;
  padding: 14px 12px 10px !important;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--tron-cyan), transparent);
  opacity: 0.6;
}
.metric-card:hover {
  border-color: var(--tron-border-active) !important;
  box-shadow: var(--tron-cyan-glow);
}
.metric-card:hover::before {
  opacity: 1;
}
.metric-label {
  font-family: var(--font-mono) !important;
  font-size: 0.65rem;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--tron-text-dim);
  margin-bottom: 4px;
}
.metric-value {
  font-family: var(--font-display) !important;
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--tron-cyan) !important;
  text-shadow: 0 0 12px rgba(0,240,255,0.4);
  margin: 0;
  line-height: 1.2;
}

/* ── SIDEBAR CONTROLS ──────────────────────────────────────── */
.sidebar-panel .card-body label,
.sidebar-panel label {
  font-family: var(--font-mono) !important;
  font-size: 0.7rem;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--tron-cyan-dim) !important;
  margin-top: 10px;
  margin-bottom: 3px;
  display: block;
}
.sidebar-panel hr {
  border-color: var(--tron-border) !important;
  margin: 12px 0;
  opacity: 0.5;
}
.sidebar-panel .form-control,
.sidebar-panel input[type="number"] {
  background: var(--tron-bg) !important;
  border: 1px solid rgba(0,240,255,0.15) !important;
  color: var(--tron-text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem;
  border-radius: 2px !important;
  padding: 5px 8px;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.sidebar-panel .form-control:focus,
.sidebar-panel input:focus {
  border-color: var(--tron-cyan) !important;
  box-shadow: 0 0 6px rgba(0,240,255,0.3) !important;
  outline: none;
}

/* ── DROPDOWNS ─────────────────────────────────────────────── */
.Select-control,
.Select-menu-outer,
.Select-option,
.css-1dimb5e-singleValue,
.css-qc6sy-singleValue {
  background-color: var(--tron-bg) !important;
  color: var(--tron-text) !important;
  border-color: rgba(0,240,255,0.15) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
}
.Select-option.is-focused,
.Select-option:hover {
  background-color: rgba(0,240,255,0.1) !important;
  color: var(--tron-cyan) !important;
}
.Select-value-label,
.Select-placeholder,
.css-1dimb5e-singleValue {
  color: var(--tron-text) !important;
}
.Select-multi-value-wrapper .Select-value {
  background: rgba(0,240,255,0.12) !important;
  border: 1px solid rgba(0,240,255,0.3) !important;
  color: var(--tron-cyan) !important;
}
.Select-clear-zone, .Select-arrow-zone {
  color: var(--tron-text-dim) !important;
}

/* ── RUN BUTTON ────────────────────────────────────────────── */
.tron-run-btn {
  background: linear-gradient(135deg, rgba(0,240,255,0.15), rgba(0,240,255,0.05)) !important;
  border: 1px solid var(--tron-cyan) !important;
  color: var(--tron-cyan) !important;
  font-family: var(--font-display) !important;
  font-weight: 700;
  font-size: 0.85rem;
  letter-spacing: 4px;
  text-transform: uppercase;
  padding: 12px 0;
  border-radius: 2px !important;
  transition: all 0.3s;
  text-shadow: 0 0 8px rgba(0,240,255,0.4);
  box-shadow: 0 0 12px rgba(0,240,255,0.15);
  cursor: pointer;
  position: relative;
  overflow: hidden;
}
.tron-run-btn:hover {
  background: linear-gradient(135deg, rgba(0,240,255,0.3), rgba(0,240,255,0.1)) !important;
  box-shadow: var(--tron-cyan-glow-strong) !important;
  color: #fff !important;
  text-shadow: 0 0 14px rgba(0,240,255,0.8);
}
.tron-run-btn:active {
  transform: scale(0.98);
}
.tron-run-btn::after {
  content: '';
  position: absolute;
  top: -2px; left: -2px; right: -2px; bottom: -2px;
  background: linear-gradient(135deg, var(--tron-cyan), transparent, var(--tron-magenta));
  opacity: 0;
  border-radius: 2px;
  z-index: -1;
  transition: opacity 0.3s;
}
.tron-run-btn:hover::after {
  opacity: 0.2;
}

/* ── TABS ──────────────────────────────────────────────────── */
.tab-container .tab {
  background: var(--tron-surface) !important;
  border: 1px solid var(--tron-border) !important;
  color: var(--tron-text-dim) !important;
  font-family: var(--font-display) !important;
  font-size: 0.68rem;
  letter-spacing: 2px;
  text-transform: uppercase;
  padding: 10px 18px !important;
  border-radius: 0 !important;
  transition: all 0.3s;
}
.tab-container .tab:hover {
  color: var(--tron-cyan) !important;
  background: rgba(0,240,255,0.05) !important;
}
.tab-container .tab--selected {
  background: linear-gradient(180deg, rgba(0,240,255,0.1), transparent) !important;
  border-bottom: 2px solid var(--tron-cyan) !important;
  color: var(--tron-cyan) !important;
  text-shadow: 0 0 8px rgba(0,240,255,0.3);
}

/* ── DATA TABLE ────────────────────────────────────────────── */
.dash-spreadsheet-container .dash-spreadsheet-inner td,
.dash-spreadsheet-container .dash-spreadsheet-inner th {
  background: var(--tron-surface) !important;
  color: var(--tron-text) !important;
  border: 1px solid rgba(0,240,255,0.08) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.75rem;
}
.dash-spreadsheet-container .dash-spreadsheet-inner th {
  background: linear-gradient(180deg, rgba(0,240,255,0.08), var(--tron-surface)) !important;
  color: var(--tron-cyan) !important;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  font-size: 0.68rem;
}
.dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td {
  background: rgba(0,240,255,0.04) !important;
}
.dash-spreadsheet-container .dash-spreadsheet-inner input {
  background: var(--tron-bg) !important;
  color: var(--tron-cyan) !important;
  border-color: rgba(0,240,255,0.2) !important;
}

/* ── CHECKLIST ─────────────────────────────────────────────── */
.form-check-label {
  font-family: var(--font-mono) !important;
  font-size: 0.75rem;
  color: var(--tron-text-dim) !important;
}
.form-check-input:checked {
  background-color: var(--tron-cyan) !important;
  border-color: var(--tron-cyan) !important;
}

/* ── DATE PICKER ───────────────────────────────────────────── */
.DateInput_input {
  background: var(--tron-bg) !important;
  color: var(--tron-text) !important;
  border-bottom: 1px solid rgba(0,240,255,0.2) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
}
.DateRangePickerInput {
  background: var(--tron-bg) !important;
  border: 1px solid rgba(0,240,255,0.15) !important;
  border-radius: 2px !important;
}
.DateRangePickerInput_arrow_svg {
  fill: var(--tron-cyan-dim) !important;
}

/* ── SCROLLBAR ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--tron-bg); }
::-webkit-scrollbar-thumb { background: rgba(0,240,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,240,255,0.4); }

/* ── STATUS TEXT ────────────────────────────────────────────── */
.tron-status {
  font-family: var(--font-mono) !important;
  font-size: 0.78rem;
  color: var(--tron-green);
  text-shadow: 0 0 6px rgba(0,255,136,0.3);
}

/* ── MISC ──────────────────────────────────────────────────── */
.tip-text {
  font-size: 0.68rem;
  color: var(--tron-text-dim);
  font-family: var(--font-mono);
  font-style: italic;
}
.alert-info {
  background: rgba(0,240,255,0.06) !important;
  border: 1px solid rgba(0,240,255,0.2) !important;
  color: var(--tron-text) !important;
  font-family: var(--font-mono);
}
h4 {
  font-family: var(--font-display) !important;
  font-size: 0.9rem !important;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--tron-cyan) !important;
  text-shadow: 0 0 8px rgba(0,240,255,0.3);
}
"""


def build_index_string(app_title: str) -> str:
    css = get_tron_css()
    return f"""
<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{app_title}</title>
    {{%favicon%}}
    {{%css%}}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>{css}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>
"""


def apply_theme(app, *, title: str):
    app.title = title
    app.index_string = build_index_string(title)
