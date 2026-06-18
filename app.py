import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="OsteoSense | Epic EHR",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; margin:0; padding:0; }
.stApp { background: #f4f5f7; }
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 100% !important;
    padding-top: 0 !important;
    padding-right: 0 !important;
    padding-bottom: 0 !important;
    padding-left: 0 !important;
}
/* ── Top bar ── */
.epic-topbar { background:#1b2a4a; display:flex; align-items:center; height:44px; border-bottom:1px solid #111e36; }
.epic-logo { background:#2a3f6f; color:white; font-size:1rem; font-weight:800; letter-spacing:1px; padding:0 20px; height:44px; display:flex; align-items:center; border-right:1px solid #111e36; min-width:68px; }
.topbar-tabs { display:flex; height:44px; align-items:stretch; flex:1; }s
.topbar-tab { color:rgba(255,255,255,0.6); font-size:0.8rem; font-weight:500; padding:0 18px; display:flex; align-items:center; border-bottom:3px solid transparent; white-space:nowrap; }
.topbar-tab.active { color:white; border-bottom:3px solid #5b9bd5; font-weight:600; }
.topbar-right { color:rgba(255,255,255,0.75); font-size:0.76rem; padding:0 20px; white-space:nowrap; }

/* ── Patient header ── */
.patient-header { background:#1e3260; padding:10px 20px; display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #152448; }
.patient-name { color:white; font-size:1.12rem; font-weight:700; margin-right:14px; }
.patient-meta { color:rgba(255,255,255,0.72); font-size:0.77rem; }
.status-badge { display:inline-block; padding:2px 10px; border-radius:3px; font-size:0.68rem; font-weight:700; letter-spacing:0.5px; }
.badge-stable   { background:#1e7a3e; color:white; }
.badge-atrisk   { background:#c26800; color:white; }
.badge-critical { background:#b00020; color:white; }
.sync-dot  { width:8px; height:8px; border-radius:50%; background:#3ccf74; display:inline-block; }
.sync-text { color:rgba(255,255,255,0.6); font-size:0.71rem; }

/* ── Sub-tabs ── */
.subtab-bar { background:white; border-bottom:1px solid #d6dae3; display:flex; padding:0 20px; }
.subtab { font-size:0.8rem; font-weight:500; color:#4a5568; padding:10px 16px; border-bottom:3px solid transparent; white-space:nowrap; }
.subtab.active { color:#1b2a4a; font-weight:700; border-bottom:3px solid #1b2a4a; }

/* ── Patient switcher ── */
.patient-switcher { background:#f0f2f5; border-bottom:1px solid #d6dae3; display:flex; align-items:center; padding:6px 20px; gap:6px; }
.pt-tab { padding:3px 13px; border-radius:14px; font-weight:500; color:#4a5568; background:white; border:1px solid #c8cdd8; font-size:0.72rem; }
.pt-tab.active { background:#1b2a4a; color:white; border-color:#1b2a4a; font-weight:600; }
.pt-label { color:#888; font-size:0.71rem; margin-right:4px; }

/* ── Status bar ── */
.status-bar { background:white; border-bottom:1px solid #d6dae3; display:flex; align-items:center; justify-content:space-between; padding:7px 20px; }
.status-dot-green  { width:9px; height:9px; border-radius:50%; background:#22c55e; display:inline-block; flex-shrink:0; }
.status-dot-orange { width:9px; height:9px; border-radius:50%; background:#f59e0b; display:inline-block; flex-shrink:0; }
.status-dot-red    { width:9px; height:9px; border-radius:50%; background:#ef4444; display:inline-block; flex-shrink:0; }
.status-text { font-size:0.8rem; color:#2d3748; display:flex; align-items:center; gap:8px; }
.ref-btn { background:#e8edf5; border:1px solid #b0bdd6; border-radius:4px; padding:5px 13px; font-size:0.74rem; font-weight:600; color:#1b2a4a; }

/* ── Left nav ── */
.left-nav { background:white; border-right:1px solid #d6dae3; padding:10px 0; height:100%; }
.nav-section-header { color:#8a9ab5; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.7px; padding:10px 14px 3px 14px; }
.nav-item { color:#2d3748; padding:6px 14px 6px 18px; font-size:0.77rem; font-weight:500; border-left:3px solid transparent; }
.nav-item.active { background:#eef2fb; color:#1b2a4a; font-weight:700; border-left:3px solid #1b2a4a; }

/* ── Content card ── */
.card-header { background:#f0f3f8; border:1px solid #d0d7e6; border-radius:5px 5px 0 0; padding:8px 16px; display:flex; align-items:center; justify-content:space-between; }
.card-brand  { font-weight:700; color:#1b2a4a; font-size:0.82rem; }
.card-title  { font-weight:600; color:#2d3748; font-size:0.82rem; margin-left:6px; }
.card-meta   { color:#8a9ab5; font-size:0.68rem; }

/* ── Three metric cards (matching Epic screenshot) ── */
.metric-row { display:flex; background:white; border:1px solid #d0d7e6; border-top:none; }
.metric-cell {
  flex:1; padding:18px 24px 14px 24px; text-align:center;
  border-right:1px solid #e4e8f0;
}
.metric-cell:last-child { border-right:none; }
.metric-LABEL {
  font-size:0.65rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.7px; color:#6b7a99; margin-bottom:2px;
}
.metric-sublabel { font-size:0.63rem; color:#9aa; font-style:italic; margin-bottom:4px; }
.metric-number {
  font-size:3.2rem; font-weight:800; line-height:1.05;
  letter-spacing:-1px;
}
.metric-number.col-normal      { color:#4a7c59; }
.metric-number.col-osteopenia  { color:#8b7000; }
.metric-number.col-osteoporosis{ color:#8b0000; }
.metric-denom { font-size:0.85rem; color:#aab; font-weight:400; margin-left:2px; }
.metric-bar-track { background:#e4e8f0; border-radius:3px; height:6px; margin:10px 0 5px 0; }
.metric-bar-fill  { height:100%; border-radius:3px; }
.metric-footer { font-size:0.67rem; color:#8a9ab5; margin-top:3px; }
.metric-footer.warn-footer { color:#c26800; font-weight:600; }

/* ── AI summary block ── */
.ai-summary-wrap { background:white; border:1px solid #d0d7e6; border-top:none; padding:10px 16px; }
.ai-summary-label { font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.7px; color:#6b7a99; margin-bottom:5px; }
.ai-summary-text  { font-size:0.81rem; color:#2d3748; line-height:1.55; }

/* ── Section within card ── */
.inner-section { background:white; border:1px solid #d0d7e6; border-top:none; padding:12px 16px; }
.inner-section-title { font-size:0.72rem; font-weight:600; color:#2d3748; margin-bottom:8px; display:flex; align-items:center; justify-content:space-between; }

/* ── Chip ── */
.chip { display:inline-block; padding:2px 9px; border-radius:11px; font-size:0.67rem; font-weight:600; border:1px solid; margin-left:6px; }
.chip-green  { color:#1a6e3a; border-color:#1a6e3a; background:#e8f8ef; }
.chip-orange { color:#8b5000; border-color:#c26800; background:#fff4e0; }
.chip-red    { color:#9b0000; border-color:#b00020; background:#fdeaea; }

/* ── Right panel ── */
.rpanel-card  { background:white; border:1px solid #d0d7e6; border-radius:5px; margin-bottom:10px; overflow:hidden; }
.rpanel-title { background:#f0f3f8; font-size:0.69rem; font-weight:700; color:#2d3748; padding:7px 12px; border-bottom:1px solid #d0d7e6; text-transform:uppercase; letter-spacing:0.4px; }
.rpanel-row   { display:flex; justify-content:space-between; align-items:baseline; padding:7px 12px; border-bottom:1px solid #eef0f5; font-size:0.76rem; }
.rpanel-row:last-child { border-bottom:none; }
.rpanel-key   { color:#8a9ab5; font-size:0.7rem; min-width:100px; }
.rpanel-val   { color:#1e2a40; font-weight:600; text-align:right; }

/* ── Data table ── */
.data-table { width:100%; border-collapse:collapse; font-size:0.77rem; }
.data-table th { background:#f0f3f8; color:#2d3748; padding:7px 12px; text-align:left; font-weight:600; border-bottom:1px solid #d0d7e6; }
.data-table td { padding:7px 12px; border-bottom:1px solid #eef0f5; }
.data-table tr:last-child td { border-bottom:none; }
.note-box  { background:#fffde7; border:1px solid #f9c800; border-radius:4px; padding:12px 14px; font-size:0.79rem; margin-bottom:10px; }
.note-meta { font-size:0.68rem; color:#888; margin-bottom:5px; }

section[data-testid="stSidebar"] { display:none !important; }
.stButton > button { font-family:'Inter',sans-serif; font-size:0.78rem; }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "subtab"          not in st.session_state: st.session_state.subtab          = "BoneScore"
if "active_patient"  not in st.session_state: st.session_state.active_patient  = "Harrington, J."

# ── Model ─────────────────────────────────────────────────────────────────────
LABEL_ORDER              = ["Normal", "Osteopenia", "Osteoporosis"]
QUS_OSTEOPENIA_THRESHOLD = -1.0

@st.cache_resource
def load_and_train():
    raw = pd.read_csv("dataset_heel.csv")
    df  = pd.DataFrame({
        "tscore":       pd.to_numeric(raw["QUS_T"],       errors="coerce"),
        "osteoporosis": pd.to_numeric(raw["osteoporosis"], errors="coerce"),
    }).dropna().reset_index(drop=True)

    def classify(row):
        if row["osteoporosis"] == 1:                          return "Osteoporosis"
        elif row["tscore"] >= QUS_OSTEOPENIA_THRESHOLD:       return "Normal"
        else:                                                 return "Osteopenia"

    df["diagnosis"] = df.apply(classify, axis=1)
    le = LabelEncoder(); le.fit(LABEL_ORDER)
    X = df[["tscore"]].values
    y = le.transform(df["diagnosis"])
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    best_name, best_model, best_score = None, None, -1
    for mn, mdl in candidates.items():
        cv = cross_val_score(mdl, X, y, cv=5, scoring="f1_weighted").mean()
        if cv > best_score:
            best_name, best_model, best_score = mn, mdl, cv
    best_model.fit(X_tr, y_tr)
    return {"model": best_model, "model_name": best_name, "f1": round(best_score, 4)}, le

entry, le = load_and_train()

# ── Patient roster ────────────────────────────────────────────────────────────
patients = {
    "Harrington, J.": {
        "name": "Harrington, Jane G.", "age": 68, "sex": "F",
        "dob": "03/14/1957", "mrn": "004-62-8819",
        "pcp": "Dr. A. Patel, MD", "dept": "Primary Care",
        "tscore": -2.5,
        "history": [
            ("2019-04", -0.8), ("2020-07", -1.2), ("2021-03", -1.5),
            ("2021-09", -1.7), ("2022-05", -1.9), ("2022-11", -2.0),
            ("2023-06", -2.2), ("2023-12", -2.3), ("2024-06", -2.5),
        ],
    },
    "Okafor, B.": {
        "name": "Okafor, Bunmi A.", "age": 54, "sex": "F",
        "dob": "07/22/1971", "mrn": "009-41-3302",
        "pcp": "Dr. S. Kim, MD", "dept": "Primary Care",
        "tscore": -1.4,
        "history": [
            ("2020-01", -0.5), ("2021-01", -0.8),
            ("2022-01", -1.0), ("2023-01", -1.2), ("2024-01", -1.4),
        ],
    },
    "Vasquez, M.": {
        "name": "Vasquez, Maria L.", "age": 72, "sex": "F",
        "dob": "11/05/1953", "mrn": "002-88-6671",
        "pcp": "Dr. A. Patel, MD", "dept": "Primary Care",
        "tscore": -3.1,
        "history": [
            ("2018-03", -1.8), ("2019-03", -2.1), ("2020-03", -2.4),
            ("2021-03", -2.6), ("2022-03", -2.8), ("2023-03", -3.0),
            ("2024-03", -3.1),
        ],
    },
    "Lindqvist, E.": {
        "name": "Lindqvist, Erik P.", "age": 61, "sex": "M",
        "dob": "04/30/1964", "mrn": "007-14-9953",
        "pcp": "Dr. R. Gupta, MD", "dept": "Primary Care",
        "tscore": -0.6,
        "history": [
            ("2021-05", -0.2), ("2022-05", -0.4),
            ("2023-05", -0.5), ("2024-05", -0.6),
        ],
    },
}

pt     = patients[st.session_state.active_patient]
tscore = pt["tscore"]

probs_raw  = entry["model"].predict_proba([[tscore]])[0]
prob_dict  = dict(zip(le.classes_, probs_raw))
pred_label = max(prob_dict, key=prob_dict.get)

DX_COLORS = {"Normal": "#4a7c59", "Osteopenia": "#8b7000", "Osteoporosis": "#8b0000"}
DX_CSS    = {"Normal": "col-normal", "Osteopenia": "col-osteopenia", "Osteoporosis": "col-osteoporosis"}
CHIP_CLS  = {"Normal": "chip-green", "Osteopenia": "chip-orange", "Osteoporosis": "chip-red"}

if pred_label == "Osteoporosis":
    badge_cls, badge_txt, dot_cls = "badge-critical", "OSTEOPOROSIS", "status-dot-red"
    status_msg = f"T-score alert — QUS {tscore:+.1f}. WHO osteoporosis criteria met. Pharmacotherapy recommended."
elif pred_label == "Osteopenia":
    badge_cls, badge_txt, dot_cls = "badge-atrisk",   "LOW BONE MASS", "status-dot-orange"
    status_msg = f"Routine monitoring — QUS T-score {tscore:+.1f}. Low bone mass. Lifestyle counseling active."
else:
    badge_cls, badge_txt, dot_cls = "badge-stable",   "NORMAL",        "status-dot-green"
    status_msg = f"Routine monitoring — QUS T-score {tscore:+.1f}. Bone density within normal range. No escalation."

prev_ts = pt["history"][-2][1] if len(pt["history"]) >= 2 else tscore
delta   = tscore - prev_ts

# ════════════════════════════════════════════════════════════════════════════
#  TOP BAR
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="epic-topbar">
  <div class="epic-logo">Epic</div>
  <div class="topbar-tabs">
    <div class="topbar-tab active">Chart</div>
    <div class="topbar-tab">Schedule</div>
    <div class="topbar-tab">Results</div>
    <div class="topbar-tab">Orders</div>
    <div class="topbar-tab">Messages</div>
  </div>
  <div class="topbar-right">{pt['pcp']} &nbsp;|&nbsp; {pt['dept']}</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PATIENT HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="patient-header">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <span class="patient-name">{pt['name']}</span>
    <span class="patient-meta">{pt['age']}{pt['sex']} &nbsp;|&nbsp; DOB: {pt['dob']} &nbsp;|&nbsp; MRN: {pt['mrn']}</span>
    <span class="status-badge {badge_cls}">{badge_txt}</span>
  </div>
  <div style="display:flex;align-items:center;gap:6px;">
    <span class="sync-dot"></span>
    <span class="sync-text">OsteoSense synced: Today {datetime.datetime.now().strftime('%I:%M %p')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SUB-TABS
# ════════════════════════════════════════════════════════════════════════════
subtabs  = ["BoneScore", "Scan History", "Medications", "Notes", "Vitals", "Summary"]
tab_html = '<div class="subtab-bar">'
for t in subtabs:
    cls = "subtab active" if t == st.session_state.subtab else "subtab"
    tab_html += f'<div class="{cls}">{t}</div>'
tab_html += '</div>'
st.markdown(tab_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PATIENT SWITCHER
# ════════════════════════════════════════════════════════════════════════════
sw_html = '<div class="patient-switcher"><span class="pt-label">Patient:</span>'
for pname in patients:
    cls = "pt-tab active" if pname == st.session_state.active_patient else "pt-tab"
    sw_html += f'<span class="{cls}">{pname}</span>'
sw_html += '</div>'
st.markdown(sw_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  STATUS BAR
# ════════════════════════════════════════════════════════════════════════════
parts     = status_msg.split("—", 1)
bold_part = parts[0].strip()
rest_part = parts[1].strip() if len(parts) > 1 else ""
st.markdown(f"""
<div class="status-bar">
  <div class="status-text">
    <span class="{dot_cls}"></span>
    <b>{bold_part}</b> — {rest_part}
  </div>
  <div class="ref-btn">📋 &nbsp;Refer to specialist</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  BODY: left nav | center | right
# ════════════════════════════════════════════════════════════════════════════
left_col, center_col, right_col = st.columns([1.05, 4.5, 1.8])

# ── LEFT NAV ─────────────────────────────────────────────────────────────────
with left_col:
    nav_sections = {
        "BONESCORE":    ["AI score summary", "Scan history"],
        "OSTEOPOROSIS": ["Medication log", "Treatment plan"],
        "QUICK LINKS":  ["Problem list", "Current meds", "Last visit note"],
    }
    nav_html = '<div class="left-nav">'
    for sec, items in nav_sections.items():
        nav_html += f'<div class="nav-section-header">{sec}</div>'
        for i, item in enumerate(items):
            cls = "nav-item active" if (sec == "BONESCORE" and i == 0) else "nav-item"
            nav_html += f'<div class="{cls}">{item}</div>'
    nav_html += '</div>'
    st.markdown(nav_html, unsafe_allow_html=True)

# ── CENTER PANEL ──────────────────────────────────────────────────────────────
with center_col:
    active_tab = st.session_state.subtab

    # ══════════════════════════════════════════════════════════════════════
    #  BONESCORE TAB
    # ══════════════════════════════════════════════════════════════════════
    if active_tab == "BoneScore":

        # ── Card header ───────────────────────────────────────────────────
        st.markdown(f"""
        <div class="card-header">
          <div>
            <span class="card-brand">OsteoSense</span>
            <span style="color:#bbb;margin:0 6px;">|</span>
            <span class="card-title">BoneScore — AI verified</span>
          </div>
          <div class="card-meta">Device: QUS-C01 &nbsp;|&nbsp; OsteoSense synced: Today {datetime.datetime.now().strftime('%I:%M %p')}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Three metric cards ────────────────────────────────────────────
        # Card 1: QUS T-score  (range −4 to +1, map to 0-100%)
        ts_pct   = max(2, min(98, int((tscore + 4) / 5 * 100)))
        ts_col   = DX_COLORS[pred_label]
        ts_css   = DX_CSS[pred_label]

        # Card 2: Top model probability
        top_prob = prob_dict[pred_label] * 100
        prob_pct = int(top_prob)

        # Card 3: Previous scan T-score
        prev_pct = max(2, min(98, int((prev_ts + 4) / 5 * 100)))
        if delta < -0.05:
            prev_footer = f'<span class="metric-footer warn-footer">▼ declining ({delta:+.2f} SD)</span>'
        elif delta > 0.05:
            prev_footer = f'<span class="metric-footer" style="color:#4a7c59;font-weight:600;">▲ improving ({delta:+.2f} SD)</span>'
        else:
            prev_footer = '<span class="metric-footer">≈ stable</span>'

        st.markdown(f"""
        <div class="metric-row">

          <div class="metric-cell">
            <div class="metric-LABEL">QUS T-score</div>
            <div class="metric-sublabel">Calcaneus — AI adjusted</div>
            <div>
              <span class="metric-number {ts_css}">{tscore:+.1f}</span>
              <span class="metric-denom">SD</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" style="width:{ts_pct}%;background:{ts_col};"></div>
            </div>
            <span class="metric-footer">primary score</span>
          </div>

          <div class="metric-cell">
            <div class="metric-LABEL">Model confidence</div>
            <div class="metric-sublabel">Predicted class probability</div>
            <div>
              <span class="metric-number {ts_css}">{top_prob:.1f}</span>
              <span class="metric-denom">%</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" style="width:{prob_pct}%;background:{ts_col};"></div>
            </div>
            <span class="metric-footer">{pred_label} &nbsp;<span class="chip {CHIP_CLS[pred_label]}">{pred_label}</span></span>
          </div>

          <div class="metric-cell">
            <div class="metric-LABEL">Previous scan</div>
            <div class="metric-sublabel">Last recorded T-score ({pt['history'][-2][0] if len(pt['history'])>=2 else '—'})</div>
            <div>
              <span class="metric-number {DX_CSS['Osteoporosis'] if prev_ts<=-2.725 else (DX_CSS['Osteopenia'] if prev_ts<=-1.0 else DX_CSS['Normal'])}">{prev_ts:+.1f}</span>
              <span class="metric-denom">SD</span>
            </div>
            <div class="metric-bar-track">
              <div class="metric-bar-fill" style="width:{prev_pct}%;background:{DX_COLORS['Osteoporosis'] if prev_ts<=-2.725 else (DX_COLORS['Osteopenia'] if prev_ts<=-1.0 else DX_COLORS['Normal'])};"></div>
            </div>
            {prev_footer}
          </div>

        </div>
        """, unsafe_allow_html=True)

        # ── AI Clinical Summary ───────────────────────────────────────────
        ai_text = {
            "Osteoporosis": f"Osteoporosis confirmed by QUS T-score {tscore:+.2f} — meets WHO diagnostic threshold (≤ −2.725). Model predicts osteoporosis with {top_prob:.1f}% confidence. Patient on bisphosphonate therapy. Fall prevention referral recommended. No acute fracture on current imaging.",
            "Osteopenia":   f"Low bone mass (osteopenia) — QUS T-score {tscore:+.2f}. Model confidence {top_prob:.1f}%. Patient ambulatory and completing daily activities independently. Calcium and Vitamin D supplementation active. Re-scan in 12 months.",
            "Normal":       f"Normal bone density — QUS T-score {tscore:+.2f}. Model predicts normal with {top_prob:.1f}% confidence. No intervention indicated at this time. Routine screening recommended per USPSTF guidelines.",
        }[pred_label]

        st.markdown(f"""
        <div class="ai-summary-wrap">
          <div class="ai-summary-label">OsteoSense AI Clinical Summary</div>
          <div class="ai-summary-text">{ai_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bar chart ─────────────────────────────────────────
        st.markdown('<div class="inner-section"><div class="inner-section-title">Diagnostic probability — model output</div>', unsafe_allow_html=True)

        labels     = ["Normal", "Osteopenia", "Osteoporosis"]
        values     = [prob_dict.get(lbl, 0) for lbl in labels]
        bar_colors = [DX_COLORS[lbl] for lbl in labels]

        fig_prob = go.Figure()
        fig_prob.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=bar_colors,
            text=[f"{v*100:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=13, family="Inter", color="#1e2a40"),
            width=0.42,
            hovertemplate="<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>",
        ))
        # Highlight predicted class
        idx = labels.index(pred_label)
        fig_prob.add_shape(
            type="rect",
            x0=idx - 0.24, x1=idx + 0.24,
            y0=0, y1=prob_dict[pred_label] + 0.05,
            line=dict(color=DX_COLORS[pred_label], width=2, dash="dot"),
            fillcolor="rgba(0,0,0,0)",
        )
        fig_prob.update_layout(
            height=235,
            margin=dict(l=10, r=10, t=24, b=8),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(tickformat=".0%", range=[0, 1.18],
                       gridcolor="#eef0f5",
                       title=dict(text="Probability", font=dict(size=11, color="#8a9ab5")),
                       tickfont=dict(size=11)),
            xaxis=dict(tickfont=dict(size=12, family="Inter", color="#2d3748"), fixedrange=True),
            font=dict(family="Inter"),
            showlegend=False, bargap=0.38,
        )
        st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Longitudinal T-score trend ────────────────────────────────────
        st.markdown('<div class="inner-section" style="border-top:none;"><div class="inner-section-title">T-score trend — longitudinal history</div>', unsafe_allow_html=True)

        hist       = pt["history"]
        hist_dates = [datetime.datetime.strptime(d + "-01", "%Y-%m-%d") for d, _ in hist]
        hist_vals  = [v for _, v in hist]
        pt_colors  = [DX_COLORS["Osteoporosis"] if v <= -2.725
                      else (DX_COLORS["Osteopenia"] if v <= -1.0 else DX_COLORS["Normal"])
                      for v in hist_vals]

        fig_trend = go.Figure()

        # WHO zone shading
        fig_trend.add_hrect(y0=-1.0,   y1=1.5,  fillcolor="rgba(74,124,89,0.07)",  line_width=0)
        fig_trend.add_hrect(y0=-2.725, y1=-1.0, fillcolor="rgba(139,112,0,0.08)",  line_width=0)
        fig_trend.add_hrect(y0=-5.0,   y1=-2.725,fillcolor="rgba(139,0,0,0.08)",   line_width=0)

        fig_trend.add_hline(y=-1.0,   line_dash="dash", line_color="#8b7000", line_width=1.2,
                            annotation_text="Osteopenia (−1.0)",
                            annotation_font_size=9, annotation_font_color="#8b7000",
                            annotation_position="top right")
        fig_trend.add_hline(y=-2.725, line_dash="dash", line_color="#8b0000", line_width=1.2,
                            annotation_text="Osteoporosis (−2.725)",
                            annotation_font_size=9, annotation_font_color="#8b0000",
                            annotation_position="bottom right")

        fig_trend.add_trace(go.Scatter(
            x=hist_dates, y=hist_vals,
            mode="lines+markers",
            line=dict(color="#1b4fa0", width=2.5),
            marker=dict(size=9, color=pt_colors, line=dict(color="white", width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>T-score: %{y:+.2f}<extra></extra>",
            showlegend=False,
        ))
        fig_trend.add_annotation(
            x=hist_dates[-1], y=hist_vals[-1],
            text=f"  {hist_vals[-1]:+.2f}",
            showarrow=False,
            font=dict(size=11, color="#1b4fa0", family="Inter"),
            xanchor="left",
        )
        # Zone legend
        for lbl, col in DX_COLORS.items():
            fig_trend.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=col, size=9), name=lbl,
            ))
        fig_trend.update_layout(
            height=270,
            margin=dict(l=10, r=80, t=8, b=8),
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(title=dict(text="T-score (SD)", font=dict(size=11, color="#8a9ab5")),
                       gridcolor="#eef0f5", range=[-4.2, 0.8],
                       tickfont=dict(size=10), zeroline=False),
            xaxis=dict(gridcolor="#eef0f5", tickfont=dict(size=10), tickformat="%b %Y"),
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=9),
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  SCAN HISTORY TAB
    # ══════════════════════════════════════════════════════════════════════
    elif active_tab == "Scan History":
        st.markdown("""<div class="card-header"><div>
          <span class="card-brand">OsteoSense</span>
          <span style="color:#bbb;margin:0 6px;">|</span>
          <span class="card-title">Scan History</span></div></div>
        <div style="background:white;border:1px solid #d0d7e6;border-top:none;padding:14px 16px;">""",
        unsafe_allow_html=True)

        tbl = """<table class="data-table">
          <tr><th>Date</th><th>Modality</th><th>T-score</th><th>Diagnosis</th><th>Provider</th></tr>"""
        for d, v in reversed(pt["history"]):
            dx  = "Osteoporosis" if v <= -2.725 else ("Osteopenia" if v <= -1.0 else "Normal")
            tbl += f"""<tr>
              <td>{d.replace('-',' ')}</td><td>Calcaneus QUS</td>
              <td style="font-weight:700;">{v:+.2f}</td>
              <td><span class="chip {CHIP_CLS[dx]}">{dx}</span></td>
              <td>{pt['pcp']}</td></tr>"""
        st.markdown(tbl + "</table></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  MEDICATIONS TAB
    # ══════════════════════════════════════════════════════════════════════
    elif active_tab == "Medications":
        st.markdown("""<div class="card-header"><div>
          <span class="card-brand">OsteoSense</span>
          <span style="color:#bbb;margin:0 6px;">|</span>
          <span class="card-title">Current Medications</span></div></div>
        <div style="background:white;border:1px solid #d0d7e6;border-top:none;padding:14px 16px;">""",
        unsafe_allow_html=True)

        tbl = """<table class="data-table">
          <tr><th>Medication</th><th>Sig</th><th>Indication</th><th>Status</th></tr>"""
        for med, sig, ind in [
            ("Alendronate 70 mg",        "PO weekly",           "Osteoporosis"),
            ("Calcium Carbonate 1000 mg", "PO BID with food",    "Bone health"),
            ("Vitamin D3 1000 IU",        "PO daily",            "Vitamin D deficiency"),
            ("Lisinopril 10 mg",          "PO daily",            "Hypertension"),
            ("Atorvastatin 20 mg",        "PO daily at bedtime", "Hyperlipidemia"),
        ]:
            tbl += f"<tr><td><b>{med}</b></td><td>{sig}</td><td>{ind}</td><td><span class='chip chip-green'>Active</span></td></tr>"
        st.markdown(tbl + "</table></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  NOTES TAB
    # ══════════════════════════════════════════════════════════════════════
    elif active_tab == "Notes":
        st.markdown("""<div class="card-header"><div>
          <span class="card-brand">OsteoSense</span>
          <span style="color:#bbb;margin:0 6px;">|</span>
          <span class="card-title">Clinical Notes</span></div></div>
        <div style="background:white;border:1px solid #d0d7e6;border-top:none;padding:14px 16px;">""",
        unsafe_allow_html=True)

        new_note = st.text_area("New note", height=110,
            value=f"S: Patient presents for routine bone density review.\nO: QUS T-score {tscore:+.2f}. Model predicts {pred_label}.\nA: {pred_label}.\nP: Continue current regimen. Repeat QUS in 12 months.")
        if st.button("Sign & Save"):
            st.success("Note signed and saved to chart.")

        for time_, ntype, author, text in [
            ("2024-06-18 10:30", "Progress Note", pt["pcp"],
             f"Annual bone density review. QUS T-score {tscore:+.2f}. Patient tolerating alendronate well. Continue regimen. Repeat QUS annually."),
            ("2023-01-15 14:45", "Consultation Note", "Dr. R. Gupta, MD",
             "Osteoporosis management consult. Initiated alendronate 70 mg weekly. Added calcium + Vit D."),
        ]:
            st.markdown(f"""
            <div class="note-box" style="margin-top:10px;">
              <div class="note-meta"><b style="color:#1b2a4a;">{author}</b> &nbsp;|&nbsp; {ntype} &nbsp;|&nbsp; {time_}
              <span class="chip chip-green" style="margin-left:8px;">SIGNED</span></div>
              <div style="white-space:pre-wrap;line-height:1.6;">{text}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  VITALS TAB
    # ══════════════════════════════════════════════════════════════════════
    elif active_tab == "Vitals":
        st.markdown("""<div class="card-header"><div>
          <span class="card-brand">OsteoSense</span>
          <span style="color:#bbb;margin:0 6px;">|</span>
          <span class="card-title">Vitals</span></div></div>
        <div style="background:white;border:1px solid #d0d7e6;border-top:none;padding:14px 16px;">""",
        unsafe_allow_html=True)

        tbl = """<table class="data-table">
          <tr><th>Vital</th><th>Value</th><th>Ref. Range</th><th>Flag</th></tr>"""
        for v, val, ref, col in [
            ("Blood Pressure", "134/82 mmHg", "< 130/80",  "#c26800"),
            ("Heart Rate",     "72 bpm",       "60–100",    "#4a7c59"),
            ("Temperature",    "98.4 °F",      "97–99",     "#4a7c59"),
            ("Weight",         "58.2 kg",       "—",         "#4a7c59"),
            ("Height",         "163 cm",        "—",         "#4a7c59"),
            ("BMI",            "21.9 kg/m²",   "18.5–24.9", "#4a7c59"),
            ("SpO₂",           "98%",           "> 95%",     "#4a7c59"),
        ]:
            flag = "⚡ Elevated" if col == "#c26800" else "✓ Normal"
            tbl += f"<tr><td>{v}</td><td><b>{val}</b></td><td style='color:#8a9ab5;'>{ref}</td><td style='color:{col};font-size:0.72rem;font-weight:600;'>{flag}</td></tr>"
        st.markdown(tbl + "</table></div>", unsafe_allow_html=True)

    else:
        st.markdown(f"""<div class="card-header"><div><span class="card-brand">OsteoSense</span>
          <span style="color:#bbb;margin:0 6px;">|</span>
          <span class="card-title">{active_tab}</span></div></div>
        <div style="background:white;border:1px solid #d0d7e6;border-top:none;padding:40px;text-align:center;color:#8a9ab5;">
          Select a section from the left navigation.
        </div>""", unsafe_allow_html=True)

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
with right_col:
    next_scan = (datetime.datetime.strptime(pt["history"][-1][0] + "-01", "%Y-%m-%d")
                 + datetime.timedelta(days=365)).strftime("%b %Y")

    # Patient details
    st.markdown(f"""
    <div class="rpanel-card">
      <div class="rpanel-title">Patient details</div>
      <div class="rpanel-row"><span class="rpanel-key">Diagnosis</span><span class="rpanel-val">{pred_label}</span></div>
      <div class="rpanel-row"><span class="rpanel-key">PCP</span><span class="rpanel-val">{pt['pcp']}</span></div>
      <div class="rpanel-row"><span class="rpanel-key">Device</span><span class="rpanel-val">QUS-C01 (Calcaneus)</span></div>
      <div class="rpanel-row"><span class="rpanel-key">Last scan</span><span class="rpanel-val">{pt['history'][-1][0].replace('-',' ')}</span></div>
      <div class="rpanel-row"><span class="rpanel-key">Next scan due</span><span class="rpanel-val">{next_scan}</span></div>
      <div class="rpanel-row"><span class="rpanel-key">Scans on record</span><span class="rpanel-val">{len(pt['history'])}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # T-score change
    d_color = DX_COLORS["Osteoporosis"] if delta < -0.05 else ("#4a7c59" if delta > 0.05 else "#8a9ab5")
    d_arrow = "▼" if delta < -0.05 else ("▲" if delta > 0.05 else "—")
    d_label = "Declining" if delta < -0.05 else ("Improving" if delta > 0.05 else "Stable")
    st.markdown(f"""
    <div class="rpanel-card">
      <div class="rpanel-title">T-score change</div>
      <div style="padding:12px 14px;text-align:center;">
        <div style="font-size:2rem;font-weight:800;color:{d_color};line-height:1;">{d_arrow} {abs(delta):.2f}</div>
        <div style="font-size:0.68rem;color:#8a9ab5;margin-top:3px;">SD since {pt['history'][-2][0] if len(pt['history'])>=2 else '—'}</div>
        <div style="font-size:0.8rem;font-weight:600;color:{d_color};margin-top:5px;">{d_label}</div>
      </div>
      <div class="rpanel-row"><span class="rpanel-key">Previous</span><span class="rpanel-val">{prev_ts:+.2f}</span></div>
      <div class="rpanel-row"><span class="rpanel-key">Current</span><span class="rpanel-val">{tscore:+.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Model confidence
    st.markdown('<div class="rpanel-card"><div class="rpanel-title">Model confidence</div>', unsafe_allow_html=True)
    for lbl in ["Normal", "Osteopenia", "Osteoporosis"]:
        pct  = prob_dict.get(lbl, 0) * 100
        w    = max(int(pct), 2)
        col  = DX_COLORS[lbl]
        bold = "font-weight:700;" if lbl == pred_label else ""
        st.markdown(f"""
        <div style="padding:6px 12px;border-bottom:1px solid #eef0f5;">
          <div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:3px;">
            <span style="{bold}color:#2d3748;">{lbl}</span>
            <span style="font-weight:700;color:{col};">{pct:.1f}%</span>
          </div>
          <div style="background:#eef0f5;border-radius:3px;height:6px;">
            <div style="width:{w}%;height:100%;border-radius:3px;background:{col};"></div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:7px 12px;font-size:0.67rem;color:#8a9ab5;">
      {entry['model_name']} &nbsp;·&nbsp; F1: {entry['f1']:.4f}
    </div></div>""", unsafe_allow_html=True)

    # Clinical guidance
    guidance = {
        "Osteoporosis": "⚠ <b>Pharmacotherapy indicated.</b> Consider bisphosphonate therapy. Fall prevention referral. Calcium + Vitamin D. Re-evaluate in 12 months.",
        "Osteopenia":   "⚡ <b>Lifestyle modification.</b> Calcium + Vitamin D. Weight-bearing exercise. Smoking cessation if applicable. Re-scan in 1–2 years.",
        "Normal":       "✓ <b>No intervention needed.</b> Continue routine USPSTF screening. Maintain adequate calcium intake. Re-scan per age-based guidelines.",
    }[pred_label]
    st.markdown(f"""
    <div class="rpanel-card">
      <div class="rpanel-title">Clinical guidance</div>
      <div style="padding:10px 12px;font-size:0.74rem;line-height:1.65;color:#2d3748;">{guidance}</div>
    </div>""", unsafe_allow_html=True)
