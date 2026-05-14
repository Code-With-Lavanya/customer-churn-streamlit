import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Prediction Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar fully hidden
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:        #0b0f1a;
  --surface:   #111827;
  --surface2:  #1a2236;
  --surface3:  #212d45;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(255,255,255,0.12);
  --blue:      #3b82f6;
  --bluel:     #60a5fa;
  --cyan:      #06b6d4;
  --green:     #10b981;
  --amber:     #f59e0b;
  --red:       #ef4444;
  --text:      #f1f5f9;
  --text2:     #94a3b8;
  --text3:     #475569;
  --r:         12px;
  --rl:        18px;
  --rxl:       22px;
  --font:      'Plus Jakarta Sans', sans-serif;
  --mono:      'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
  font-family: var(--font) !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

.stApp {
  background: var(--bg) !important;
  background-image: radial-gradient(ellipse 80% 40% at 50% -5%,
    rgba(59,130,246,0.09) 0%, transparent 60%);
}

/* ── collapse sidebar toggle button ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }

/* ── full-width layout ── */
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ═══════ NAV ═══════ */
.navbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 13px 36px;
  background: rgba(17,24,39,0.95);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap; gap: 10px;
}
.nav-left  { display: flex; align-items: center; gap: 10px; }
.nav-icon  {
  width: 38px; height: 38px; border-radius: 10px;
  background: linear-gradient(135deg, var(--blue), var(--cyan));
  display: flex; align-items: center; justify-content: center;
  font-size: 19px; box-shadow: 0 0 22px rgba(59,130,246,0.4);
  flex-shrink: 0;
}
.nav-title {
  font-size: 1.1rem; font-weight: 800; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #fff, var(--bluel));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-badge {
  font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 3px 9px; border-radius: 100px;
  background: rgba(59,130,246,0.12); color: var(--bluel);
  border: 1px solid rgba(59,130,246,0.25);
}
.nav-right { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.nav-stat  { font-size: 0.72rem; color: var(--text2); line-height: 1.4; }
.nav-stat  strong { display: block; font-size: 0.84rem; }
.nav-sep   { width: 1px; height: 28px; background: var(--border2); }

/* ═══════ PAGE ═══════ */
.page {
  padding: 28px 36px 56px;
  max-width: 1500px;
  margin: 0 auto;
}

/* ═══════ KPI ROW ═══════ */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 22px;
}
.kpi {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--rl); padding: 18px 20px;
  transition: border-color 0.2s, transform 0.15s;
}
.kpi:hover { border-color: var(--border2); transform: translateY(-2px); }
.kpi-top   { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.kpi-ico   { width: 36px; height: 36px; border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 16px; }
.kpi-badge { font-size: 0.66rem; font-weight: 600; padding: 2px 8px; border-radius: 100px; }
.kpi-lbl   { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text2); margin-bottom: 4px; }
.kpi-val   { font-family: var(--mono); font-size: 1.8rem; font-weight: 700; line-height: 1; letter-spacing: -0.03em; }
.kpi-sub   { font-size: 0.7rem; color: var(--text3); margin-top: 4px; }

/* ═══════ INPUT CARD ═══════ */
.input-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--rxl); padding: 24px 22px;
  height: 100%;
}
.card-title {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text3);
  border-bottom: 1px solid var(--border);
  padding-bottom: 12px; margin-bottom: 18px;
  display: flex; align-items: center; gap: 7px;
}

/* ── Option buttons ── */
.opt-group  { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 4px; }
.opt-btn    {
  padding: 6px 13px; border-radius: 8px; font-size: 0.79rem;
  font-weight: 500; cursor: pointer; border: 1px solid var(--border2);
  background: var(--surface2); color: var(--text2);
  transition: all 0.15s; font-family: var(--font);
  white-space: nowrap;
}
.opt-btn:hover  { border-color: var(--bluel); color: var(--bluel); }
.opt-active     { background: rgba(59,130,246,0.15) !important; border-color: var(--blue) !important; color: var(--bluel) !important; font-weight: 600 !important; }

/* ── Field labels ── */
.field-lbl { font-size: 0.76rem; font-weight: 600; color: var(--text2); margin-bottom: 6px; margin-top: 16px; letter-spacing: 0.03em; }

/* ── Predict btn ── */
.stButton > button {
  background: linear-gradient(135deg, var(--blue), var(--cyan)) !important;
  color: #fff !important; border: none !important;
  border-radius: var(--r) !important; font-family: var(--font) !important;
  font-weight: 700 !important; font-size: 0.95rem !important;
  letter-spacing: 0.02em !important; padding: 13px 0 !important;
  width: 100% !important;
  box-shadow: 0 4px 22px rgba(59,130,246,0.3) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  box-shadow: 0 4px 36px rgba(59,130,246,0.5) !important;
  transform: translateY(-1px) !important;
}

/* ── Streamlit slider/number overrides ── */
[data-testid="stSlider"] > div > div > div { background: var(--surface3) !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--blue) !important; }
[data-testid="stNumberInput"] > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r) !important;
}
[data-testid="stNumberInput"] input { color: var(--text) !important; }
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: var(--text2) !important; font-size: 0.8rem !important; }

/* ═══════ RESULT PANELS ═══════ */
.result-card {
  background: var(--surface); border-radius: var(--rxl);
  padding: 26px 22px; border: 1px solid;
  text-align: center; position: relative; overflow: hidden;
  margin-bottom: 14px;
}
.result-card::before {
  content:''; position:absolute; inset:0;
  background:radial-gradient(ellipse 90% 55% at 50% 0%, currentColor, transparent 65%);
  opacity: 0.06;
}
.res-lbl  { font-size: 0.66rem; letter-spacing: 0.13em; text-transform: uppercase; font-weight: 700; opacity: 0.75; margin-bottom: 6px; }
.res-pct  { font-family: var(--mono); font-size: 3.8rem; font-weight: 800; line-height: 1; letter-spacing: -0.04em; }
.res-conf { font-size: 0.76rem; opacity: 0.55; margin-top: 5px; }
.res-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 18px; border-radius: 100px; font-size: 0.76rem;
  font-weight: 700; letter-spacing: 0.04em; border: 1px solid; margin-top: 14px;
}

/* ── Dual bar ── */
.dual-bar  { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--r); padding: 16px 18px; margin-bottom: 14px; }
.db-labels { display: flex; justify-content: space-between; margin-bottom: 7px; }
.db-lbl    { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; }
.db-track  { height: 9px; border-radius: 9px; overflow: hidden; display: flex; background: var(--surface3); }
.db-g      { background: var(--green); }
.db-r      { background: var(--red); }
.db-vals   { display: flex; justify-content: space-between; margin-top: 5px; }
.db-v      { font-size: 0.7rem; font-family: var(--mono); color: var(--text2); }

/* ── Insight ── */
.insight {
  background: var(--surface2); border: 1px solid var(--border);
  border-left: 3px solid var(--blue);
  border-radius: 0 var(--r) var(--r) 0;
  padding: 12px 14px; font-size: 0.79rem; line-height: 1.65; color: var(--text2);
}
.insight strong { color: var(--text); }

/* ── Risk rows ── */
.risk-block { background: var(--surface); border: 1px solid var(--border); border-radius: var(--rxl); padding: 20px 22px; margin-bottom: 14px; }
.risk-row  { display: flex; align-items: center; gap: 11px; padding: 10px 0; border-bottom: 1px solid var(--border); }
.risk-row:last-child { border-bottom: none; }
.ri-em  { font-size: 14px; width: 20px; flex-shrink: 0; }
.ri-txt { flex: 1; }
.ri-nm  { font-size: 0.8rem; font-weight: 500; }
.ri-sb  { font-size: 0.67rem; color: var(--text3); }
.ri-bg  { width: 90px; height: 5px; background: var(--surface3); border-radius: 5px; overflow: hidden; flex-shrink: 0; }
.ri-fill{ height: 100%; border-radius: 5px; }
.ri-pct { font-size: 0.72rem; font-family: var(--mono); font-weight: 600; width: 34px; text-align: right; flex-shrink: 0; }

/* ── Feature table ── */
.feat-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--rxl); padding: 20px 22px; }
.ft { width: 100%; border-collapse: collapse; font-size: 0.79rem; }
.ft th { text-align: left; padding: 7px 10px; font-size: 0.64rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text3); border-bottom: 1px solid var(--border); }
.ft td { padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: middle; }
.ft tr:last-child td { border-bottom: none; }
.ft tr:hover td { background: var(--surface2); }
.fm  { font-family: var(--mono); font-size: 0.78rem; }
.fb  { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.64rem; font-weight: 600; }
.fn  { background: rgba(59,130,246,0.12); color: var(--bluel); }
.fa  { background: rgba(245,158,11,0.12);  color: #fcd34d; }
.fz  { background: rgba(71,85,105,0.12);   color: #94a3b8; }

/* ── Rec cards ── */
.rec-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; margin-top: 22px; }
.rec { background: var(--surface); border: 1px solid; border-top-width: 3px; border-radius: var(--rl); padding: 18px; transition: transform 0.15s; }
.rec:hover { transform: translateY(-3px); }
.rec-ico  { font-size: 1.3rem; margin-bottom: 9px; }
.rec-ttl  { font-weight: 700; font-size: 0.83rem; color: var(--text); margin-bottom: 7px; }
.rec-desc { font-size: 0.76rem; color: var(--text2); line-height: 1.6; }

/* ── Footer ── */
.footer {
  margin-top: 38px; padding: 18px 0;
  border-top: 1px solid var(--border);
  display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 8px;
}
.footer-brand {
  font-weight: 700; font-size: 0.84rem;
  background: linear-gradient(90deg, #fff, var(--bluel));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.footer-meta { font-size: 0.68rem; color: #334155; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header    { visibility: hidden !important; }
[data-testid="stToolbar"]    { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ═══════ RESPONSIVE ═══════ */
@media (max-width: 1200px) {
  .kpi-row  { grid-template-columns: repeat(2, 1fr); }
  .rec-grid { grid-template-columns: repeat(2, 1fr); }
  .page     { padding: 20px 24px 48px; }
}
@media (max-width: 900px) {
  .navbar  { padding: 11px 18px; }
  .page    { padding: 16px 16px 40px; }
  .nav-right { gap: 12px; }
}
@media (max-width: 700px) {
  .kpi-row  { grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .kpi-val  { font-size: 1.4rem; }
  .rec-grid { grid-template-columns: 1fr; }
  .nav-right { display: none; }
  .page    { padding: 12px 10px 36px; }
}
@media (max-width: 480px) {
  .kpi-row { gap: 8px; }
  .kpi     { padding: 12px 14px; }
  .kpi-val { font-size: 1.2rem; }
  .res-pct { font-size: 3rem; }
  .input-card, .result-card, .risk-block, .feat-wrap { padding: 16px 14px; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load("rf.joblib")
    except FileNotFoundError:
        return None

model = load_model()

FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "InternetService_Fiber optic", "InternetService_No",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

def build_X(tenure, monthly, total, internet, contract, payment):
    return np.array([[
        tenure, monthly, total,
        int(internet == "Fiber Optic"),
        int(internet == "No Internet"),
        int(contract == "One Year"),
        int(contract == "Two Year"),
        int(payment == "Credit Card (Automatic)"),
        int(payment == "Electronic Check"),
        int(payment == "Mailed Check"),
    ]])


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE  (for option-button selections)
# ─────────────────────────────────────────────────────────────────────────────
if "internet"  not in st.session_state: st.session_state.internet  = "Fiber Optic"
if "contract"  not in st.session_state: st.session_state.contract  = "Month-to-Month"
if "payment"   not in st.session_state: st.session_state.payment   = "Electronic Check"

def set_internet(v):  st.session_state.internet = v
def set_contract(v):  st.session_state.contract = v
def set_payment(v):   st.session_state.payment  = v


# ─────────────────────────────────────────────────────────────────────────────
#  NAVBAR  (placeholder — rendered after we know the prediction)
# ─────────────────────────────────────────────────────────────────────────────
nav_slot = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE OPEN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='page'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TOP SECTION:  INPUT  (left)  |  RESULTS  (right)
# ─────────────────────────────────────────────────────────────────────────────
col_in, col_res = st.columns([1, 1.5], gap="large")

# ════════════════════════════════════════════
#  INPUT COLUMN
# ════════════════════════════════════════════
with col_in:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>📋 Customer Profile</div>", unsafe_allow_html=True)

    # ── Tenure ──
    st.markdown("<div class='field-lbl' style='margin-top:0'>📅 Tenure (months)</div>", unsafe_allow_html=True)
    tenure = st.slider("tenure_sl", 0, 72, 24, label_visibility="collapsed")

    # ── Monthly Charges ──
    st.markdown("<div class='field-lbl'>💳 Monthly Charges ($)</div>", unsafe_allow_html=True)
    monthly = st.slider("monthly_sl", 18, 120, 65, label_visibility="collapsed")

    # ── Total Charges ──
    st.markdown("<div class='field-lbl'>💰 Total Charges ($)</div>", unsafe_allow_html=True)
    total = st.number_input("total_ni", min_value=0.0,
                             value=round(float(tenure * monthly), 2),
                             step=50.0, format="%.2f",
                             label_visibility="collapsed")

    # ── Internet Service ──
    st.markdown("<div class='field-lbl'>🌐 Internet Service</div>", unsafe_allow_html=True)
    i_opts = ["DSL", "Fiber Optic", "No Internet"]
    i_cols = st.columns(len(i_opts))
    for idx, opt in enumerate(i_opts):
        with i_cols[idx]:
            active = "opt-active" if st.session_state.internet == opt else ""
            if st.button(opt, key=f"inet_{opt}",
                         use_container_width=True):
                set_internet(opt)
                st.rerun()

    # ── Contract Type ──
    st.markdown("<div class='field-lbl'>📄 Contract Type</div>", unsafe_allow_html=True)
    c_opts = ["Month-to-Month", "One Year", "Two Year"]
    c_cols = st.columns(len(c_opts))
    for idx, opt in enumerate(c_opts):
        with c_cols[idx]:
            if st.button(opt, key=f"con_{opt}", use_container_width=True):
                set_contract(opt)
                st.rerun()

    # ── Payment Method ──
    st.markdown("<div class='field-lbl'>🏦 Payment Method</div>", unsafe_allow_html=True)
    p_opts = ["Bank Transfer (Auto)", "Credit Card (Automatic)",
              "Electronic Check", "Mailed Check"]
    p_cols = st.columns(2)
    for idx, opt in enumerate(p_opts):
        with p_cols[idx % 2]:
            if st.button(opt, key=f"pay_{opt}", use_container_width=True):
                set_payment(opt)
                st.rerun()

    # ── Selection indicators ──
    st.markdown(f"""
    <div style='margin-top:16px;background:var(--surface2);border:1px solid var(--border);
                border-radius:var(--r);padding:12px 14px;font-size:0.77rem;
                display:flex;flex-direction:column;gap:5px'>
      <div style='display:flex;justify-content:space-between'>
        <span style='color:var(--text3)'>Internet</span>
        <span style='color:var(--bluel);font-weight:600'>{st.session_state.internet}</span>
      </div>
      <div style='display:flex;justify-content:space-between'>
        <span style='color:var(--text3)'>Contract</span>
        <span style='color:var(--bluel);font-weight:600'>{st.session_state.contract}</span>
      </div>
      <div style='display:flex;justify-content:space-between'>
        <span style='color:var(--text3)'>Payment</span>
        <span style='color:var(--bluel);font-weight:600'>{st.session_state.payment.split("(")[0].strip()}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Analyze This Customer", use_container_width=True)

    if model is None:
        st.markdown("""
        <div style='margin-top:12px;padding:10px 12px;background:rgba(245,158,11,0.08);
                    border:1px solid rgba(245,158,11,0.2);border-radius:var(--r);
                    font-size:0.73rem;color:#a16207'>
          ⚠️ <strong>rf.joblib not found</strong> — running in demo mode with estimated probabilities.
        </div>
        """, unsafe_allow_html=True)

    # Model info box
    st.markdown("""
    <div style='margin-top:12px;padding:12px 14px;
                background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.12);
                border-radius:var(--r);font-size:0.72rem;color:#64748b;line-height:1.7'>
      <span style='color:#94a3b8;font-weight:600'>Model</span> · XGBoost Classifier &nbsp;|&nbsp;
      <span style='color:#94a3b8;font-weight:600'>Train</span> · 89.6% &nbsp;|&nbsp;
      <span style='color:#94a3b8;font-weight:600'>Test</span> · 76.2%
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # /input-card


# ════════════════════════════════════════════
#  COMPUTE PREDICTION
# ════════════════════════════════════════════
internet = st.session_state.internet
contract = st.session_state.contract
payment  = st.session_state.payment

X = build_X(tenure, monthly, total, internet, contract, payment)

if model is not None:
    churn_prob = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") \
                 else float(model.predict(X)[0])
else:
    base = 0.27
    if contract == "Month-to-Month": base += 0.22
    elif contract == "One Year":     base += 0.08
    if internet == "Fiber Optic":    base += 0.12
    if payment  == "Electronic Check": base += 0.10
    base -= (tenure / 72) * 0.18
    base += (monthly - 65) / 120 * 0.08
    churn_prob = float(np.clip(base, 0.02, 0.97))

retain_prob = 1.0 - churn_prob
is_churn    = churn_prob >= 0.5
risk_level  = "High" if churn_prob > 0.65 else ("Medium" if churn_prob > 0.35 else "Low")

GRN = "#10b981"; RED = "#ef4444"; AMB = "#f59e0b"; BLU = "#3b82f6"
vc   = RED if is_churn else GRN
rc   = {"High": RED, "Medium": AMB, "Low": GRN}[risk_level]


# ════════════════════════════════════════════
#  RESULTS COLUMN
# ════════════════════════════════════════════
with col_res:

    # KPI row (4 cards inside results column)
    c_map = {"Month-to-Month": (RED, "⚠️ High Risk"), "One Year": (AMB, "✅ Moderate"), "Two Year": (GRN, "🔒 Committed")}
    cc, cl = c_map[contract]
    st.markdown(f"""
    <div class='kpi-row'>
      <div class='kpi'>
        <div class='kpi-top'>
          <div class='kpi-ico' style='background:rgba(59,130,246,0.1)'>📅</div>
          <div class='kpi-badge' style='background:rgba(59,130,246,0.1);color:#60a5fa'>{tenure}mo</div>
        </div>
        <div class='kpi-lbl'>Tenure</div>
        <div class='kpi-val' style='color:#60a5fa'>{tenure}</div>
        <div class='kpi-sub'>months active</div>
      </div>
      <div class='kpi'>
        <div class='kpi-top'>
          <div class='kpi-ico' style='background:rgba(245,158,11,0.1)'>💳</div>
          <div class='kpi-badge' style='background:rgba(245,158,11,0.1);color:#fcd34d'>/mo</div>
        </div>
        <div class='kpi-lbl'>Monthly</div>
        <div class='kpi-val' style='color:#fcd34d'>${monthly}</div>
        <div class='kpi-sub'>per cycle</div>
      </div>
      <div class='kpi'>
        <div class='kpi-top'>
          <div class='kpi-ico' style='background:rgba(16,185,129,0.1)'>💰</div>
          <div class='kpi-badge' style='background:rgba(16,185,129,0.1);color:#34d399'>LTV</div>
        </div>
        <div class='kpi-lbl'>Total Revenue</div>
        <div class='kpi-val' style='color:#34d399'>${int(total):,}</div>
        <div class='kpi-sub'>lifetime value</div>
      </div>
      <div class='kpi'>
        <div class='kpi-top'>
          <div class='kpi-ico' style='background:rgba(239,68,68,0.08)'>📄</div>
          <div class='kpi-badge' style='color:{cc};background:rgba(0,0,0,0.15)'>{cl}</div>
        </div>
        <div class='kpi-lbl'>Contract</div>
        <div class='kpi-val' style='color:{cc};font-size:0.88rem;letter-spacing:0;padding-top:5px'>{contract}</div>
        <div class='kpi-sub'>{internet}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge + Verdict side by side
    g1, g2 = st.columns([1.1, 1], gap="small")

    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(churn_prob * 100, 1),
            number=dict(suffix="%", font=dict(size=42, color=vc, family="JetBrains Mono"), valueformat=".1f"),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(255,255,255,0.07)",
                          tickfont=dict(color="#475569", size=10), nticks=6),
                bar=dict(color=vc, thickness=0.27),
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
                steps=[
                    dict(range=[0, 35],  color="rgba(16,185,129,0.08)"),
                    dict(range=[35, 65], color="rgba(245,158,11,0.08)"),
                    dict(range=[65,100], color="rgba(239,68,68,0.08)"),
                ],
                threshold=dict(line=dict(color="rgba(255,255,255,0.3)", width=2),
                               thickness=0.8, value=50),
            ),
            title=dict(text=f"<span style='color:#64748b;font-size:11px'>RISK SCORE · {risk_level.upper()}</span>",
                       font=dict(size=11)),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, margin=dict(l=16, r=16, t=28, b=0),
            font=dict(color="#e2e8f0", family="Plus Jakarta Sans"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with g2:
        pill  = "⚠️ CHURN PREDICTED" if is_churn else "✅ RETENTION LIKELY"
        advice = ("Immediate intervention recommended. Consider proactive outreach, "
                  "personalised discount, or contract upgrade offer.") \
                 if is_churn else \
                 ("Customer appears stable. Maintain service quality and monitor "
                  "billing sensitivity for long-term retention.")

        st.markdown(f"""
        <div class='result-card' style='color:{vc};border-color:{vc}33;
             background:linear-gradient(135deg,{vc}08,{vc}03)'>
          <div class='res-lbl'>Churn Probability</div>
          <div class='res-pct' style='color:{vc}'>{churn_prob:.1%}</div>
          <div class='res-conf'>confidence · {max(churn_prob,retain_prob):.1%}</div>
          <div class='res-pill' style='color:{vc};border-color:{vc}44;background:{vc}11'>{pill}</div>
        </div>
        <div class='dual-bar'>
          <div class='db-labels'>
            <span class='db-lbl' style='color:{GRN}'>Retain</span>
            <span class='db-lbl' style='color:{RED}'>Churn</span>
          </div>
          <div class='db-track'>
            <div class='db-g' style='width:{retain_prob*100:.1f}%'></div>
            <div class='db-r' style='width:{churn_prob*100:.1f}%'></div>
          </div>
          <div class='db-vals'>
            <span class='db-v' style='color:{GRN}'>{retain_prob:.1%}</span>
            <span class='db-v' style='color:{RED}'>{churn_prob:.1%}</span>
          </div>
        </div>
        <div class='insight'>💡 <strong>Insight:</strong> {advice}</div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  BOTTOM ROW:  RISK FACTORS  |  FEATURE VECTOR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
b1, b2 = st.columns([1, 1.3], gap="large")

with b1:
    def rc_col(v): return RED if v > 0.65 else (AMB if v > 0.35 else GRN)
    c_risk  = {"Month-to-Month": 0.88, "One Year": 0.42, "Two Year": 0.12}[contract]
    t_risk  = float(np.clip(1 - tenure / 72, 0, 1))
    m_risk  = (monthly - 18) / (120 - 18)
    pm_risk = {"Electronic Check": 0.80, "Mailed Check": 0.48,
               "Credit Card (Automatic)": 0.20, "Bank Transfer (Auto)": 0.16}.get(payment, 0.4)
    i_risk  = {"Fiber Optic": 0.70, "DSL": 0.36, "No Internet": 0.20}[internet]

    facs = [
        ("📄", "Contract type",   contract,                      c_risk),
        ("📅", "Customer tenure", f"{tenure} months",            t_risk),
        ("💸", "Monthly charges", f"${monthly}/mo",               m_risk),
        ("🏦", "Payment method",  payment.split("(")[0].strip(), pm_risk),
        ("🌐", "Internet service",internet,                       i_risk),
    ]
    rows = "".join(f"""
      <div class='risk-row'>
        <span class='ri-em'>{em}</span>
        <div class='ri-txt'>
          <div class='ri-nm'>{nm}</div>
          <div class='ri-sb'>{sb}</div>
        </div>
        <div class='ri-bg'><div class='ri-fill' style='width:{int(v*100)}%;background:{rc_col(v)}'></div></div>
        <span class='ri-pct' style='color:{rc_col(v)}'>{v:.0%}</span>
      </div>""" for em, nm, sb, v in facs)

    st.markdown(f"""
    <div class='risk-block'>
      <div class='card-title'>🔍 Risk Factor Analysis</div>
      {rows}
    </div>
    """, unsafe_allow_html=True)

with b2:
    X_flat = X[0].tolist()
    short  = ["tenure","MonthlyCharges","TotalCharges",
              "InternetSvc_FiberOptic","InternetSvc_No",
              "Contract_OneYear","Contract_TwoYear",
              "Pay_CreditCard","Pay_ElecCheck","Pay_MailedCheck"]
    types  = ["Numeric"]*3 + ["Encoded"]*7
    dvals  = [str(int(tenure)), str(int(monthly)), f"{total:.0f}"] + \
             [str(int(v)) for v in X_flat[3:]]

    rows_html = ""
    for i,(f,v,t) in enumerate(zip(short, dvals, types)):
        b = f"<span class='fb fn'>Numeric</span>" if t=="Numeric" \
            else (f"<span class='fb fa'>ACTIVE</span>" if v=="1"
                  else f"<span class='fb fz'>0</span>")
        rows_html += f"""<tr>
          <td style='color:#475569;font-family:var(--mono);font-size:0.72rem'>{i}</td>
          <td style='color:#cbd5e1'>{f}</td>
          <td><span class='fm'>{v}</span></td>
          <td>{b}</td></tr>"""

    st.markdown(f"""
    <div class='feat-wrap'>
      <div class='card-title'>
        🗂️ Model Input Vector
        <span style='margin-left:auto;font-size:0.63rem;font-weight:400;
                     color:var(--text3);letter-spacing:0'>10 features</span>
      </div>
      <div style='overflow-x:auto'>
        <table class='ft'>
          <thead><tr><th>#</th><th>Feature</th><th>Value</th><th>Status</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  RECOMMENDATION CARDS
# ─────────────────────────────────────────────────────────────────────────────
if is_churn:
    recs = [
        (RED,      "🎁", "Loyalty Offer",       "Offer 15–20% discount for 3 billing cycles to incentivise contract renewal and reduce churn likelihood."),
        (AMB,      "📞", "Proactive Outreach",   "Schedule a customer success call within 48 hours to understand pain points and offer personalised solutions."),
        ("#6366f1","🔒", "Contract Upgrade",     "Present a discounted annual plan. Long-term contracts statistically reduce churn probability by ~60%."),
        ("#06b6d4","⭐", "Service Review",       f"{'Assess Fiber Optic plan satisfaction' if internet=='Fiber Optic' else 'Upgrade internet tier'} to improve perceived value."),
    ]
else:
    recs = [
        (GRN,      "✅", "Maintain Quality",     "Customer satisfaction appears high. Focus on consistent service and proactive communication."),
        (BLU,      "📈", "Upsell Opportunity",   "Stable customers are ideal for premium tier upsells — consider a personalised targeted offer."),
        ("#6366f1","🔄", "Renewal Strategy",     f"Send a timely renewal reminder for the {'monthly' if contract=='Month-to-Month' else 'annual'} plan with loyalty perks."),
        (AMB,      "📊", "Usage Monitoring",     "Set automated alerts for engagement drop-off to catch disengagement signals before they escalate."),
    ]

rec_cards = "".join(f"""
  <div class='rec' style='border-color:{col}33;border-top-color:{col}'>
    <div class='rec-ico'>{em}</div>
    <div class='rec-ttl'>{ttl}</div>
    <div class='rec-desc'>{dsc}</div>
  </div>""" for col, em, ttl, dsc in recs)

st.markdown(f"<div class='rec-grid'>{rec_cards}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER + NAVBAR (fill nav_slot now we have the data)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='footer'>
  <div class='footer-brand'>🔮 ChurnIQ · Customer Intelligence Platform</div>
  <div class='footer-meta'>
    XGBoost · 10 features · Train 89.6% · Test 76.2% · {"Demo mode" if model is None else "Live model"}
  </div>
</div>
</div>
""", unsafe_allow_html=True)  # closes .page

nav_slot.markdown(f"""
<div class='navbar'>
  <div class='nav-left'>
    <div class='nav-icon'>🔮</div>
    <div>
      <div class='nav-title'>ChurnIQ</div>
    </div>
    <div class='nav-badge'>XGBoost</div>
  </div>
  <div class='nav-right'>
    <div class='nav-stat'><span>Churn Risk</span><strong style='color:{vc}'>{churn_prob:.1%}</strong></div>
    <div class='nav-sep'></div>
    <div class='nav-stat'><span>Risk Level</span><strong style='color:{rc}'>{risk_level}</strong></div>
    <div class='nav-sep'></div>
    <div class='nav-stat'><span>Retention</span><strong style='color:{GRN}'>{retain_prob:.1%}</strong></div>
    <div class='nav-sep'></div>
    <div class='nav-stat'><span>Status</span><strong>{"Live" if model else "Demo"}</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)