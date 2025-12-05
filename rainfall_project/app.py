# app.py —  Weather Dashboard (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import altair as alt
import os

st.set_page_config(page_title="Rainfall Dashboard", layout="wide", initial_sidebar_state="collapsed")

# -------------------------
#  CSS / Visual Design
# -------------------------
st.markdown(
    """
<style>
/* Global reset */
* { box-sizing: border-box; }

/* Animated background */
.stApp {
  background: linear-gradient(135deg,#667eea 0%,#764ba2 25%,#f093fb 50%,#f5576c 75%,#ffd89b 100%);
  background-size: 400% 400%;
  animation: gradientBG 15s ease infinite;
  min-height: 100vh;
}

@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* Main container (glass) */
.main-container{
  background: rgba(255,255,255,0.95);
  backdrop-filter: blur(12px);
  border-radius: 20px;
  padding: 28px;
  margin: 24px auto;
  max-width: 1320px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.18);
  border: 1px solid rgba(255,255,255,0.45);
}

/* Header */
.header { text-align:center; margin-bottom:20px; }
.title {
  font-size:44px; font-weight:900;
  background: linear-gradient(90deg,#667eea,#764ba2);
  -webkit-background-clip:text; color:transparent;
}

/* Nav tabs */
.nav-tabs { display:flex; gap:12px; justify-content:center; margin:22px 0; flex-wrap:wrap; }
.tab-btn {
  padding:10px 22px; border-radius:12px; font-weight:700; cursor:pointer; border:none;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.tab-active { background: linear-gradient(90deg,#667eea,#764ba2); color:white; }
.tab-inactive { background: white; color:#667eea; border:2px solid rgba(102,126,234,0.12); }

/* Cards */
.card { background:white; padding:18px; border-radius:14px; box-shadow: 0 10px 30px rgba(0,0,0,0.06); }
.card h3{ margin:0 0 10px 0; color:#333; font-size:18px; }

/* Result / big card */
.result-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,249,250,0.95));
  padding:28px; border-radius:16px; text-align:center; box-shadow: 0 18px 45px rgba(0,0,0,0.12);
}

/* Confidence meter */
.confidence-meter { width:100%; height:36px; border-radius:18px; background:#e9ecef; overflow:hidden; border:1px solid #e0e6ea; }
.confidence-fill { height:100%; transition: width 1.2s ease-out; }

/* Minor tweaks */
.footer { text-align:center; color:#6c757d; margin-top:20px; font-size:13px; }
@media (max-width:900px){
  .title{ font-size:28px; }
  .main-container{ padding:16px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
#  Utility: Load model & data
# -------------------------
@st.cache_resource
def load_model_and_scaler(model_path="model/model.pkl", scaler_path="model/scaler.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@st.cache_data
def load_dataset(csv_path="data/Rainfall.csv"):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # clean column names and types
    df.columns = df.columns.str.strip()
    # if there's a rainfall yes/no column, standardize
    if "rainfall" in df.columns.str.lower().tolist():
        # find exact column name if case-varied
        col = [c for c in df.columns if c.strip().lower() == "rainfall"][0]
        df.rename(columns={col: "rainfall"}, inplace=True)
    # map yes/no to 1/0 if present
    if "rainfall" in df.columns:
        df["rainfall"] = df["rainfall"].map({"yes": 1, "no": 0}).astype(float)
    # numeric conversion
    for c in df.columns:
        if c != "rainfall":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # fill numeric missing with column mean
    df = df.fillna(df.mean(numeric_only=True))
    return df

model, scaler = load_model_and_scaler()
df = load_dataset()

if df.empty:
    st.warning("Dataset not found or empty. Put `Rainfall.csv` into `data/` folder.")
if model is None or scaler is None:
    st.warning("Model or scaler not found. Put `model.pkl` and `scaler.pkl` into `model/` folder.")

# Halt if critical missing
if df.empty or model is None or scaler is None:
    st.markdown("<div class='main-container'><h2 style='text-align:center'>App needs model and data to run</h2></div>", unsafe_allow_html=True)
    st.stop()

# feature columns (all except target)
feature_cols = [c for c in df.columns if c != "rainfall"]
mean_values = df[feature_cols].mean()
min_values = df[feature_cols].min()
max_values = df[feature_cols].max()

# -------------------------
#  Main container & header
# -------------------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='header'><div class='title'>Rainfall Prediction</div></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#5b6b7a; margin-bottom:12px;'>Interactive weather parameter controls, analytics and confident rainfall predictions.</p>", unsafe_allow_html=True)

# -------------------------
#  Navigation tabs
# -------------------------
tabs = ["Home", "Input", "Analytics", "About"]
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

tab_cols = st.columns(len(tabs))
for i, t in enumerate(tabs):
    cls = "tab-btn tab-active" if st.session_state.active_tab == t else "tab-btn tab-inactive"
    if tab_cols[i].button(t, key=f"tab_{i}", help=f"Open {t}"):
        st.session_state.active_tab = t

st.markdown("<hr style='margin-top:14px; opacity:0.08'/>", unsafe_allow_html=True)

# -------------------------
#  Home (Hero + Quick Predict + Recent Stats)
# -------------------------
if st.session_state.active_tab == "Home":
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("<div class='card'><h3>🌥️ Quick Predict</h3><p style='color:#5b6b7a; margin-top:6px;'>Adjust a few key parameters below to get an instant prediction</p></div>", unsafe_allow_html=True)
        st.write("")  # spacing

        # quick sliders (choose a few representative features)
        quick_features = []
        # heuristics: prefer temp, humidity, wind, pressure if present
        for name in ["maxtemp", "temparature", "mintemp", "humidity", "windspeed", "pressure"]:
            for c in feature_cols:
                if name in c.lower() and c not in quick_features:
                    quick_features.append(c)
        # fallback if none found
        if not quick_features:
            quick_features = feature_cols[:4]

        q_cols = st.columns(len(quick_features))
        for idx, f in enumerate(quick_features):
            with q_cols[idx]:
                if f not in st.session_state:
                    st.session_state[f] = float(mean_values[f])
                st.session_state[f] = st.slider(
                    label=f.replace("_", " ").title(),
                    min_value=float(min_values[f]),
                    max_value=float(max_values[f]),
                    value=float(st.session_state[f]),
                    step=(max_values[f] - min_values[f]) / 100 if max_values[f] != min_values[f] else 1.0,
                    key=f"quick_{f}"
                )

        st.write("")  # spacing
        if st.button("🔮 Quick Predict Now", width="content"):

            # collect inputs
            inputs = []
            for c in feature_cols:
                # if user changed quick inputs, use them, otherwise default session or mean
                if c in st.session_state and str(c).startswith("quick_") is False:
                    # some quick keys are prefixed; check both
                    pass
                # choose value precedence: quick slider -> session_state value -> mean
                if f"quick_{c}" in st.session_state:
                    val = st.session_state[f"quick_{c}"]
                elif c in st.session_state:
                    val = st.session_state[c]
                else:
                    val = float(mean_values[c])
                inputs.append(val)

            X = pd.DataFrame([inputs], columns=feature_cols)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            # predict_proba may not be available for all models; handle gracefully
            try:
                proba = model.predict_proba(X_scaled)[0][1]
            except Exception:
                # fallback: use decision_function if exists, else 0.5
                proba = 0.5
            # show result block
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown("<div style='font-size:60px; margin-bottom:6px;'>🌧️☔</div>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='color:#4673d8;'>RAIN EXPECTED</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color:#667eea;'>Probability: {proba:.1%}</h4>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size:60px; margin-bottom:6px;'>☀️🌤️</div>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='color:#d08b1f;'>NO RAIN</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color:#f39c12;'>Probability of rain: {proba:.1%}</h4>", unsafe_allow_html=True)

                # ---------------- Confidence meter (correct logic) ----------------
                if prediction == 1:
                    confidence = proba
                else:
                    confidence = 1 - proba
                confidence_pct = confidence * 100
                # gradient/color by strength
                if confidence_pct < 40:
                    grad = "#f26b6b"  # red-ish
                elif confidence_pct < 70:
                    grad = "#f39c12"  # orange
                else:
                    grad = "#2ecc71"  # green
                st.markdown(f"""
                    <div style="margin-top:12px;">
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width:{confidence_pct}%; background:{grad}; border-radius:18px;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:8px; color:#6c757d; font-size:13px;">
                            <div>Low</div><div>Medium</div><div>High</div>
                        </div>
                        <div style="text-align:center; margin-top:10px; font-weight:700;">Confidence: {confidence_pct:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # small stats card + quick chart
        st.markdown("<div class='card'><h3>Dataset Snapshot</h3></div>", unsafe_allow_html=True)
        total = len(df)
        rain_count = int(df['rainfall'].sum()) if 'rainfall' in df.columns else 0
        no_rain_count = total - rain_count
        st.metric("Records", f"{total}")
        st.metric("Rain days", f"{rain_count}")
        # mini pie chart
        pie_df = pd.DataFrame({
            "label": ["No Rain", "Rain"],
            "count": [no_rain_count, rain_count]
        })
        pie = alt.Chart(pie_df).mark_arc(innerRadius=30).encode(
            theta="count",
            color=alt.Color("label", scale=alt.Scale(range=["#ffd89b","#667eea"])),
            tooltip=["label","count"]
        ).properties(height=220)
        st.altair_chart(pie, width="stretch")


# -------------------------
#  Input (Full controls)
# -------------------------
elif st.session_state.active_tab == "Input":
    st.markdown("<div class='card'><h3>⚙️ Input Parameters</h3><p style='color:#5b6b7a'>Adjust detailed weather parameters used for prediction.</p></div>", unsafe_allow_html=True)
    st.write("")
    # categorize features
    categories = {
        "Temperature": [c for c in feature_cols if "temp" in c.lower()],
        "Humidity": [c for c in feature_cols if "humid" in c.lower()],
        "Wind": [c for c in feature_cols if "wind" in c.lower()],
        "Pressure": [c for c in feature_cols if "press" in c.lower()],
        "Other": [c for c in feature_cols if not any(k in c.lower() for k in ["temp","humid","wind","press"])]
    }
    for cat, cols in categories.items():
        if not cols:
            continue
        with st.expander(f"📊 {cat} ({len(cols)})", expanded=False):
            cols_layout = st.columns(2)
            for i, c in enumerate(cols):
                col_slot = cols_layout[i % 2]
                with col_slot:
                    if c not in st.session_state:
                        st.session_state[c] = float(mean_values[c])
                    st.session_state[c] = st.slider(
                        label=c.replace("_"," ").title(),
                        min_value=float(min_values[c]),
                        max_value=float(max_values[c]),
                        value=float(st.session_state[c]),
                        step=(max_values[c] - min_values[c]) / 200 if max_values[c] != min_values[c] else 0.1,
                        key=f"in_{c}"
                    )

    st.write("")
    # st.markdown("<div style='text-align:center; margin-top:14px;'><button class='tab-btn tab-active' id='run_full_predict'>Run full prediction</button></div>", unsafe_allow_html=True)
    # The button above is decorative (JS-less). Provide the real Streamlit button:
    if st.button("Run full prediction", key="run_full_pred"):
        inputs = [float(st.session_state[c]) for c in feature_cols]
        X = pd.DataFrame([inputs], columns=feature_cols)
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        try:
            p = model.predict_proba(Xs)[0][1]
        except Exception:
            p = 0.5
        st.success(f"Prediction: {'Rain' if pred==1 else 'No Rain'} — Probability of rain: {p:.1%}")

# -------------------------
#  Analytics
# -------------------------
elif st.session_state.active_tab == "Analytics":
    st.markdown("<div class='card'><h3>📈 Analytics & Insights</h3></div>", unsafe_allow_html=True)
    # correlation heatmap style table (approx)
    corrs = df[feature_cols + (['rainfall'] if 'rainfall' in df.columns else [])].corr()
    # show top correlations with rainfall
    if 'rainfall' in corrs.columns:
        corr_with_rain = corrs['rainfall'].abs().sort_values(ascending=False).drop('rainfall')
        top = corr_with_rain.head(8).reset_index()
        top.columns = ['Feature', 'AbsCorrelation']
        st.table(top.style.format({"AbsCorrelation": "{:.3f}"}))

    st.markdown("### Feature distributions")
    sel = st.selectbox("Select feature", feature_cols, index=0)
    hist = alt.Chart(df).mark_bar().encode(
        alt.X(sel, bin=alt.Bin(maxbins=40)),
        y='count()',
        color=alt.Color('rainfall:N', scale=alt.Scale(range=["#ffd89b","#667eea"]), sort=['0','1'])
    ).properties(height=350)
    st.altair_chart(hist, width="stretch")


# -------------------------
#  About
# -------------------------
elif st.session_state.active_tab == "About":
    st.markdown("<div class='card'><h3>ℹ️ About This App</h3></div>", unsafe_allow_html=True)
    st.markdown("""
- **Premium Weather Dashboard** — interactive Streamlit app for rainfall prediction  
- **Model**: your trained classifier (RandomForest or similar) loaded from `model/model.pkl`  
- **Data**: `data/Rainfall.csv` used for feature defaults and analytics  
- **Confidence meter** uses probability (or 1 - probability) depending on predicted class.
""")

# -------------------------
# Footer + close container
# -------------------------
st.markdown(f"""
<div class="footer">
    🌦️ Rainfall Prediction • Built {datetime.now().year} • Data records: {len(df)} 
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
