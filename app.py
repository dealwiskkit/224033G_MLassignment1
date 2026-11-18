import calendar
import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Colombo Weather | Ensemble Dashboard",
                   page_icon="🌦️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Keep theme controls; hide viewer actions that are not relevant to users.
   Streamlit's minimal toolbar mode hides unwanted menu actions natively. */
[data-testid="stMainMenuItem-print"],
[data-testid="stMainMenuItem-recordScreencast"],
footer {display: none !important;}
header[data-testid="stHeader"] {background: transparent;}

/* Overall dashboard spacing and readability */
.block-container {padding-top: 1rem; padding-bottom: 2.2rem; max-width: 1250px;}
h3 {margin-top: 0.65rem; margin-bottom: 0.55rem; font-size: 1.5rem;}
div[data-testid="stCaptionContainer"] p {font-size: 1rem;}

/* Weather-themed masthead */
.hero {
    background: linear-gradient(115deg, #075e9c 0%, #159bd1 55%, #68c9dc 83%, #f5c760 110%);
    border-radius: 18px;
    padding: 1.35rem 1.7rem;
    margin-bottom: 1rem;
    color: white;
    box-shadow: 0 5px 24px rgba(7, 86, 137, 0.16);
}
.hero-title {font-size: 2rem; font-weight: 800; line-height: 1.2;}
.hero-sub {font-size: 1.02rem; line-height: 1.65; opacity: 0.98; margin-top: 0.5rem;}

/* Summary metrics */
div[data-testid="stMetric"] {
    background: rgba(79, 159, 215, 0.08);
    border: 1px solid rgba(35, 147, 213, 0.3);
    border-radius: 14px;
    padding: 0.8rem 1rem;
    box-shadow: 0 2px 9px rgba(9, 92, 143, 0.06);
}
div[data-testid="stMetricLabel"] p {font-size: 0.98rem;}
div[data-testid="stMetricValue"] {font-size: 1.8rem; color: var(--primary-color, #138bc4);}

/* Navigation: real full-width buttons, not small radio dots */
section[data-testid="stSidebar"] .block-container {padding: 1.1rem 0.85rem 1.5rem;}
.nav-brand {
    border-radius: 15px;
    padding: 1rem 0.8rem 0.9rem;
    margin-bottom: 1.35rem;
    background: linear-gradient(125deg, rgba(8, 112, 184, 0.18), rgba(77, 184, 232, 0.07));
    border: 1px solid rgba(77, 184, 232, 0.22);
}
/* Inherit text color from the currently selected Streamlit theme.
   --text-color is not a reliable global CSS token in Streamlit 1.67. */
.nav-brand-title {font-size: 1.35rem; font-weight: 850; line-height: 1.35; color: inherit !important;}
.nav-brand-sub {font-size: 0.84rem; color: inherit !important; opacity: 0.85; margin-top: 0.3rem;}
.nav-group-label {
    margin: 0.1rem 0.15rem 0.55rem;
    font-size: 0.78rem; letter-spacing: 0.12em;
    font-weight: 800; opacity: 0.85; color: inherit !important;
}
section[data-testid="stSidebar"] .stButton button {
    width: 100%;
    min-height: 3.15rem;
    justify-content: flex-start;
    text-align: left;
    border-radius: 11px;
    font-weight: 680;
    font-size: 1.02rem;
    margin-bottom: 0.45rem;
    padding: 0.68rem 0.85rem;
    border: 1px solid rgba(91, 156, 193, 0.32);
    background: rgba(91, 156, 193, 0.11);
    /* Resolve through Streamlit's light/dark theme, not a fixed light-mode hex. */
    color: inherit !important;
    transition: background 150ms ease, transform 150ms ease, border-color 150ms ease;
}
/* Streamlit renders button labels inside p/span elements. Ensure they
   inherit the button's readable color in both themes. */
section[data-testid="stSidebar"] .stButton button[kind="secondary"] p,
section[data-testid="stSidebar"] .stButton button[kind="secondary"] span,
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"] p,
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-secondary"] span {
    color: inherit !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(57, 161, 219, 0.23);
    border-color: #3da7db;
    transform: translateX(2px);
}
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(95deg, #0869ad, #169ed5);
    border-color: #159bd1;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(14, 126, 196, 0.27);
}
section[data-testid="stSidebar"] .stButton button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton button[data-testid="stBaseButton-primary"] p {color: #ffffff !important;}
section[data-testid="stSidebar"] .stButton button:focus-visible {
    outline: 3px solid #ffcd6c;
    outline-offset: 2px;
}
.nav-help {font-size: 0.85rem; line-height: 1.5; opacity: 0.88; color: inherit !important; margin: 0.65rem 0.15rem 0;}

@media (max-width: 680px) {
    .block-container {padding-top: 0.6rem;}
    .hero {padding: 1rem; border-radius: 14px;}
    .hero-title {font-size: 1.55rem;}
    .hero-sub {font-size: 0.9rem;}
    div[data-testid="stMetricValue"] {font-size: 1.45rem;}
}
</style>
""", unsafe_allow_html=True)


# ---------- data ----------
@st.cache_data
def load():
    feats = pd.read_csv("features.csv", index_col=0, parse_dates=True)
    res = pd.read_csv("results.csv")
    preds = pd.read_csv("predictions.csv", index_col=0, parse_dates=True)
    imp = pd.read_csv("importance.csv", index_col=0).iloc[:, 0]
    cv = pd.read_csv("cv_results.csv")
    return feats, res, preds, imp, cv

@st.cache_resource
def load_model():
    return joblib.load("best_model.joblib")

feats, res, preds, imp, cv = load()
model = load_model()
cols = [c for c in feats.columns if c != "target"]
res_sorted = res.sort_values("RMSE").reset_index(drop=True)
best = res_sorted.iloc[0]


def to_long_df(frame):
    out = frame.reset_index()
    return out.rename(columns={out.columns[0]: "date"})


# ---------- sidebar navigation (UI only; the five original pages are unchanged) ----------
NAV_ITEMS = [
    ("📊 Data", "📊  Weather data", "nav_data"),
    ("🏆 Models", "🏆  Compare models", "nav_models"),
    ("🔍 Features", "🔍  Explore features", "nav_features"),
    ("🔮 Predict", "🔮  Predict temperature", "nav_predict"),
    ("ℹ️ About", "ℹ️  About this project", "nav_about"),
]

if "weather_page" not in st.session_state:
    st.session_state["weather_page"] = "📊 Data"


def select_page(destination):
    st.session_state["weather_page"] = destination


with st.sidebar:
    st.markdown(
        '<div class="nav-brand">'
        '<div class="nav-brand-title">🌤️ Colombo Weather</div>'
        '<div class="nav-brand-sub">Ensemble learning dashboard</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-group-label">EXPLORE DASHBOARD</div>', unsafe_allow_html=True)

    for destination, button_label, button_key in NAV_ITEMS:
        st.button(
            button_label,
            key=button_key,
            type="primary" if st.session_state["weather_page"] == destination else "secondary",
            use_container_width=True,
            on_click=select_page,
            args=(destination,),
        )

    st.markdown(
        '<div class="nav-help">Choose a section above to explore weather data, '
        'compare models or predict tomorrow’s temperature.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    # Streamlit minimal mode hides the built-in Rerun item.
    # This provides a user-friendly replacement without changing the model.
    if st.button("↻  Refresh dashboard", key="nav_refresh", use_container_width=True):
        st.rerun()

page = st.session_state["weather_page"]


# ---------- header ----------
st.markdown("""
<div class="hero">
  <div class="hero-title">🌦️ Colombo Weather Dashboard</div>
  <div class="hero-sub">Predicting tomorrow's maximum temperature with ensemble learning:
  Bagging, Boosting, Stacking and a Hybrid ARIMA + RF model · Data: Open-Meteo</div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Days of data", f"{len(feats):,}")
m2.metric("Period", f"{feats.index.min().year} – {feats.index.max().year}")
m3.metric("Best model (test set)", best["Model"].split(" (")[0])
m4.metric("Best RMSE", f"{best['RMSE']:.3f} °C")


# ---------- page 1: data ----------
if page == "📊 Data":
    st.subheader("Daily maximum temperature")
    period = st.radio("Show", ["Last 12 months", "Last 3 years", "All data"],
                      horizontal=True, index=1)
    d = feats.tail(365) if period == "Last 12 months" else feats.tail(365 * 3) if period == "Last 3 years" else feats
    d = to_long_df(d[["tmax"]])
    line = alt.Chart(d).mark_line(color="#f28c28").encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("tmax:Q", title="Max temp (°C)", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("tmax:Q", title="Max temp (°C)")],
    ).properties(height=320, width="container")
    st.altair_chart(line)

    c1, c2 = st.columns(2, gap="large")
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    monthly = feats.groupby("month")[["tmax", "rain"]].mean().reset_index()
    monthly["Month"] = monthly["month"].apply(lambda m: calendar.month_abbr[m])

    with c1:
        st.subheader("Average max temp by month")
        st.altair_chart(alt.Chart(monthly).mark_bar(color="#f28c28").encode(
            x=alt.X("Month:N", sort=month_names, title=None),
            y=alt.Y("tmax:Q", title="°C", scale=alt.Scale(zero=False)),
            tooltip=["Month", alt.Tooltip("tmax:Q", format=".1f")],
        ).properties(height=260, width="container"))
    with c2:
        st.subheader("Average daily rainfall by month")
        st.altair_chart(alt.Chart(monthly).mark_bar(color="#2a7fba").encode(
            x=alt.X("Month:N", sort=month_names, title=None),
            y=alt.Y("rain:Q", title="mm per day"),
            tooltip=["Month", alt.Tooltip("rain:Q", format=".1f")],
        ).properties(height=260, width="container"))

    with st.expander("See the latest rows of the dataset"):
        st.dataframe(feats.tail(10))


# ---------- page 2: models ----------
elif page == "🏆 Models":
    st.subheader("Hold-out test results")
    st.caption("Models were trained on the first 80% of days and tested on the most recent 20%. Lower RMSE is better.")
    bar = alt.Chart(res_sorted).mark_bar().encode(
        y=alt.Y("Model:N", sort="x", title=None),
        x=alt.X("RMSE:Q", title="RMSE (°C), lower is better"),
        color=alt.Color("Type:N", title="Approach"),
        tooltip=["Model", "Type", alt.Tooltip("RMSE:Q", format=".3f"),
                 alt.Tooltip("MAE:Q", format=".3f"), alt.Tooltip("R2:Q", format=".3f")],
    ).properties(height=380, width="container")
    st.altair_chart(bar)

    with st.expander("Show full results table"):
        st.dataframe(res_sorted.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "R2": "{:.3f}"}),
                     hide_index=True)

    st.subheader("Is the ranking reliable?")
    st.caption("Time-series cross-validation (5 folds). Differences smaller than the std are within noise.")
    st.dataframe(cv.style.format({"CV RMSE mean": "{:.3f}", "CV RMSE std": "{:.3f}"}), hide_index=True)
    st.info("The top models are statistically close. Ensembles give modest gains on this smooth, seasonal target.")

    st.subheader("Actual vs predicted")
    f1, f2 = st.columns([3, 1])
    options = [c for c in preds.columns if c != "Actual"]
    default = [m for m in ["Stacking (Ridge meta)", "Hybrid ARIMA+RF"] if m in options]
    choice = f1.multiselect("Models to show", options, default=default)
    window = f2.selectbox("Window", ["Last 60 days", "Last 120 days", "Last 365 days"], index=1)
    n = int(window.split()[1])
    long = to_long_df(preds[["Actual"] + choice].tail(n)).melt("date", var_name="Series", value_name="temp")
    lines = alt.Chart(long).mark_line().encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("temp:Q", title="Max temp (°C)", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", title=None),
        strokeWidth=alt.condition(alt.datum.Series == "Actual", alt.value(3), alt.value(1.5)),
        tooltip=["date:T", "Series:N", alt.Tooltip("temp:Q", format=".1f")],
    ).properties(height=340, width="container")
    st.altair_chart(lines)


# ---------- page 3: features ----------
elif page == "🔍 Features":
    st.subheader("What drives the predictions?")
    fi = imp.sort_values(ascending=False).reset_index()
    fi.columns = ["feature", "importance"]
    st.altair_chart(alt.Chart(fi).mark_bar(color="#0b6fb8").encode(
        y=alt.Y("feature:N", sort="-x", title=None),
        x=alt.X("importance:Q", title="Importance (CatBoost)"),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".1f")],
    ).properties(height=380, width="container"))
    st.markdown(
        "- **`tmax`** (today's max temperature) matters most, because temperature changes slowly day to day.\n"
        "- **`tmax_roll7`** (7-day average) captures the recent trend.\n"
        "- **`rain`** ranks third. This is the information ARIMA cannot use, which is why the hybrid beats ARIMA alone."
    )


# ---------- page 4: predict ----------
elif page == "🔮 Predict":
    st.subheader("Predict tomorrow's maximum temperature")
    st.caption("Starting values come from the latest day in the dataset. Change them to try your own scenario.")
    last = feats.iloc[-1]
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.form("predict_form"):
            a, b = st.columns(2)
            date = a.date_input("Today's date", feats.index[-1].date())
            tmax = b.number_input("Today's max temp (°C)", value=float(last["tmax"]), step=0.1)
            tmin = a.number_input("Today's min temp (°C)", value=float(last["tmin"]), step=0.1)
            rain = b.number_input("Today's rainfall (mm)", value=float(last["rain"]), step=0.5)
            wind = a.number_input("Today's max wind (km/h)", value=float(last["wind"]), step=0.5)

            with st.expander("Advanced: recent history"):
                x1, x2 = st.columns(2)
                l1 = x1.number_input("Max temp 1 day ago", value=float(last["tmax_lag1"]), step=0.1)
                l2 = x2.number_input("Max temp 2 days ago", value=float(last["tmax_lag2"]), step=0.1)
                l3 = x1.number_input("Max temp 3 days ago", value=float(last["tmax_lag3"]), step=0.1)
                l7 = x2.number_input("Max temp 7 days ago", value=float(last["tmax_lag7"]), step=0.1)
                roll7 = x1.number_input("Avg max temp, last 7 days", value=float(last["tmax_roll7"]), step=0.1)
                rain7 = x2.number_input("Total rain, last 7 days (mm)", value=float(last["rain_roll7"]), step=0.5)

            submitted = st.form_submit_button("Predict tomorrow", type="primary", width="stretch")

    with right:
        if submitted:
            dt = pd.Timestamp(date)
            row = {"tmax": tmax, "tmin": tmin, "rain": rain, "wind": wind,
                   "tmax_lag1": l1, "tmax_lag2": l2, "tmax_lag3": l3, "tmax_lag7": l7,
                   "tmax_roll7": roll7, "rain_roll7": rain7, "month": dt.month,
                   "doy_sin": np.sin(2 * np.pi * dt.dayofyear / 365),
                   "doy_cos": np.cos(2 * np.pi * dt.dayofyear / 365)}
            p = float(model.predict(pd.DataFrame([row])[cols])[0])
            rmse = float(res.loc[res["Model"] == "CatBoost", "RMSE"].min()) if (res["Model"] == "CatBoost").any() else 0.95
            with st.container(border=True):
                st.metric("Predicted max temp tomorrow", f"{p:.1f} °C", delta=f"{p - tmax:+.1f} °C vs today")
                st.write(f"Likely range: **{p - rmse:.1f} – {p + rmse:.1f} °C**")
                st.caption(f"Range is ±1 typical error (test RMSE ≈ {rmse:.2f} °C). Model: CatBoost.")
        else:
            st.info("Fill in the values on the left and press **Predict tomorrow**.")


# ---------- page 5: about ----------
else:
    st.subheader("About this project")
    st.markdown("""
**Goal:** predict tomorrow's maximum temperature in Colombo.

**Data:** daily weather for Colombo (2015 – 2025) from the Open-Meteo archive API.

**Features:** today's temperature, rainfall and wind, past temperatures (1, 2, 3 and 7 days ago), 7-day rolling averages, and month / day-of-year seasonality.

**Methods compared**
- **Bagging:** Bagging trees, Random Forest
- **Boosting:** Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Voting and Stacking:** RF + LightGBM + CatBoost, with a Ridge meta-learner
- **Hybrid:** ARIMA for the time pattern, plus Random Forest on ARIMA's residuals
- **Baselines:** persistence (tomorrow = today), Linear Regression, a single Decision Tree

**Evaluation:** chronological 80/20 split and 5-fold time-series cross-validation (RMSE, MAE, R²).

**Key finding:** ensembles beat the baselines, but the top models are close. Colombo's temperature is stable and seasonal, so extra model complexity gives modest gains.

**Live prediction model:** CatBoost, chosen for speed and near-top accuracy.
""")