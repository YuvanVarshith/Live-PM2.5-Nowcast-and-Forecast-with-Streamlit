"""
app.py - Chennai Live PM2.5 Nowcast & 24-48h Forecast

Notes:
- Tries OpenAQ first (can be disabled with USE_OPENAQ=False).
- Falls back to Open-Meteo air-quality if OpenAQ is unavailable.
- Ensures timezone-aware indices (Asia/Kolkata) to avoid tz-naive/tz-aware errors.
- Features: lag & rolling features, IsolationForest anomalies, GradientBoostingRegressor,
  recursive forecasting (persistence weather for multi-step).
"""

from datetime import datetime, timedelta
import io
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
LAT = 13.0827
LON = 80.2707
RADIUS_M = 15000
TIMEZONE = "Asia/Kolkata"
MAX_HISTORY_DAYS = 14
FORECAST_HORIZON = 48
LAGS = list(range(1, 25))  # 1..24 lags

# Toggle OpenAQ usage: set to False to always use Open-Meteo (quiet, stable)
USE_OPENAQ = True

# ---------- HELPERS / DATA FETCH ----------
@st.cache_data(ttl=3600)
def fetch_openaq_pm25(lat: float, lon: float, days: int = 7):
    """
    Try OpenAQ v2 measurements. If no data or error -> return empty DataFrame.
    Handles 410 (Gone) explicitly and returns empty so caller can fallback.
    """
    if not USE_OPENAQ:
        st.info("Skipping OpenAQ (USE_OPENAQ=False). Using Open-Meteo fallback.")
        return pd.DataFrame(columns=["pm25"])

    end_utc = datetime.utcnow()
    start_utc = end_utc - timedelta(days=days)
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "parameter": "pm25",
        "coordinates": f"{lat},{lon}",
        "radius": RADIUS_M,
        "date_from": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_to": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": 10000,
        "page": 1,
        "sort": "desc",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 410:
            st.info("OpenAQ returned 410 Gone for this query — using fallback (Open-Meteo).")
            return pd.DataFrame(columns=["pm25"])
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        rows = []
        for rec in results:
            utc_ts = rec.get("date", {}).get("utc")
            val = rec.get("value")
            if utc_ts is not None and val is not None:
                rows.append({"utc": utc_ts, "value": val})
    except requests.exceptions.RequestException as e:
        st.info(f"OpenAQ request failed ({e}). Falling back to Open-Meteo.")
        return pd.DataFrame(columns=["pm25"])
    except Exception as e:
        st.info(f"OpenAQ fetch error: {e}. Falling back to Open-Meteo.")
        return pd.DataFrame(columns=["pm25"])

    if not rows:
        return pd.DataFrame(columns=["pm25"])

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["utc"], utc=True)
    df = df.set_index("datetime").sort_index()
    hourly = df["value"].resample("1H").mean().to_frame("pm25")
    # convert to local timezone (tz-aware)
    hourly.index = hourly.index.tz_convert(TIMEZONE)
    hourly.index.name = "datetime"
    return hourly

@st.cache_data(ttl=3600)
def fetch_open_meteo_pm25(lat: float, lon: float, days: int = 7):
    """
    Open-Meteo Air Quality fallback (hourly pm2_5).
    Returns tz-aware DataFrame indexed by local time (TIMEZONE).
    """
    base = "https://air-quality-api.open-meteo.com/v1/air-quality"
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE
    }
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.get("time"))
        values = hourly.get("pm2_5", [])
        df = pd.DataFrame({"datetime": times, "pm25": values})
        df = df.set_index("datetime").sort_index()
        # ensure tz-aware in TIMEZONE
        if df.index.tz is None:
            df.index = df.index.tz_localize(TIMEZONE)
        else:
            df.index = df.index.tz_convert(TIMEZONE)
        df.index.name = "datetime"
        return df
    except Exception as e:
        st.warning(f"Open-Meteo air-quality fallback error: {e}")
        return pd.DataFrame(columns=["pm25"])

@st.cache_data(ttl=3600)
def fetch_open_meteo_weather(lat: float, lon: float, start_iso: str, end_iso: str):
    """Fetch hourly weather from Open-Meteo and ensure tz-aware index."""
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,pressure_msl",
        "start_date": start_iso.split("T")[0],
        "end_date": end_iso.split("T")[0],
        "timezone": TIMEZONE
    }
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.get("time"))
        df = pd.DataFrame({
            "datetime": times,
            "temperature": hourly.get("temperature_2m"),
            "relative_humidity": hourly.get("relativehumidity_2m"),
            "windspeed": hourly.get("windspeed_10m"),
            "pressure": hourly.get("pressure_msl"),
        })
        df = df.set_index("datetime").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize(TIMEZONE)
        else:
            df.index = df.index.tz_convert(TIMEZONE)
        df.index.name = "datetime"
        return df
    except Exception as e:
        st.warning(f"Open-Meteo weather fetch error: {e}")
        return pd.DataFrame(columns=["temperature", "relative_humidity", "windspeed", "pressure"])

def ensure_tz_index(df: pd.DataFrame):
    """Ensure DataFrame index is tz-aware with TIMEZONE. If empty, return as-is."""
    if df is None or df.empty:
        return df
    df2 = df.copy()
    if df2.index.tz is None:
        df2.index = df2.index.tz_localize(TIMEZONE)
    else:
        df2.index = df2.index.tz_convert(TIMEZONE)
    return df2

def merge_dfs(pm25_df: pd.DataFrame, weather_df: pd.DataFrame):
    """Merge pm25 and weather on hourly index, ensuring tz-aware indices and filling minimally."""
    if pm25_df is None:
        pm25_df = pd.DataFrame(columns=["pm25"])
    if weather_df is None:
        weather_df = pd.DataFrame(columns=["temperature", "relative_humidity", "windspeed", "pressure"])

    pm25_df = ensure_tz_index(pm25_df) if not pm25_df.empty else pm25_df
    weather_df = ensure_tz_index(weather_df) if not weather_df.empty else weather_df

    if (pm25_df is None or pm25_df.empty) and (weather_df is None or weather_df.empty):
        return pd.DataFrame()

    # derive start/end
    starts = []
    ends = []
    if not (pm25_df is None or pm25_df.empty):
        starts.append(pm25_df.index.min())
        ends.append(pm25_df.index.max())
    if not (weather_df is None or weather_df.empty):
        starts.append(weather_df.index.min())
        ends.append(weather_df.index.max())
    start = min(starts)
    end = max(ends)

    idx = pd.date_range(start=start.floor("H"), end=end.ceil("H"), freq="H", tz=TIMEZONE)
    df = pd.DataFrame(index=idx)
    if not (pm25_df is None or pm25_df.empty):
        df = df.join(pm25_df[["pm25"]], how="left")
    else:
        df["pm25"] = np.nan
    if not (weather_df is None or weather_df.empty):
        df = df.join(weather_df[["temperature", "relative_humidity", "windspeed", "pressure"]], how="left")
    else:
        df[["temperature", "relative_humidity", "windspeed", "pressure"]] = np.nan

    # fill a bit (forward then back) - adjust as needed
    df["pm25"] = df["pm25"].ffill().bfill()
    df["temperature"] = df["temperature"].ffill().bfill()
    df["relative_humidity"] = df["relative_humidity"].ffill().bfill()
    df["windspeed"] = df["windspeed"].ffill().bfill()
    df["pressure"] = df["pressure"].ffill().bfill()

    return df

# ---------- FEATURE ENGINEERING / MODELING ----------
def create_features(df: pd.DataFrame):
    """Create lag features, rolling stats, and time features. Returns DataFrame with features (and pm25)."""
    X = df.copy()
    X["hour"] = X.index.hour
    X["dayofweek"] = X.index.dayofweek
    for lag in LAGS:
        X[f"lag_{lag}"] = X["pm25"].shift(lag)
    X["rolling_mean_3"] = X["pm25"].rolling(3).mean().shift(1)
    X["rolling_mean_6"] = X["pm25"].rolling(6).mean().shift(1)
    X["rolling_std_6"] = X["pm25"].rolling(6).std().shift(1)
    X["temp_x_humidity"] = X["temperature"] * X["relative_humidity"]
    X = X.ffill().bfill()
    return X

@st.cache_data(ttl=3600)
def detect_anomalies(X: pd.DataFrame, contamination=0.02):
    """IsolationForest to detect anomalies; returns boolean Series indexed like X."""
    if X is None or X.empty:
        return pd.Series(dtype=bool)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    df_in = X[["pm25", "temperature", "relative_humidity", "windspeed", "pressure"]].fillna(method="ffill").fillna(method="bfill")
    model.fit(df_in)
    preds = model.predict(df_in)  # -1 anomaly, 1 normal
    is_anom = pd.Series(preds == -1, index=X.index)
    return is_anom

def prepare_training_data(df: pd.DataFrame, forecast_horizon=1):
    """Return X (features) and y (target) for supervised learning predicting t+forecast_horizon."""
    X_feat = create_features(df)
    y = X_feat["pm25"].shift(-forecast_horizon)
    X_feat = X_feat.drop(columns=["pm25"], errors="ignore")
    mask = y.notna()
    X_feat = X_feat.loc[mask]
    y = y.loc[mask]
    return X_feat, y

@st.cache_resource
def train_model(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def recursive_forecast(model, recent_df: pd.DataFrame, steps=48):
    """
    Recursive forecast: use model to predict next hour, append predicted pm25 as new row,
    use persistence for weather for multi-step (can be replaced with forecasted weather later).
    """
    if recent_df is None or recent_df.empty:
        return pd.Series(dtype=float)
    df_work = recent_df.copy()
    preds = []
    for i in range(steps):
        # build features for latest timestamp
        feat_row = create_features(df_work).iloc[-1:].drop(columns=["pm25"], errors="ignore")
        pred = float(model.predict(feat_row)[0])
        next_time = df_work.index[-1] + pd.Timedelta(hours=1)
        # use last known weather values (persistence)
        next_row = {
            "pm25": pred,
            "temperature": df_work["temperature"].iloc[-1],
            "relative_humidity": df_work["relative_humidity"].iloc[-1],
            "windspeed": df_work["windspeed"].iloc[-1],
            "pressure": df_work["pressure"].iloc[-1],
        }
        next_df = pd.DataFrame(next_row, index=[next_time])
        next_df.index = ensure_tz_index(next_df).index  # ensure tz-aware
        df_work = pd.concat([df_work, next_df])
        preds.append((next_time, pred))
    index = pd.DatetimeIndex([p[0] for p in preds], tz=TIMEZONE)
    series = pd.Series([p[1] for p in preds], index=index)
    return series

# ---------- STREAMLIT APP UI ----------
st.set_page_config(page_title="Chennai PM2.5 Nowcast & Forecast", layout="wide")
st.title("Chennai — Live PM2.5 Nowcast & 24–48h Forecast")
st.write("Model: GradientBoostingRegressor • Anomaly detection: IsolationForest")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Config")
    days_pull = st.number_input("Days of history to pull", min_value=3, max_value=MAX_HISTORY_DAYS, value=7)
    forecast_hours = st.number_input("Forecast horizon (hours)", min_value=6, max_value=FORECAST_HORIZON, value=48)
    run_button = st.button("Fetch Data & Train Model")
    st.checkbox("Use OpenAQ (fall back to Open-Meteo if unavailable)", value=USE_OPENAQ, key="use_openaq_checkbox")

with col2:
    st.subheader("Notes")
    st.markdown("- OpenAQ (preferred) then Open-Meteo Air Quality (fallback).")
    st.markdown("- Timezone aligned to **Asia/Kolkata** to avoid tz errors.")
    st.markdown("- Multi-step forecast uses persistence for weather (improve later by adding weather forecast inputs).")

# Sync checkbox to global flag (UI control)
try:
    USE_OPENAQ = st.session_state.get("use_openaq_checkbox", USE_OPENAQ)
except Exception:
    pass

# Data fetch + merge
if run_button or "df" not in st.session_state:
    st.session_state["last_run"] = datetime.now().isoformat()
    with st.spinner("Fetching PM2.5 and weather data..."):
        pm25 = fetch_openaq_pm25(LAT, LON, days=int(days_pull))
        if pm25.empty:
            st.info("OpenAQ returned no pm2.5 data — using Open-Meteo air-quality fallback.")
            pm25 = fetch_open_meteo_pm25(LAT, LON, days=int(days_pull))

        start_iso = (datetime.now() - timedelta(days=int(days_pull))).strftime("%Y-%m-%dT00:00")
        end_iso = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%dT23:59")
        weather = fetch_open_meteo_weather(LAT, LON, start_iso, end_iso)

        if (pm25 is None or pm25.empty) and (weather is None or weather.empty):
            st.error("No data from OpenAQ or Open-Meteo. Check internet or endpoints.")
            st.stop()

        df = merge_dfs(pm25, weather)
        if df.empty:
            st.error("Merged dataframe empty. Aborting.")
            st.stop()

    st.session_state["df"] = df.copy()
    st.success("Data fetched & merged.")
else:
    df = st.session_state.get("df")
    if df is None or df.empty:
        st.error("No data loaded. Click 'Fetch Data & Train Model'.")
        st.stop()

# Show recent data
st.subheader("Recent Data (last 48 hours)")
st.dataframe(df.tail(48).round(2))

# Anomaly detection
with st.spinner("Detecting anomalies..."):
    anomalies = detect_anomalies(df, contamination=0.02)
st.subheader("Anomaly Summary")
st.write(f"Anomalies detected in fetched period: **{int(anomalies.sum())}**")
latest_pm = df["pm25"].iloc[-1] if "pm25" in df.columns else float("nan")
st.write(f"Latest observed pm2.5: **{latest_pm:.2f} µg/m³**")
if anomalies.iloc[-1]:
    st.warning("Latest reading flagged as anomaly.")
else:
    st.success("Latest reading appears normal.")

# Prepare training data for 1-hour ahead
with st.spinner("Preparing training data..."):
    X_all, y_all = prepare_training_data(df, forecast_horizon=1)

# Remove anomaly rows from training
mask_not_anom = ~anomalies.reindex(X_all.index, fill_value=False)
X_clean = X_all.loc[mask_not_anom]
y_clean = y_all.loc[mask_not_anom]

# Time-split 80/20 (if too small, fallback to randomized split)
if len(X_clean) < 20:
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
else:
    split_idx = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split_idx]
    y_train = y_clean.iloc[:split_idx]
    X_test = X_clean.iloc[split_idx:]
    y_test = y_clean.iloc[split_idx:]

st.subheader("Model Training")
st.write(f"Training samples: {len(X_train)} — Validation samples: {len(X_test)}")
with st.spinner("Training GradientBoostingRegressor..."):
    model = train_model(X_train, y_train)
st.success("Model trained.")

# Evaluate
with st.spinner("Evaluating..."):
    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
    else:
        mae = float("nan")
st.metric("Validation MAE (1-hour ahead)", f"{mae:.2f} µg/m³" if not math.isnan(mae) else "N/A")

# Forecast
with st.spinner("Generating recursive forecast..."):
    recent_window = df.copy().iloc[-(max(LAGS) + 24):]  # ensure enough history
    preds = recursive_forecast(model, recent_window, steps=int(forecast_hours))

# Plot observed + forecast
st.subheader(f"{int(forecast_hours)}-hour Forecast")
observed_window = df["pm25"].iloc[-72:] if "pm25" in df.columns else pd.Series(dtype=float)
combined = pd.concat([observed_window, preds])
st.line_chart(combined)

# Nowcast / next hour
if not preds.empty:
    next_time = preds.index[0]
    st.subheader("Nowcast / Next hour")
    st.write(f"Predicted PM2.5 at **{next_time.strftime('%Y-%m-%d %H:%M %Z')}**: **{preds.iloc[0]:.2f} µg/m³**")

# Feature importance
st.subheader("Feature importance (Top 15)")
try:
    fi = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
    st.bar_chart(fi)
except Exception as e:
    st.write("Could not compute feature importances:", e)

# Anomalies timeline (recent)
st.subheader("Recent anomalies (last 7 days)")
try:
    recent_anoms = anomalies[anomalies].last("7D") if not anomalies.empty else pd.Series(dtype=bool)
except Exception:
    recent_anoms = anomalies[anomalies] if not anomalies.empty else pd.Series(dtype=bool)

if recent_anoms.empty:
    st.write("No anomalies detected in the last 7 days.")
else:
    st.dataframe(pd.DataFrame({"anomaly": recent_anoms}).tail(200))

# Download data
st.subheader("Download merged dataset")
csv_buf = io.StringIO()
out_df = pd.DataFrame({
    "pm25": df["pm25"],
    "temperature": df["temperature"],
    "relative_humidity": df["relative_humidity"],
    "windspeed": df["windspeed"],
    "pressure": df["pressure"],
    "is_anomaly": anomalies
})
out_df.to_csv(csv_buf)
st.download_button("Download merged dataset (CSV)", data=csv_buf.getvalue(), file_name="chennai_pm25_merged.csv", mime="text/csv")

st.markdown("---")
st.markdown("## Notes & Next Improvements")
st.markdown("""
- Fallback to Open-Meteo air-quality is used when OpenAQ is not available.
- Multi-step forecasts currently use persistence for weather. For better accuracy add hourly weather forecasts as model inputs (Open-Meteo supports forecasted hourly weather).
- Consider LightGBM/XGBoost + hyperparameter tuning, and containerize for deployment.
""")
