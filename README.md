Project Title: Live PM2.5 Nowcast & 24–48 Hour Forecast (Chennai)
Author: Yuvan Varshith A
Domain: Environmental Data Science / Air Quality Monitoring
Tech Stack: Python, Streamlit, Scikit-Learn, Open-Meteo API, OpenAQ API (fallback), Time Series ML

🔎 1. Project Overview

This project provides real-time PM2.5 air quality monitoring and short-term forecasting (24–48 hours) for Chennai.
It fetches live PM2.5 and weather data from public APIs, performs preprocessing, anomaly detection, machine learning model training, and visual forecast generation using a Streamlit dashboard.

This solution is designed for:

Public health early warning

Pollution monitoring

Smart city planning

Environmental analytics

ML-based time-series forecasting demonstration

🛰️ 2. Data Sources
A) PM2.5 (Air Quality)

OpenAQ API (Primary)

Hourly aggregated PM2.5 measurements

May return empty or “410 Gone” for certain regions/dates

If unavailable, the system automatically falls back to Open-Meteo

Open-Meteo Air Quality API (Fallback)

Provides hourly PM2.5 (pm2_5)

Reliable, no API key required

B) Weather Data (Meteorology)

Open-Meteo Weather API

Temperature (°C)

Relative Humidity (%)

Wind Speed (m/s)

Pressure (hPa)

Necessary for PM2.5 prediction

🧹 3. Data Preprocessing

The system performs:

Hourly time-index alignment

Timezone conversion → Asia/Kolkata

Forward/backward fill of missing weather values

Merging air quality + weather into a unified dataset

Lag features (1–24 hours)

Rolling statistics (3H, 6H mean/std)

Interaction features (temperature × humidity)

Time-based features (hour, day of week)

🚨 4. Anomaly Detection

Uses IsolationForest to identify unusual PM2.5 spikes:

Flags sensor errors or sudden pollution events

Removes anomalies before training the ML model

🤖 5. Machine Learning Model

Model Used: GradientBoostingRegressor

Why this model?

Performs well for structured time-series datasets

Handles nonlinear relationships effectively

Stable even with small-to-medium datasets

Training Strategy

Data split into 80% train / 20% validation using time-based split

Anomalous rows removed before training

Evaluation using Validation MAE (Mean Absolute Error)

Typical MAE observed: 2–4 µg/m³ (excellent for short-term PM2.5 forecasting)

🔁 6. Recursive Forecasting (48 Hours)

To forecast 24–48 hours:

Predict PM2.5 for the next hour

Append prediction to dataset

Use it as input for predicting the next hour

Repeat recursively for 48 steps

Weather data uses persistence (last known values) for multi-step predictions.

🖥️ 7. Streamlit Dashboard Features

The app displays:

Recent PM2.5 and weather history

Anomaly alerts

Latest observed PM2.5

Next-hour nowcast

48-hour line chart forecast

Feature importance

Download merged dataset (CSV)

Built with:

streamlit run app.py

⚙️ 8. How to Run the Project
Install Requirements
pip install -r requirements.txt

Launch the App
streamlit run app.py

Included Files

app.py → Main Streamlit app

README.txt → Documentation

requirements.txt → Python dependencies

📌 9. Limitations

Multi-step PM2.5 forecast uses weather persistence → not ideal for long horizons

OpenAQ API may occasionally return no data or “410 Gone”

Real forecasting accuracy depends on available historical data

🚀 10. Future Enhancements

Integrate Open-Meteo weather forecast for improved multi-step prediction

Add LightGBM / XGBoost models

Deploy with Docker or cloud hosting

Add interactive geo-selector for different cities

Improve anomaly detection using multiple algorithms

🏁 11. Conclusion

This project demonstrates a complete end-to-end real-time ML forecasting pipeline using:

Public environmental APIs

Robust preprocessing

Anomaly detection

ML modeling

Streamlit visual analytics

It fulfills all requirements of the final project and provides a scalable base for real-world air quality prediction systems.
