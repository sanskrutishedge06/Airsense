"""
AirSense — Flask Prediction Server (all 3 horizons + fuzzy logic + demo test route)

WHAT THIS DOES:
  1. /predict fetches ~2 days of recent hourly data for the requested
     location (needed to build real lag1/lag3/lag24 features).
  2. Builds the feature row your models were trained on.
  3. Predicts PM2.5 for 1hr, 6hr, AND 24hr in one response, using each
     horizon's saved model.
  4. Runs the fuzzy logic layer on each prediction to get a risk category.
  5. Returns everything as one JSON object shaped like:
     {
       "location": {...},
       "current_pm2_5": ...,
       "current_reading_time": "...",
       "predictions": {
         "1hr":  {"predicted_pm2_5": ..., "risk_category": "...", "risk_score": ...},
         "6hr":  {...},
         "24hr": {...}
       },
       "location_note": "..."
     }
  6. /test?pm25=100 runs ONLY the fuzzy logic classifier on a manually
     entered value, for demos — no live data or model involved.

If a horizon's model file is missing or fails to load (e.g. a TensorFlow
version mismatch for the LSTM model), that horizon is simply left out of
"predictions" instead of crashing the whole endpoint — the front-end
already handles missing horizons gracefully.

Install (in your venv):
    pip install flask requests pandas numpy scikit-learn scikit-fuzzy joblib tensorflow

Run:
    python app.py
Then visit:
    http://127.0.0.1:5000/
"""

from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

TRAINED_LOCATION = {"lat": 18.7327, "lon": 73.6752, "name": "Talegaon Dabhade"}

FEATURE_COLS = [
    "pm2_5", "pm10", "co", "no2", "o3",
    "hour", "day_of_week", "is_weekend",
    "pm2_5_lag1", "pm2_5_lag3", "pm2_5_lag24",
]

# ---------------------------------------------------
# Load models for each horizon. Each entry can be:
#   ("sklearn", model)            -> joblib model, call .predict() directly
#   ("keras", model, scaler)      -> keras model, needs scaled + reshaped input
# If a file is missing or fails to load, that horizon is skipped (not fatal).
# ---------------------------------------------------
HORIZONS = {}

try:
    HORIZONS["1hr"] = ("sklearn", joblib.load("model_target_1hr.pkl"))
    print("Loaded 1hr model (sklearn).")
except Exception as e:
    print(f"WARNING: could not load 1hr model: {e}")

try:
    HORIZONS["6hr"] = ("sklearn", joblib.load("model_target_6hr.pkl"))
    print("Loaded 6hr model (sklearn).")
except Exception as e:
    print(f"WARNING: could not load 6hr model: {e}")

try:
    import tensorflow as tf
    lstm_model = tf.keras.models.load_model("model_target_24hr.h5", compile=False)
    scaler_24hr = joblib.load("scaler_target_24hr.pkl")
    HORIZONS["24hr"] = ("keras", lstm_model, scaler_24hr)
    print("Loaded 24hr model (keras/LSTM).")
except Exception as e:
    print(f"WARNING: could not load 24hr model, skipping this horizon: {e}")

if not HORIZONS:
    raise RuntimeError("No models could be loaded. Check that model files are present.")

# ---------------------------------------------------
# Fuzzy logic system (built once at startup)
# ---------------------------------------------------
pm25_var = ctrl.Antecedent(np.arange(0, 501, 1), "pm25")
risk_var = ctrl.Consequent(np.arange(0, 101, 1), "risk")

pm25_var["low"] = fuzz.trimf(pm25_var.universe, [0, 0, 60])
pm25_var["medium"] = fuzz.trimf(pm25_var.universe, [30, 90, 150])
pm25_var["high"] = fuzz.trimf(pm25_var.universe, [100, 250, 500])

risk_var["good"] = fuzz.trimf(risk_var.universe, [0, 0, 30])
risk_var["moderate"] = fuzz.trimf(risk_var.universe, [20, 50, 70])
risk_var["poor"] = fuzz.trimf(risk_var.universe, [60, 80, 90])
risk_var["severe"] = fuzz.trimf(risk_var.universe, [80, 100, 100])

rule1 = ctrl.Rule(pm25_var["low"], risk_var["good"])
rule2 = ctrl.Rule(pm25_var["medium"], risk_var["moderate"])
rule3 = ctrl.Rule(pm25_var["high"], risk_var["poor"])

risk_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])


def classify_aqi(predicted_pm25):
    sim = ctrl.ControlSystemSimulation(risk_ctrl_system)
    sim.input["pm25"] = float(np.clip(predicted_pm25, 0, 500))
    sim.compute()
    score = sim.output["risk"]

    if score < 30:
        category = "Good"
    elif score < 60:
        category = "Moderate"
    elif score < 85:
        category = "Poor"
    else:
        category = "Severe"
    return category, round(score, 1)


def predict_one_horizon(horizon_key, X):
    """Runs the correct model type for a given horizon and returns a
    (predicted_value, category, score) tuple, or None if unavailable."""
    if horizon_key not in HORIZONS:
        return None

    entry = HORIZONS[horizon_key]
    if entry[0] == "sklearn":
        model = entry[1]
        predicted = float(model.predict(X)[0])
    elif entry[0] == "keras":
        model, scaler = entry[1], entry[2]
        X_scaled = scaler.transform(X)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        predicted = float(model.predict(X_lstm, verbose=0).flatten()[0])
    else:
        return None

    category, score = classify_aqi(predicted)
    return predicted, category, score


def fetch_recent_data(lat, lon):
    """Fetches ~2 days of history so real lag1/lag3/lag24 values can be
    computed, instead of reusing the current reading as a placeholder."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone",
        "past_days": 2,
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df.rename(columns={
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "ozone": "o3",
    }, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
def predict():
    lat = float(request.args.get("lat", TRAINED_LOCATION["lat"]))
    lon = float(request.args.get("lon", TRAINED_LOCATION["lon"]))

    df = fetch_recent_data(lat, lon)

    now = pd.Timestamp.now().floor("h")
    diffs = (df["time"] - now).abs()
    idx = int(diffs.values.argmin())

    if idx < 24:
        return jsonify({
            "error": "Not enough historical hours returned to compute lag features. Try again shortly."
        }), 400

    current = df.iloc[idx]
    lag1 = df.iloc[idx - 1]["pm2_5"]
    lag3 = df.iloc[idx - 3]["pm2_5"]
    lag24 = df.iloc[idx - 24]["pm2_5"]

    row = {
        "pm2_5": current["pm2_5"],
        "pm10": current["pm10"],
        "co": current["co"],
        "no2": current["no2"],
        "o3": current["o3"],
        "hour": current["time"].hour,
        "day_of_week": current["time"].dayofweek,
        "is_weekend": int(current["time"].dayofweek in [5, 6]),
        "pm2_5_lag1": lag1,
        "pm2_5_lag3": lag3,
        "pm2_5_lag24": lag24,
    }
    X = pd.DataFrame([row])[FEATURE_COLS]

    predictions = {}
    for horizon_key in ["1hr", "6hr", "24hr"]:
        result = predict_one_horizon(horizon_key, X)
        if result is not None:
            predicted, category, score = result
            predictions[horizon_key] = {
                "predicted_pm2_5": round(predicted, 2),
                "risk_category": category,
                "risk_score": score,
            }

    is_trained_location = (
        abs(lat - TRAINED_LOCATION["lat"]) < 0.05
        and abs(lon - TRAINED_LOCATION["lon"]) < 0.05
    )

    return jsonify({
        "location": {"lat": lat, "lon": lon},
        "current_reading_time": str(current["time"]),
        "current_pm2_5": round(float(current["pm2_5"]), 1),
        "current_pollutants": {
            "pm2_5": {"value": round(float(current["pm2_5"]), 1), "unit": "\u00b5g/m\u00b3"},
            "pm10": {"value": round(float(current["pm10"]), 1), "unit": "\u00b5g/m\u00b3"},
            "co": {"value": round(float(current["co"]), 1), "unit": "\u00b5g/m\u00b3"},
            "no2": {"value": round(float(current["no2"]), 1), "unit": "\u00b5g/m\u00b3"},
            "o3": {"value": round(float(current["o3"]), 1), "unit": "\u00b5g/m\u00b3"},
        },
        "predictions": predictions,
        "location_note": (
            f"Model trained on {TRAINED_LOCATION['name']} data. "
            + ("This request matches the trained location."
               if is_trained_location else
               "This location differs from the trained location "
               f"({TRAINED_LOCATION['name']}); prediction accuracy is not validated here.")
        ),
    })


@app.route("/test", methods=["GET"])
def test_classifier():
    """
    DEMO / PRESENTATION TOOL ONLY.
    Bypasses live data and the ML models entirely. Lets you manually enter
    a PM2.5 value and see the fuzzy logic classifier's output directly, so
    you can demonstrate all four risk categories on demand regardless of
    the real weather on presentation day.
    """
    pm25_value = request.args.get("pm25", type=float)
    if pm25_value is None:
        return jsonify({"error": "Please provide a pm25 value, e.g. /test?pm25=100"}), 400

    category, risk_score = classify_aqi(pm25_value)
    return jsonify({
        "mode": "manual_test",
        "note": "This is a demo of the fuzzy logic layer only, not a live prediction.",
        "input_pm2_5": pm25_value,
        "risk_category": category,
        "risk_score": risk_score,
    })


if __name__ == "__main__":
    app.run(debug=True)