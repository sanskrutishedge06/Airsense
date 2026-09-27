"""
AirSense - Flask Prediction Server
WHAT THIS DOES:
Runs a small web server. When someone visits /predict, it:
 1. Fetches the last ~2 days of hourly pollution data for the requested
 location (needed to correctly build lag1/lag3/lag24 features, not
 just the current reading).
 2. Builds the exact feature row your models were trained on.
 3. Predicts PM2.5 for 1hr, 6hr and 24hr horizons using your trained
 Random Forest models.
 4. Runs the fuzzy logic layer to turn each prediction into a risk
 category (Good / Moderate / Poor / Severe).
 5. Returns everything as JSON, including the current pollutant readings.
Also exposes /test, a manual demo route that bypasses live data and the
ML models entirely, letting you type any PM2.5 number and see how the
fuzzy logic layer classifies it -- useful for presentations, since it
does not depend on the real weather on demo day.
IMPORTANT: The models were trained ONLY on Talegaon Dabhade data.
They will technically return a prediction for ANY location (since
Open-Meteo covers the whole world), but accuracy is only validated for
Talegaon Dabhade. See the "location_note" field in every /predict response.
Install first (in your venv):
 pip install flask requests pandas numpy scikit-learn scikit-fuzzy joblib
Run:
 python app.py
Then visit:
 http://127.0.0.1:5000/predict?lat=18.7327&lon=73.6752
"""
from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
app = Flask(__name__)
# ---------------------------------------------------
# Load your trained models (Random Forest, 1-hour, 6-hour and 24-hour
# horizons). All three were trained on the same 11 features, so one
# feature row can be reused for every horizon.
# ---------------------------------------------------
MODEL_FILES = {
 "1hr": "model_target_1hr.pkl",
 "6hr": "model_target_6hr.pkl",
 "24hr": "model_target_24hr.pkl",
}
models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
TRAINED_LOCATION = {"lat": 18.7327, "lon": 73.6752, "name": "Talegaon Dabhade"}
FEATURE_COLS = [
 "pm2_5", "pm10", "co", "no2", "o3",
 "hour", "day_of_week", "is_weekend",
 "pm2_5_lag1", "pm2_5_lag3", "pm2_5_lag24",
]
# ---------------------------------------------------
# Build the fuzzy logic system once, at startup
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
def fetch_recent_data(lat, lon):
 """
 Fetches the last 2 days + today's forecast hours so we have enough
 history to compute real lag1/lag3/lag24 features, not placeholders.
 Also returns the location's UTC offset, so the server can work out
 the correct "current" local hour even though it runs on UTC time
 (e.g. on Render).
 """
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
 return df, data.get("utc_offset_seconds", 0)
@app.route("/")
def home():
 return render_template("index.html")
@app.route("/predict", methods=["GET"])
def predict():
 lat = float(request.args.get("lat", TRAINED_LOCATION["lat"]))
 lon = float(request.args.get("lon", TRAINED_LOCATION["lon"]))
 df, utc_offset = fetch_recent_data(lat, lon)
 # "now" in the location's local time (server clock is UTC on Render)
 now = (pd.Timestamp.now(tz="UTC").tz_localize(None)
 + pd.Timedelta(seconds=utc_offset)).floor("h")
 # Find the row closest to "now" so we predict from the most current reading
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
 # Run all three horizons on the same feature row
 predictions = {}
 for name, m in models.items():
   p = float(m.predict(X)[0])
   cat, score = classify_aqi(p)
   predictions[name] = {
      "predicted_pm2_5": round(p, 2),
      "risk_category": cat,
      "risk_score": score,
 }
 pollutants = {
 "pm2_5": round(float(current["pm2_5"]), 2),
 "pm10": round(float(current["pm10"]), 2),
 "co": round(float(current["co"]), 2),
 "no2": round(float(current["no2"]), 2),
 "o3": round(float(current["o3"]), 2),
 }
 is_trained_location = (
 abs(lat - TRAINED_LOCATION["lat"]) < 0.05
 and abs(lon - TRAINED_LOCATION["lon"]) < 0.05
 )
 return jsonify({
 "location": {"lat": lat, "lon": lon},
 "current_reading_time": str(current["time"]),
 "current_pm2_5": current["pm2_5"],
 "pollutants": pollutants,
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
 you can demonstrate all four risk categories (Good/Moderate/Poor/Severe)
 on demand, regardless of the real weather on presentation day.
 This does NOT reflect a live prediction -- it only tests the
 classification layer in isolation.
 Example:
 /test?pm25=120
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