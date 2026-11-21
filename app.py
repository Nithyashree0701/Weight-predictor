from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "linear_regression_model.joblib"
model = joblib.load(MODEL_PATH)

# Typical average heights (in inches) used to gauge input reliability
MEAN_HEIGHT = {"Male": 69.0, "Female": 64.0}
BASE_ACCURACY = 90.0  # Your model's baseline R² score (%)

LBS_TO_KG = 0.45359237


def predict_weight(gender, height):
    gender = str(gender).strip().capitalize()
    if gender not in ("Male", "Female"):
        raise ValueError("Gender must be 'Male' or 'Female'.")
    height = float(height)
    if height <= 0:
        raise ValueError("Height must be a positive number.")

    df = pd.DataFrame([[gender, height]], columns=["Gender", "Height"])
    weight_lbs = float(model.predict(df)[0])
    weight_kg = round(weight_lbs * LBS_TO_KG, 2)

    # Calculate dynamic confidence based on distance from typical height
    deviation = abs(height - MEAN_HEIGHT[gender])
    confidence = max(10.0, min(99.0, round(BASE_ACCURACY - (deviation * 1.2), 1)))

    return weight_kg, confidence



@app.route("/", methods=["GET", "POST"])
def home():
    result, accuracy, error = None, None, None
    gender, height = "Male", ""
    if request.method == "POST":
        gender = request.form.get("gender", "Male")
        height = request.form.get("height", "")
        try:
            result, accuracy = predict_weight(gender, height)
        except (ValueError, KeyError) as e:
            error = str(e)
    return render_template(
        "index.html",
        result=result,
        accuracy=accuracy,
        error=error,
        gender=gender,
        height=height,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        weight, accuracy = predict_weight(data["gender"], data["height"])
        return jsonify({"predicted_weight": weight, "model_accuracy": accuracy})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
