from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model, scaler, and label encoder using pandas
model = pd.read_pickle("traffic_model.pkl")
scaler = pd.read_pickle("scaler.pkl")
label_encoder = pd.read_pickle("label_encoder.pkl")

# Home route
@app.route("/")
def home():
    return render_template("index.html", prediction_text="", hour="", temp="", weather="")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        hour = request.form["hour"]
        temp = request.form["temp"]
        weather = request.form["weather"]

        # Convert input values
        hour_int = int(hour)
        temp_float = float(temp)

        # Encode weather condition
        weather_encoded = label_encoder.transform([weather])[0]

        # Feature scaling
        input_features = pd.DataFrame([[hour_int, temp_float, weather_encoded]], columns=["hour", "temp", "weather"])
        input_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        return render_template("index.html", 
                               prediction_text=f"Estimated Traffic Volume: {int(prediction)}",
                               hour=hour, temp=temp, weather=weather)

    except Exception as e:
        return render_template("index.html", 
                               prediction_text=f"Error: {str(e)}",
                               hour=hour, temp=temp, weather=weather)

if __name__ == "__main__":
    app.run(debug=True)
