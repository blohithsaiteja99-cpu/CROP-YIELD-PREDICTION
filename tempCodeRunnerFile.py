from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "yield_model.pkl"

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    else:
        print("Model file not found")
        model = None

except Exception as e:
    print("Error loading model:", e)
    model = None


# Home Page
@app.route('/')
def home():
    return render_template("index.html")


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Get form data
        region = request.form['Region']
        soil_type = request.form['Soil_Type']
        crop = request.form['Crop']
        weather = request.form['Weather_Condition']

        rainfall = float(request.form['Rainfall_mm'])
        temperature = float(request.form['Temperature_Celsius'])
        fertilizer = float(request.form['Fertilizer_Used'])
        irrigation = int(request.form['Irrigation_Used'])
        days = int(request.form['Days_to_Harvest'])

        # Create dataframe
        sample = pd.DataFrame({
            'Region': [region],
            'Soil_Type': [soil_type],
            'Crop': [crop],
            'Weather_Condition': [weather],
            'Rainfall_mm': [rainfall],
            'Temperature_Celsius': [temperature],
            'Fertilizer_Used': [fertilizer],
            'Irrigation_Used': [irrigation],
            'Days_to_Harvest': [days]
        })

        # Predict
        prediction = model.predict(sample)

        output = round(float(prediction[0]), 2)

        return f"Predicted Crop Yield: {output}"

    except Exception as e:

        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)