from flask import Flask, request, jsonify
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('maintenance_prediction_model.h5')
scaler = joblib.load('./maintenance_prediction/scaler.pkl')

@app.route('/')
def home():
    return "Bike Maintenance Prediction Model"

@app.route('/maintenance-predict', methods=['POST'])
def predict():
    data = request.json
    
    # Create DataFrame from received JSON data
    input_data = pd.DataFrame([data])
    
    # Ensure the input columns match the model's expected features
    input_data = input_data[['station_id', 'Hour', 'Month', 'Day', 'DayOfWeek', 'weather_sunny', 'weather_rainy', 'holidays_public_holiday']]
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled).argmax(axis=1)[0]
    
    # Return the prediction as a JSON response
    return jsonify({'maintenance_needed': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
