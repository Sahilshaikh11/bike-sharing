from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('bike_demand_rs_model.pkl')

@app.route('/')
def home():
    return "Bike Demand Prediction Model"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        
        # Validate received data
        expected_features = ['station_id', 'Hour', 'Month', 'Day', 'DayOfWeek', 'weather_sunny', 'weather_rainy', 'holidays_public_holiday']
        for feature in expected_features:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Create DataFrame from received JSON data
        input_data = pd.DataFrame([data])

        # Ensure the input columns match the model's expected features
        input_data = input_data[expected_features]

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
