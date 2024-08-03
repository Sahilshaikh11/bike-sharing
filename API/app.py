# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# # Load the trained model
# model = joblib.load('bike_demand_model.pkl')

# # Initialize Flask app
# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)

#     # Extract data from request
#     hour = data.get('Hour', 0)
#     temperature = data.get('Temperature', 0.0)
#     humidity = data.get('Humidity', 0.0)
#     wind_speed = data.get('Wind_speed', 0.0)
#     visibility = data.get('Visibility', 0)
#     dew_point_temperature = data.get('Dew_point_temperature', 0.0)
#     solar_radiation = data.get('Solar_Radiation', 0.0)
#     rainfall = data.get('Rainfall', 0.0)
#     snowfall = data.get('Snowfall', 0.0)
#     year = data.get('Year', 0)
#     month = data.get('Month', 0)
#     day = data.get('Day', 0)
#     day_of_week = data.get('DayOfWeek', 0)
#     season = data.get('Season', 'Autumn')
#     holiday = data.get('Holiday', 'No Holiday')
#     functioning_day = data.get('Functioning_Day', 'Yes')
    
#     # Create a DataFrame for the input
#     input_data = pd.DataFrame({
#         'Hour': [hour],
#         'Temperature': [temperature],
#         'Humidity': [humidity],
#         'Wind_speed': [wind_speed],
#         'Visibility': [visibility],
#         'Dew_point_temperature': [dew_point_temperature],
#         'Solar_Radiation': [solar_radiation],
#         'Rainfall': [rainfall],
#         'Snowfall': [snowfall],
#         'Year': [year],
#         'Month': [month],
#         'Day': [day],
#         'DayOfWeek': [day_of_week],
#         'Seasons_Autumn': [1 if season == 'Autumn' else 0],
#         'Seasons_Spring': [1 if season == 'Spring' else 0],
#         'Seasons_Summer': [1 if season == 'Summer' else 0],
#         'Seasons_Winter': [1 if season == 'Winter' else 0],
#         'Holiday_Holiday': [1 if holiday == 'Holiday' else 0],
#         'Holiday_No Holiday': [1 if holiday == 'No Holiday' else 0],
#         'Functioning_Day_No': [1 if functioning_day == 'No' else 0],
#         'Functioning_Day_Yes': [1 if functioning_day == 'Yes' else 0]
#     })

#     # Make prediction
#     prediction = model.predict(input_data)

#     # Return the result as JSON
#     return jsonify({'predicted_demand': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------

# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# # Load the trained model
# model = joblib.load('bike_demand_model.pkl')

# # Initialize Flask app
# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)

#     # Extract data from request
#     hour = data.get('Hour', 0)
#     temperature = data.get('Temperature', 0.0)
#     humidity = data.get('Humidity', 0.0)
#     wind_speed = data.get('Wind_speed', 0.0)
#     visibility = data.get('Visibility', 0)
#     rainfall = data.get('Rainfall', 0.0)
#     snowfall = data.get('Snowfall', 0.0)
#     year = data.get('Year', 0)
#     month = data.get('Month', 0)
#     day = data.get('Day', 0)
#     day_of_week = data.get('DayOfWeek', 0)
#     season = data.get('Season', 'Autumn')
#     holiday = data.get('Holiday', 'No Holiday')
#     functioning_day = data.get('Functioning_Day', 'Yes')
    
#     # Create a DataFrame for the input
#     input_data = pd.DataFrame({
#         'Hour': [hour],
#         'Temperature': [temperature],
#         'Humidity': [humidity],
#         'Wind_speed': [wind_speed],
#         'Visibility': [visibility],
#         'Dew_point_temperature': [],  # Default value for Dew_point_temperature
#         'Solar_Radiation': [0.2],  # Default value for Solar_Radiation
#         'Rainfall': [rainfall],
#         'Snowfall': [snowfall],
#         'Year': [year],
#         'Month': [month],
#         'Day': [day],
#         'DayOfWeek': [day_of_week],
#         'Seasons_Autumn': [1 if season == 'Autumn' else 0],
#         'Seasons_Spring': [1 if season == 'Spring' else 0],
#         'Seasons_Summer': [1 if season == 'Summer' else 0],
#         'Seasons_Winter': [1 if season == 'Winter' else 0],
#         'Holiday_Holiday': [1 if holiday == 'Holiday' else 0],
#         'Holiday_No Holiday': [1 if holiday == 'No Holiday' else 0],
#         'Functioning_Day_No': [1 if functioning_day == 'No' else 0],
#         'Functioning_Day_Yes': [1 if functioning_day == 'Yes' else 0],
#     })

#     # Make prediction
#     prediction = model.predict(input_data)

#     # Return the result as JSON
#     return jsonify({'predicted_demand': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)

# ----------------------------------------------------------------------------------------------------------------------------------------------------

from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained models
demand_model = joblib.load('bike_demand_model.pkl')
maintenance_model = joblib.load('bike_maintenance_model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.get_json(force=True)

    # Extract data from request
    hour = data.get('Hour', 0)
    temperature = data.get('Temperature', 0.0)
    humidity = data.get('Humidity', 0.0)
    wind_speed = data.get('Wind_speed', 0.0)
    visibility = data.get('Visibility', 0)
    rainfall = data.get('Rainfall', 0.0)
    snowfall = data.get('Snowfall', 0.0)
    year = data.get('Year', 0)
    month = data.get('Month', 0)
    day = data.get('Day', 0)
    day_of_week = data.get('DayOfWeek', 0)
    season = data.get('Season', 'Autumn')
    holiday = data.get('Holiday', 'No Holiday')
    functioning_day = data.get('Functioning_Day', 'Yes')
    station_id = data.get('Station_Id', 1)  # Default to 'Station_Id_1'

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Hour': [hour],
        'Temperature(è\x9a\x93)': [temperature],  # Match encoding used in training
        'Humidity': [humidity],
        'Wind_speed': [wind_speed],
        'Visibility': [visibility],
        'Dew point temperature(è\x9a\x93)': [4.073812785],  # Default value
        'Solar_Radiation': [0.5691107306],  # Default value
        'Rainfall': [rainfall],
        'Snowfall': [snowfall],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'DayOfWeek': [day_of_week],
        'Seasons_Autumn': [1 if season == 'Autumn' else 0],
        'Seasons_Spring': [1 if season == 'Spring' else 0],
        'Seasons_Summer': [1 if season == 'Summer' else 0],
        'Seasons_Winter': [1 if season == 'Winter' else 0],
        'Holiday_Holiday': [1 if holiday == 'Holiday' else 0],
        'Holiday_No Holiday': [1 if holiday == 'No Holiday' else 0],
        'Functioning_Day_No': [1 if functioning_day == 'No' else 0],
        'Functioning_Day_Yes': [1 if functioning_day == 'Yes' else 0]
    })

    # Add Station_Id columns dynamically
    for i in range(1, 51):  # From Station_Id_1 to Station_Id_50
        input_data[f'Station_Id_{i}'] = [1 if station_id == i else 0]

    # Ensure all features are present in the input DataFrame
    # Get all feature columns that are expected
    expected_features = [col for col in demand_model.feature_names_in_]
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match the model's expected feature order
    input_data = input_data[expected_features]

    # Make prediction
    prediction = demand_model.predict(input_data)

    # Return the result as JSON
    return jsonify({'predicted_demand': prediction[0]})


@app.route('/predict_maintenance', methods=['POST'])
def predict_maintenance():
    data = request.get_json(force=True)

    # Extract data from request
    hour = data.get('Hour', 0)
    temperature = data.get('Temperature', 0.0)
    humidity = data.get('Humidity', 0.0)
    wind_speed = data.get('Wind_speed', 0.0)
    visibility = data.get('Visibility', 0)
    rainfall = data.get('Rainfall', 0.0)
    snowfall = data.get('Snowfall', 0.0)
    year = data.get('Year', 0)
    month = data.get('Month', 0)
    day = data.get('Day', 0)
    day_of_week = data.get('DayOfWeek', 0)
    season = data.get('Season', 'Autumn')
    holiday = data.get('Holiday', 'No Holiday')
    functioning_day = data.get('Functioning_Day', 'Yes')
    station_id = data.get('Station_Id', 1)  # Default to 'Station_Id_1'

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Hour': [hour],
        'Temperature(è\x9a\x93)': [temperature],  # Match encoding used in training
        'Humidity': [humidity],
        'Wind_speed': [wind_speed],
        'Visibility': [visibility],
        'Dew point temperature(è\x9a\x93)': [4.073812785],  # Default value
        'Solar_Radiation': [0.5691107306],  # Default value
        'Rainfall': [rainfall],
        'Snowfall': [snowfall],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'DayOfWeek': [day_of_week],
        'Seasons_Autumn': [1 if season == 'Autumn' else 0],
        'Seasons_Spring': [1 if season == 'Spring' else 0],
        'Seasons_Summer': [1 if season == 'Summer' else 0],
        'Seasons_Winter': [1 if season == 'Winter' else 0],
        'Holiday_Holiday': [1 if holiday == 'Holiday' else 0],
        'Holiday_No Holiday': [1 if holiday == 'No Holiday' else 0],
        'Functioning_Day_No': [1 if functioning_day == 'No' else 0],
        'Functioning_Day_Yes': [1 if functioning_day == 'Yes' else 0]
    })

    # Add Station_Id columns dynamically
    for i in range(1, 51):  # From Station_Id_1 to Station_Id_50
        input_data[f'Station_Id_{i}'] = [1 if station_id == i else 0]

    # Ensure all features are present in the input DataFrame
    # Get all feature columns that are expected
    expected_features = [col for col in demand_model.feature_names_in_]
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match the model's expected feature order
    input_data = input_data[expected_features]

    # Make prediction
    prediction = maintenance_model.predict(input_data)

    # Return the result as JSON
    return jsonify({'predicted_maintenance_needed': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
