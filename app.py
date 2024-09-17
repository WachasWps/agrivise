import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Logging setup
logging.basicConfig(level=logging.INFO)

# Folder path where all CSV files are stored (adjust for your file structure)
folder_path = "finalCSVPrice"

# Function to prepare the dataset
def prepare_data(state_name, data_path):
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop_duplicates(subset=['Date'])

    if state_name not in df.columns:
        raise ValueError(f"No data found for the state: {state_name}")

    ts = df.set_index('Date')[state_name]
    ts = ts.ffill().bfill().dropna()
    ts = ts.asfreq('D')
    return ts

# Function to predict price using ARIMA
def predict_price_arima(date_to_predict, state_name, commodity_name):
    # Construct the file path based on the commodity name
    file_path = os.path.join(folder_path, f"{commodity_name}.csv")

    # Prepare the time series data
    ts = prepare_data(state_name, file_path)

    # Convert the prediction date to a datetime object
    forecast_date = pd.to_datetime(date_to_predict)

    # Check if the prediction date is valid
    if forecast_date <= ts.index[-1]:
        raise ValueError(f"The prediction date must be after the last available date in the data: {ts.index[-1].date()}")

    # Calculate the steps between the last date in the series and the forecast date
    steps = (forecast_date - ts.index[-1]).days

    # Fit the ARIMA model
    model = ARIMA(ts, order=(5, 0, 3))
    model_fit = model.fit()

    # Forecast the price
    forecast = model_fit.forecast(steps=steps)
    return forecast.iloc[-1]

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.json
        commodity_name = data.get('commodity')
        state_name = data.get('state')
        date_to_predict = data.get('date')

        if not commodity_name or not state_name or not date_to_predict:
            return jsonify({"error": "Missing required parameters"}), 400

        predicted_price = predict_price_arima(date_to_predict, state_name, commodity_name)

        return jsonify({"predicted_price": predicted_price}), 200

    except Exception as e:
        app.logger.error(f"Exception occurred during prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
