from flask import Flask, request, jsonify
import logging
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log request receipt
        logging.info("Received request")
        
        # Get data from request
        data = request.get_json()
        if not data:
            logging.error("No data provided in request")
            return jsonify({"error": "No data provided"}), 400
        
        commodity = data.get('commodity')
        state = data.get('state')
        date = data.get('date')
        
        if not all([commodity, state, date]):
            logging.error("Missing required fields in request data")
            return jsonify({"error": "Missing required fields"}), 400
        
        # Log received data
        logging.info(f"Data received: Commodity: {commodity}, State: {state}, Date: {date}")
        
        # Sample data processing and prediction logic
        # Replace with actual data and model logic
        try:
            # Example placeholder data and model setup
            sample_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            model = ARIMA(sample_data, order=(1, 1, 1))
            result = model.fit()
            prediction = result.predict(start=len(sample_data), end=len(sample_data))
            
            # Log prediction
            logging.info(f"Prediction result: {prediction.tolist()}")
            
            return jsonify({"prediction": prediction.tolist()})
        except Exception as model_error:
            logging.error(f"Error during prediction: {model_error}")
            return jsonify({"error": "Error during prediction"}), 500

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)