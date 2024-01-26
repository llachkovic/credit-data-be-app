from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_types import DataPoint

app = Flask(__name__)


@app.route('/', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()

        try:
            request_data = DataPoint(**data)
            model: LogisticRegression = joblib.load('models/linear_regression.joblib')
            scaler: StandardScaler = joblib.load('models/linear_regression.scaler.joblib')
            data_array = np.array([request_data.__dict__[key] for key in request_data.__dict__])
            data_array_scaled = scaler.transform(data_array.reshape(1, -1))
            print(data_array_scaled)
            prediction = model.predict_proba(data_array_scaled)[0]

            response = {
                'probability_good': prediction[0],
                'probability_bad': prediction[1]
            }
            return jsonify(response)

        except TypeError as e:
            error_result = {'error': f'Invalid request format - {e}'}
            return jsonify(error_result), 400

    else:
        # Return a response if the request does not contain JSON data
        error_result = {'error': 'Invalid request format'}
        return jsonify(error_result), 400


if __name__ == '__main__':
    app.run()
