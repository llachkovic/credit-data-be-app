from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from data_types import one_hot_encode_payload
from flask_cors import CORS

from data_types import DataPoint

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        df = pd.DataFrame([data])
        categorical_columns = ['checkingAccountStatus', 'creditHistory', 'purpose', 'savingsAccount',
                               'employmentDuration', 'personalStatus', 'otherDebtors', 'property',
                               'otherInstallmentPlans', 'housing', 'job', 'telephone', 'foreignWorker']
        for column in categorical_columns:
            df = one_hot_encode_payload(df, column)

        data_point = DataPoint(df)
        model: LogisticRegression = joblib.load('models/linear_regression.joblib')
        scaler: StandardScaler = joblib.load('models/linear_regression.scaler.joblib')

        data_array = np.array([data_point.__dict__[key] for key in data_point.__dict__])
        data_array_scaled = scaler.transform(data_array.reshape(1, -1))
        print(data_array_scaled)
        prediction = model.predict_proba(data_array_scaled)[0]

        response = {
            'probability_good': prediction[0],
            'probability_bad': prediction[1]
        }
        return jsonify(response)

    else:
        error_result = {'error': 'Invalid request format'}
        return jsonify(error_result), 400


if __name__ == '__main__':
    app.run()
