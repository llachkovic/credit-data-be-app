import numpy as np
import pandas as pd
from flask import request, jsonify, Blueprint, g
from sklearn.linear_model import LogisticRegression

from app.data_types import DataPoint, one_hot_encode_payload
from app.models import db, PredictionResult

main_blueprint = Blueprint("main", __name__)


@main_blueprint.route('/', methods=['POST'])
def predict():
    model: LogisticRegression = g.model
    if request.is_json:
        data = request.get_json()
        df = pd.DataFrame([data])

        categorical_columns = ['checkingAccountStatus', 'creditHistory', 'purpose', 'savingsAccount',
                               'employmentDuration', 'personalStatus', 'otherDebtors', 'property',
                               'otherInstallmentPlans', 'housing', 'job', 'telephone', 'foreignWorker']
        for column in categorical_columns:
            df = one_hot_encode_payload(df, column)

        data_point = DataPoint(df)
        print(data_point.__dict__)

        data_array = np.array([data_point.__dict__[key] for key in data_point.__dict__])
        print(data_array)
        prediction = model.predict_proba(data_array.reshape(1, -1))[0]
        print(prediction)

        new_record = PredictionResult(probability=prediction[0])
        db.session.add(new_record)
        db.session.commit()

        response = {
            'id': new_record.id,
            'probability': new_record.probability
        }

        return jsonify(response)

    else:
        error_result = {'error': 'Invalid request format'}
        return jsonify(error_result), 400
