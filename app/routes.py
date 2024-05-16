import numpy as np
import pandas as pd
from flask import request, jsonify, Blueprint, current_app
from sklearn.linear_model import LogisticRegression

from app.data_types import DataPoint
from app.models import db, PredictionResult

main_blueprint = Blueprint("main", __name__)


@main_blueprint.route('/predict', methods=['POST'])
def predict_result():
    model: LogisticRegression = current_app.model
    if request.is_json:
        data = request.get_json()
        df = pd.DataFrame([data])

        numerical_features = df.select_dtypes(include=['int64', 'float64'])
        categorical_features = df.select_dtypes(include=['object'])

        categorical_features = pd.get_dummies(categorical_features)

        df_encoded = pd.concat([numerical_features, categorical_features], axis=1)

        data_point = DataPoint(df_encoded)
        print(data_point.__dict__)

        data_array = np.array([data_point.__dict__[key] for key in data_point.__dict__])
        print(data_array)
        prediction = model.predict_proba(data_array.reshape(1, -1))[0]
        print(prediction)

        new_record = PredictionResult(
            probability=prediction[0],
            age=data['age'],
            checking_account_status=data['checkingAccountStatus'],
            credit_amount=data['creditAmount'],
            credit_history=data['creditHistory'],
            duration_in_months=data['durationInMonths'],
            employment_duration=data['employmentDuration'],
            foreign_worker=data['foreignWorker'],
            housing=data['housing'],
            installment_rate=data['installmentRate'],
            job=data['job'],
            number_of_existing_credits=data['numberOfExistingCredits'],
            number_of_people_liable_for_maintenance=data['numberOfPeopleLiableForMaintenance'],
            other_debtors=data['otherDebtors'],
            other_installment_plans=data['otherInstallmentPlans'],
            personal_status=data['personalStatus'],
            present_residence_since=data['presentResidenceSince'],
            property=data['property'],
            purpose=data['purpose'],
            savings_account=data['savingsAccount'],
            telephone=data['telephone'],
            email=data['email']
        )

        try:
            db.session.add(new_record)
            db.session.commit()
        except Exception as e:
            error_message = str(e)
            return jsonify({'error': error_message}), 500

        db.session.refresh(new_record)

        response = {key: getattr(new_record, key) for key in new_record.__dict__.keys() if not key.startswith('_')}
        return jsonify(response)

    else:
        error_result = {'error': 'Invalid request format'}
        return jsonify(error_result), 400


@main_blueprint.route('/results', methods=['GET'])
def get_all_results():
    results = PredictionResult.query.all()
    response = [{'id': result.id, 'probability': result.probability, 'email': result.email} for result in results]
    return jsonify(response)


@main_blueprint.route('/results/<int:result_id>', methods=['GET'])
def get_result(result_id):
    result = PredictionResult.query.get(result_id)
    if result:
        response = {key: getattr(result, key) for key in result.__dict__.keys() if not key.startswith('_')}
        return jsonify(response)

    else:
        return jsonify({'error': 'Result not found'}), 404
