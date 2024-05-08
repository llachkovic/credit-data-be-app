from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    probability = db.Column(db.Float)
    age = db.Column(db.Integer)
    checking_account_status = db.Column(db.String)
    credit_amount = db.Column(db.Integer)
    credit_history = db.Column(db.String)
    duration_in_months = db.Column(db.Integer)
    employment_duration = db.Column(db.String)
    foreign_worker = db.Column(db.String)
    housing = db.Column(db.String)
    installment_rate = db.Column(db.Integer)
    job = db.Column(db.String)
    number_of_existing_credits = db.Column(db.Integer)
    number_of_people_liable_for_maintenance = db.Column(db.Integer)
    other_debtors = db.Column(db.String)
    other_installment_plans = db.Column(db.String)
    personal_status = db.Column(db.String)
    present_residence_since = db.Column(db.Integer)
    property = db.Column(db.String)
    purpose = db.Column(db.String)
    savings_account = db.Column(db.String)
    telephone = db.Column(db.String)

