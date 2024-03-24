from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    probability = db.Column(db.Float)
