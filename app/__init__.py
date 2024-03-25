from flask import Flask, g
from flask_cors import CORS

from app.config import Config
from app.models import db
from app.routes import main_blueprint
from classifier import get_model_and_scaler


def create_app():
    app = Flask(__name__)

    CORS(app)

    # Configuration settings
    app.config.from_object(Config)

    # Initialize the database
    db.init_app(app)

    model, scaler = get_model_and_scaler()

    @app.before_request
    def load_model_and_scaler():
        g.model = model
        g.scaler = scaler

    # Register routes
    app.register_blueprint(main_blueprint)

    return app
