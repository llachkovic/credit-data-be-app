from flask import Flask
from flask_cors import CORS

from app.config import Config
from app.models import db
from app.routes import main_blueprint
from classifier import get_model


def create_app():
    app = Flask(__name__)

    CORS(app)

    # Configuration settings
    app.config.from_object(Config)

    # Initialize the database
    db.init_app(app)

    model = get_model()

    app.model = model

    # Register routes
    app.register_blueprint(main_blueprint)

    return app
