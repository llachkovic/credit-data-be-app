from flask import Flask

from app.models import db

# Initialize the SQLite database
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mydb.db"
db.init_app(app)
with app.app_context():
    db.create_all()
