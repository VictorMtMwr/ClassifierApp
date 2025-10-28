from flask import Flask
from .config import MAX_CONTENT_LENGTH
from .models_loader import load_models
from .routes.predict import init_routes as init_predict_routes
from .routes.retrain import init_retrain_route

def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

    especies, formas, plantas = load_models()

    app.register_blueprint(init_predict_routes(especies, formas, plantas))
    app.register_blueprint(init_retrain_route())

    return app
