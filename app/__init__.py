from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import MAX_CONTENT_LENGTH
from .models_loader import load_models
from .routes.predict import init_routes as init_predict_routes
from .routes.retrain import init_retrain_route

def create_app():
    app = FastAPI(title="ClassifierApp API", version="1.0.0")
    
    # Configurar CORS si es necesario
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    especies, formas, plantas = load_models()

    app.include_router(init_predict_routes(especies, formas, plantas))
    app.include_router(init_retrain_route())

    return app
