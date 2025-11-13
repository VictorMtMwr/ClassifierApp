from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import MAX_CONTENT_LENGTH
from .models_loader import load_models
from .routes.predict import init_routes as init_predict_routes
from .routes.retrain import init_retrain_route

def create_app():
    app = FastAPI(
        title="ClassifierApp API",
        version="1.0.0",
        description="""
        API REST para clasificación de plantas usando modelos de aprendizaje profundo.
        
        ## Características
        
        - **Tres modelos de clasificación**: Especies, Formas de hojas, y Plantas
        - **Reentrenamiento dinámico**: Reentrena modelos sin reiniciar la aplicación
        - **Sistema de versionado**: Gestión automática de versiones de modelos con backups
        - **Recarga automática**: Los modelos reentrenados quedan disponibles inmediatamente
        - **Soporte GPU/CPU**: Entrenamiento en GPU, predicciones en CPU
        
        ## Endpoints principales
        
        - **POST /predict**: Clasificar una imagen usando los tres modelos
        - **POST /retrain**: Reentrenar un modelo específico
        - **GET /retrain/versions**: Listar versiones disponibles de un modelo
        - **POST /retrain/restore-version**: Restaurar una versión específica de un modelo
        
        ## Sistema de versionado
        
        Cada vez que se reentrena un modelo, se crea automáticamente una versión del modelo anterior.
        Las versiones se guardan en `backups/` con numeración secuencial y timestamps.
        Se mantienen automáticamente las 3 versiones más recientes.
        """,
        contact={
            "name": "ClassifierApp",
        }
    )
    
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
