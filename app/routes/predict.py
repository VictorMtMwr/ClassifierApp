from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import os
from ..preprocess import preprocess_image
from ..utils.locks import predict_lock, safe_predict
from ..config import SPECIES, SHAPES, PLANTS  # Estas son variables estáticas que se actualizan dinámicamente
from ..models_loader import get_models

def init_routes(especies, formas, plantas):
    bp = APIRouter(prefix="/predict", tags=["predict"])

    @bp.post(
        "",
        summary="Clasificar imagen",
        description="""
        Clasifica una imagen usando tres modelos de aprendizaje profundo:
        - **Modelo 1 (Especies)**: Clasifica la especie de la planta y su estado (deseased/healthy)
        - **Modelo 2 (Formas)**: Clasifica la forma de la hoja
        - **Modelo 3 (Plantas)**: Clasifica si es una planta o no
        
        Los modelos se cargan dinámicamente, por lo que siempre usan las versiones más recientes
        (incluso después de reentrenar sin reiniciar la aplicación).
        
        **Notas:**
        - Las predicciones se ejecutan en CPU para no interferir con el entrenamiento en GPU
        - Múltiples predicciones pueden ejecutarse simultáneamente
        - Las imágenes se redimensionan automáticamente a 128x128 píxeles
        """,
        response_description="Predicciones de los tres modelos con probabilidades"
    )
    async def predict(image: UploadFile = File(..., description="Archivo de imagen a clasificar (JPG, PNG, etc.)")):
        # Obtener los modelos actuales (pueden haber sido recargados)
        especies, formas, plantas = get_models()
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail='El archivo debe ser una imagen')
        
        image_bytes = await image.read()
        input_data = preprocess_image(image_bytes)

        # Usar lock para evitar conflictos, pero permitir ejecución concurrente
        # Las predicciones se ejecutan en CPU sin interferir con GPU
        with predict_lock:
            # Forzar ejecución en CPU para predicciones
            # Esto evita que las predicciones interfieran con el entrenamiento en GPU
            with tf.device('/CPU:0'):
                pred1 = safe_predict(especies, input_data)
                if isinstance(pred1, list):
                    pred1 = pred1[0]
                pred1 = pred1.numpy()
                # Asegurar que pred1 sea un array 1D
                if pred1.ndim > 1:
                    pred1 = pred1.flatten()
                
                pred2 = safe_predict(formas, input_data)
                if isinstance(pred2, list):
                    pred2 = pred2[0]
                pred2 = pred2.numpy()
                # Asegurar que pred2 sea un array 1D
                if pred2.ndim > 1:
                    pred2 = pred2.flatten()
                
                pred3 = safe_predict(plantas, input_data)
                if isinstance(pred3, list):
                    pred3 = pred3[0]
                pred3 = pred3.numpy()
                # Asegurar que pred3 sea un array 1D
                if pred3.ndim > 1:
                    pred3 = pred3.flatten()

        # Obtener índices de las clases predichas
        idx1 = int(np.argmax(pred1))
        idx2 = int(np.argmax(pred2))
        idx3 = int(np.argmax(pred3))
        
        # Obtener clases dinámicamente del modelo cargado (lee desde config.py actualizado)
        from ..config import get_classes_from_model
        species_classes = get_classes_from_model('especies')
        shapes_classes = get_classes_from_model('formas')
        plants_classes = get_classes_from_model('plantas')
        
        # Validar que los índices estén dentro del rango
        if idx1 >= len(species_classes):
            raise HTTPException(status_code=500, detail=f'Índice de clase fuera de rango: {idx1} >= {len(species_classes)}')
        if idx2 >= len(shapes_classes):
            raise HTTPException(status_code=500, detail=f'Índice de clase fuera de rango: {idx2} >= {len(shapes_classes)}')
        if idx3 >= len(plants_classes):
            raise HTTPException(status_code=500, detail=f'Índice de clase fuera de rango: {idx3} >= {len(plants_classes)}')

        result = {
            'model1': {
                'class': idx1,
                'class_name': species_classes[idx1],
                'probability': float(np.max(pred1)),
            },
            'model2': {
                'class': idx2,
                'class_name': shapes_classes[idx2],
                'probability': float(np.max(pred2)),
            },
            'model3': {
                'class': idx3,
                'class_name': str(plants_classes[idx3]),
                'probability': float(np.max(pred3)),
            }
        }
        return result
    
    return bp
