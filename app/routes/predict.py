from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
import os
from ..preprocess import preprocess_image
from ..utils.locks import predict_lock, safe_predict
from ..config import SPECIES, SHAPES, PLANTS

def init_routes(especies, formas, plantas):
    bp = APIRouter(prefix="/predict", tags=["predict"])

    @bp.post("")
    async def predict(image: UploadFile = File(...)):
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
        
        # Validar que los índices estén dentro del rango
        if idx1 >= len(SPECIES):
            raise HTTPException(status_code=500, detail=f'Índice de clase fuera de rango: {idx1} >= {len(SPECIES)}')
        if idx2 >= len(SHAPES):
            raise HTTPException(status_code=500, detail=f'Índice de clase fuera de rango: {idx2} >= {len(SHAPES)}')
        if idx3 >= len(PLANTS):
            raise HTTPException(status_code=500, detail=f'Índice de clase fuera de rango: {idx3} >= {len(PLANTS)}')

        result = {
            'model1': {
                'class': idx1,
                'class_name': SPECIES[idx1],
                'probability': float(np.max(pred1)),
                'all_probabilities': {SPECIES[i]: float(pred1[i]) for i in range(len(SPECIES))} if len(pred1) == len(SPECIES) else {}
            },
            'model2': {
                'class': idx2,
                'class_name': SHAPES[idx2],
                'probability': float(np.max(pred2)),
                'all_probabilities': {SHAPES[i]: float(pred2[i]) for i in range(len(SHAPES))} if len(pred2) == len(SHAPES) else {}
            },
            'model3': {
                'class': idx3,
                'class_name': str(PLANTS[idx3]),
                'probability': float(np.max(pred3)),
                'all_probabilities': {str(PLANTS[i]): float(pred3[i]) for i in range(len(PLANTS))} if len(pred3) == len(PLANTS) else {}
            }
        }
        return result
    
    return bp
