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
                
                pred2 = safe_predict(formas, input_data)
                if isinstance(pred2, list):
                    pred2 = pred2[0]
                pred2 = pred2.numpy()
                
                pred3 = safe_predict(plantas, input_data)
                if isinstance(pred3, list):
                    pred3 = pred3[0]
                pred3 = pred3.numpy()

        result = {
            'model1': {
                'class': int(np.argmax(pred1)),
                'class_name': SPECIES[int(np.argmax(pred1))],
                'probability': float(np.max(pred1))
            },
            'model2': {
                'class': int(np.argmax(pred2)),
                'class_name': SHAPES[int(np.argmax(pred2))],
                'probability': float(np.max(pred2))
            },
            'model3': {
                'class': int(np.argmax(pred3)),
                'class_name': PLANTS[int(np.argmax(pred3))],
                'probability': float(np.max(pred3))
            }
        }
        return result
    
    return bp
