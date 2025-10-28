from flask import Blueprint, request, jsonify
import numpy as np
import tensorflow as tf
from ..preprocess import preprocess_image
from ..utils.locks import predict_lock, safe_predict
from ..config import SPECIES, SHAPES, PLANTS

bp = Blueprint('predict', __name__)

def init_routes(especies, formas, plantas):
    @bp.route('/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()
        input_data = preprocess_image(image_bytes)

        with predict_lock:
            # Forzar ejecución en CPU para predicciones
            with tf.device('/CPU:0'):
                pred1 = safe_predict(especies, input_data)[0].numpy()
                pred2 = safe_predict(formas, input_data)[0].numpy()
                pred3 = safe_predict(plantas, input_data)[0].numpy()

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
        return jsonify(result)
    return bp
