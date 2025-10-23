from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import threading
import os

# ================================
# CONFIGURACIÓN DEL SERVIDOR
# ================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB máx

# ================================
# CARGA DE MODELOS
# ================================
# Usa rutas absolutas válidas en tu sistema (Windows o Linux)
MODEL_DIR = r"C:\Users\victo\Desktop\ClassifierApp"

especies = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_especies.h5"))
formas = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_hojas.h5"))
plantas = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_plantas.h5"))

# ================================
# LISTAS DE CLASES
# ================================
PLANTS = [
    False, True
]

SHAPES = [
    'Eliptica', 'Imparipinnada', 'Lanceolada', 'Obovada', 'Ovada', 'Palmeada', 'Trifoliada'
]

SPECIES = [
    'cassava_deseased', 'cassava_healthy',
    'corn_deseased', 'corn_healthy',
    'cucumber_deseased', 'cucumber_healthy',
    'eggplant_deseased', 'eggplant_healthy',
    'yam_deseased', 'yam_healthy'
]

# ================================
# FUNCIÓN DE PREPROCESAMIENTO
# ================================
def preprocess_image(image_bytes, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ================================
# BLOQUEO DE PREDICCIÓN (THREAD-SAFE)
# ================================
predict_lock = threading.Lock()

@tf.function
def safe_predict(model, input_data):
    """Ejecución segura para modelos Keras/TensorFlow"""
    return model(input_data, training=False)

# ================================
# ENDPOINT DE PREDICCIÓN
# ================================
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    input_data = preprocess_image(image_bytes)

    with predict_lock:
        pred1 = safe_predict(especies, input_data)[0].numpy()
        pred2 = safe_predict(formas, input_data)[0].numpy()
        pred3 = safe_predict(plantas, input_data)[0].numpy()

    species_idx = int(np.argmax(pred1))
    shape_idx = int(np.argmax(pred2))
    plant_idx = int(np.argmax(pred3))

    result = {
        'model1': {
            'class': species_idx,
            'class_name': SPECIES[species_idx],
            'probability': float(np.max(pred1))
        },
        'model2': {
            'class': shape_idx,
            'class_name': SHAPES[shape_idx],
            'probability': float(np.max(pred2))
        },
        'model3': {
            'class': plant_idx,
            'class_name': PLANTS[plant_idx],
            'probability': float(np.max(pred3))
        }
    }

    return jsonify(result)

# ================================
# INICIO DEL SERVIDOR
# ================================
if __name__ == '__main__':
    # host='0.0.0.0' permite acceso desde otras PCs en la red
    app.run(host='0.0.0.0', port=5000, debug=True)
