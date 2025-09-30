from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import threading

app = Flask(__name__)


especies = tf.keras.models.load_model("/home/ubuntu/Proyecto/ClassifierApp/modelo_especies.h5")
formas = tf.keras.models.load_model("/home/ubuntu/Proyecto/ClassifierApp/modelo_hojas.h5")
plantas = tf.keras.models.load_model("/home/ubuntu/Proyecto/ClassifierApp/modelo_plantas.h5")


PLANTS = [
    False, True
    ]

SHAPES = [
    'Eliptica', 'Imparipinnada', 'Lanceolada', 'Obovada', 'Ovada', 'Palmeada', 'Trifoliada'
    ]

SPECIES = [
    'yam_healthy', 'yam_deseased',
    'eggplant_healthy', 'eggplant_deseased',
    'cucumber_healthy', 'cucumber_deseased',
    'corn_healthy', 'corn_deseased',
    'cassava_healthy', 'cassava_deseased'
]


def preprocess_image(image_bytes, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

predict_lock = threading.Lock()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    input_data = preprocess_image(image_bytes)

    with predict_lock:
        pred1 = especies.predict(input_data)[0]
        pred2 = formas.predict(input_data)[0]
        pred3 = plantas.predict(input_data)[0]

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

if __name__ == '__main__':
    app.run(debug=True)