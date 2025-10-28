import tensorflow as tf
import os
from .config import MODEL_DIR

def load_models():
    print("Cargando modelos...")
    # Cargar modelos de inferencia en CPU para evitar ocupar GPU
    with tf.device('/CPU:0'):
        especies = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_especies.h5"))
        formas = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_hojas.h5"))
        plantas = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_plantas.h5"))
    print("Modelos cargados correctamente.")
    return especies, formas, plantas
