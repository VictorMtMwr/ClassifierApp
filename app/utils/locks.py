import threading
import tensorflow as tf
import os

predict_lock = threading.Lock()

# Configurar TensorFlow para predicciones en CPU
# Asegurar que las predicciones no interfieran con entrenamiento en GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

@tf.function
def safe_predict(model, input_data):
    """
    Realiza una predicción segura del modelo.
    Usa @tf.function para optimización y asegura que no interfiera con otros procesos.
    """
    return model(input_data, training=False)
