import threading
import tensorflow as tf

predict_lock = threading.Lock()

@tf.function
def safe_predict(model, input_data):
    return model(input_data, training=False)
