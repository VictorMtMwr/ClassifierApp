import tensorflow as tf
import os
import threading
from .config import MODEL_DIR
from .utils.model_versioning import get_model_version_path

# Almacenamiento global de modelos con lock para thread-safety
_models = {
    'especies': None,
    'formas': None,
    'plantas': None
}
_models_lock = threading.Lock()

def load_models():
    """Carga todos los modelos y los almacena en el sistema global"""
    print("Cargando modelos...")
    # Cargar modelos de inferencia en CPU para evitar ocupar GPU
    with tf.device('/CPU:0'):
        especies = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_especies.h5"))
        formas = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_hojas.h5"))
        plantas = tf.keras.models.load_model(os.path.join(MODEL_DIR, "modelo_plantas.h5"))
    
    # Almacenar en el sistema global
    with _models_lock:
        _models['especies'] = especies
        _models['formas'] = formas
        _models['plantas'] = plantas
    
    print("Modelos cargados correctamente.")
    return especies, formas, plantas

def get_models():
    """Obtiene los modelos actuales del sistema global"""
    with _models_lock:
        return _models['especies'], _models['formas'], _models['plantas']

def reload_model(model_name):
    """
    Recarga un modelo específico después de reentrenamiento
    
    Args:
        model_name: 'especies', 'formas' o 'plantas'
    
    Returns:
        El modelo recargado
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}. Debe ser 'especies', 'formas' o 'plantas'")
    
    model_file_map = {
        'especies': 'modelo_especies.h5',
        'formas': 'modelo_hojas.h5',
        'plantas': 'modelo_plantas.h5'
    }
    
    model_path = os.path.join(MODEL_DIR, model_file_map[model_name])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    
    print(f"Recargando modelo {model_name}...")
    
    # Cargar modelo en CPU para inferencia
    with tf.device('/CPU:0'):
        new_model = tf.keras.models.load_model(model_path)
    
    # Actualizar en el sistema global
    with _models_lock:
        # Limpiar el modelo anterior de memoria
        old_model = _models[model_name]
        if old_model is not None:
            del old_model
            # Forzar garbage collection para liberar memoria
            import gc
            gc.collect()
        
        _models[model_name] = new_model
    
    print(f"Modelo {model_name} recargado correctamente.")
    return new_model

def reload_all_models():
    """Recarga todos los modelos"""
    print("Recargando todos los modelos...")
    especies = reload_model('especies')
    formas = reload_model('formas')
    plantas = reload_model('plantas')
    print("Todos los modelos recargados correctamente.")
    return especies, formas, plantas

def load_model_version(model_name: str, version: int = None):
    """
    Carga una versión específica de un modelo desde backups
    
    Args:
        model_name: 'especies', 'formas' o 'plantas'
        version: Número de versión (None para la más reciente)
    
    Returns:
        El modelo cargado
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}. Debe ser 'especies', 'formas' o 'plantas'")
    
    version_path = get_model_version_path(model_name, version)
    
    if version_path is None:
        if version is None:
            raise FileNotFoundError(f"No se encontraron versiones del modelo {model_name}")
        else:
            raise FileNotFoundError(f"No se encontró la versión {version} del modelo {model_name}")
    
    print(f"Cargando versión {version or 'más reciente'} del modelo {model_name} desde {version_path}...")
    
    # Cargar modelo en CPU para inferencia
    with tf.device('/CPU:0'):
        model = tf.keras.models.load_model(version_path)
    
    print(f"Modelo {model_name} (versión {version or 'más reciente'}) cargado correctamente.")
    return model
