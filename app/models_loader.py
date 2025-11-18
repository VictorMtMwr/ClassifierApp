import tensorflow as tf
import os
import threading
import glob
import shutil
from .config import MODEL_DIR, BACKUP_DIR
from .utils.model_versioning import get_model_version_path

# Almacenamiento global de modelos con lock para thread-safety
_models = {
    'especies': None,
    'formas': None,
    'plantas': None
}
_models_lock = threading.Lock()


def find_model_path(model_name: str) -> str:
    """
    Busca el último modelo versionado y asegura que esté en MODEL_DIR.
    Si está en BACKUP_DIR, lo copia a MODEL_DIR antes de cargarlo.
    Solo busca modelos versionados (con patrón _v*_*.h5).
    
    Prioridad de búsqueda:
    1. Modelo versionado más reciente en MODEL_DIR (si existe, lo usa directamente)
    2. Último modelo versionado usando el sistema de versionado (si está en MODEL_DIR)
    3. Si no está en MODEL_DIR, busca en BACKUP_DIR y lo copia a MODEL_DIR
    
    Args:
        model_name: 'especies', 'formas' o 'plantas'
    
    Returns:
        Ruta al archivo del modelo versionado más reciente en MODEL_DIR
    
    Raises:
        FileNotFoundError: Si no se encuentra ningún modelo versionado
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    model_file_map = {
        'especies': 'modelo_especies.h5',
        'formas': 'modelo_hojas.h5',
        'plantas': 'modelo_plantas.h5'
    }
    
    model_base_name = os.path.splitext(model_file_map[model_name])[0]  # 'modelo_especies', 'modelo_hojas', etc.
    
    # Función auxiliar para ordenar por timestamp (más reciente primero)
    def get_timestamp_from_filename(filename):
        """Extrae el timestamp del nombre del archivo para ordenar"""
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) >= 3:
            # El timestamp está en la última parte antes de .h5
            timestamp = parts[-1].replace('.h5', '')
            return timestamp
        return ''
    
    # Primero buscar en MODEL_DIR directamente
    pattern_model = os.path.join(MODEL_DIR, f"{model_base_name}_v*_*.h5")
    versioned_models = glob.glob(pattern_model)
    if versioned_models:
        versioned_models.sort(key=get_timestamp_from_filename, reverse=True)
        model_path_in_models = versioned_models[0]
        print(f"✅ Modelo {model_name} versionado encontrado en {MODEL_DIR}: {os.path.basename(model_path_in_models)}")
        return model_path_in_models
    
    # Si no está en MODEL_DIR, buscar en BACKUP_DIR
    source_path = None
    source_description = None
    
    # 1. Intentar obtener último modelo versionado usando el sistema de versionado
    version_path = get_model_version_path(model_name, version=None)
    if version_path and os.path.exists(version_path):
        # Verificar si está en MODEL_DIR o BACKUP_DIR
        version_path_abs = os.path.abspath(version_path)
        model_dir_abs = os.path.abspath(MODEL_DIR)
        version_dir = os.path.dirname(version_path_abs)
        
        if os.path.normpath(version_dir) == os.path.normpath(model_dir_abs):
            print(f"✅ Modelo {model_name} versionado encontrado en {MODEL_DIR}: {os.path.basename(version_path)}")
            return version_path
        else:
            # Está en BACKUP_DIR, necesitamos copiarlo
            source_path = version_path
            source_description = "último modelo versionado (sistema de versionado)"
    
    # 2. Si no hay metadatos JSON o no se encontró, buscar directamente en BACKUP_DIR
    if source_path is None:
        pattern_backup = os.path.join(BACKUP_DIR, f"{model_base_name}_v*_*.h5")
        versioned_backups = glob.glob(pattern_backup)
        if versioned_backups:
            versioned_backups.sort(key=get_timestamp_from_filename, reverse=True)
            source_path = versioned_backups[0]
            source_description = "modelo versionado más reciente"
    
    # Si se encontró un modelo en BACKUP_DIR, copiarlo a MODEL_DIR
    if source_path and os.path.exists(source_path):
        # Asegurar que MODEL_DIR existe
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Obtener el nombre del archivo versionado
        versioned_filename = os.path.basename(source_path)
        target_path = os.path.join(MODEL_DIR, versioned_filename)
        
        print(f"📋 Modelo {model_name} versionado no encontrado en {MODEL_DIR}")
        print(f"   Encontrado ({source_description}) en {BACKUP_DIR}: {versioned_filename}")
        print(f"   Copiando a {MODEL_DIR}...")
        
        # Copiar el modelo versionado a MODEL_DIR
        shutil.copy2(source_path, target_path)
        
        print(f"✅ Modelo {model_name} versionado copiado exitosamente a {MODEL_DIR}: {versioned_filename}")
        return target_path
    
    # Si no se encuentra nada, lanzar error
    raise FileNotFoundError(
        f"No se encontró ningún modelo versionado para {model_name}. Buscado en:\n"
        f"  - Modelos versionados en {MODEL_DIR}\n"
        f"  - Modelos versionados en {BACKUP_DIR}"
    )


def load_models():
    """Carga todos los modelos y los almacena en el sistema global"""
    print("Cargando modelos...")
    # Cargar modelos de inferencia en CPU para evitar ocupar GPU
    with tf.device('/CPU:0'):
        especies_path = find_model_path('especies')
        formas_path = find_model_path('formas')
        plantas_path = find_model_path('plantas')
        
        especies = tf.keras.models.load_model(especies_path)
        formas = tf.keras.models.load_model(formas_path)
        plantas = tf.keras.models.load_model(plantas_path)
    
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
    
    # Usar la función find_model_path para buscar el modelo
    model_path = find_model_path(model_name)
    
    print(f"Recargando modelo {model_name} desde {model_path}...")
    
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
