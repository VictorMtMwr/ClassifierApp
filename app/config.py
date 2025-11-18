import os
import tensorflow as tf
from pathlib import Path

# Cargar variables de entorno desde .env si existe
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Variables de entorno cargadas desde {env_path}")
except ImportError:
    # python-dotenv no está instalado, usar variables de entorno del sistema
    pass

# Configurar TensorFlow para permitir operaciones concurrentes
# Esto evita que las predicciones interfieran con el entrenamiento
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Permitir que TensorFlow use múltiples threads sin bloquear
        # Configurar threads para permitir operaciones concurrentes entre CPU y GPU
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)
        # Asegurar que TensorFlow use GPU por defecto
        print(f"✅ TensorFlow configurado con {len(gpus)} GPU(s) disponible(s)")
    except RuntimeError as e:
        print(f"Error configurando GPU: {e}")
else:
    print("⚠️  No se detectaron GPUs. TensorFlow usará CPU.")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')
BACKUP_DIR = os.path.join(BASE_DIR, '..', 'backups')
MAX_BACKUPS = 3

MAX_CONTENT_LENGTH = 20 * 1024 * 1024
HOST = '0.0.0.0'
PORT = 8000
DEBUG = True

# Clases por defecto (se usan solo si no hay modelos cargados)
_DEFAULT_PLANTS = [False, True]

_DEFAULT_SHAPES = [
    'Elliptic', 'Imparipinnate', 'Lanceolate', 'Obovate', 'Ovate', 'Palmeate', 'Trifoliate'
]

_DEFAULT_SPECIES = [
    'cucumis-sativus_deseased',
    'cucumis-sativus_healthy',
    'discorea-alata-l_deseased',
    'discorea-alata-l_healthy',
    'manihot-esculenta_deseased',
    'manihot-esculenta_healthy',
    'solanum-melongena_deseased',
    'solanum-melongena_healthy',
    'zea-mays_deseased',
    'zea-mays_healthy'
]

# Backup de clases para restauración en caso de error en reentrenamiento
_CLASSES_BACKUP = {
    'SPECIES': None,
    'SHAPES': None,
    'PLANTS': None
}

def get_classes_from_model(model_name: str):
    """
    Obtiene las clases dinámicamente del modelo cargado.
    Primero intenta leer desde config.py (que se actualiza dinámicamente),
    luego valida con el número de clases del modelo.
    
    Args:
        model_name: 'especies', 'formas' o 'plantas'
    
    Returns:
        list: Lista de clases del modelo
    """
    try:
        from .models_loader import get_models
        from .utils.label_detector import detect_classes_in_data
        import ast
        
        especies_model, formas_model, plantas_model = get_models()
        
        # Primero intentar leer desde las variables estáticas del módulo (más rápido y confiable)
        # Estas variables se actualizan dinámicamente mediante update_config_with_detected_classes
        # Después de reload_config(), estas variables se actualizan automáticamente
        config_classes = None
        try:
            if model_name == 'especies':
                config_classes = SPECIES.copy() if hasattr(SPECIES, 'copy') else list(SPECIES)  # Variable estática actualizada dinámicamente
            elif model_name == 'formas':
                config_classes = SHAPES.copy() if hasattr(SHAPES, 'copy') else list(SHAPES)  # Variable estática actualizada dinámicamente
            elif model_name == 'plantas':
                config_classes = PLANTS.copy() if hasattr(PLANTS, 'copy') else list(PLANTS)  # Variable estática actualizada dinámicamente
        except Exception as e:
            # Si falla al leer desde el módulo, continuar con lectura desde archivo
            pass
        
        # Si no se pudo leer desde el módulo, leer desde archivo config.py
        if config_classes is None:
            config_path = os.path.join(BASE_DIR, 'config.py')
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_content = f.read()
                
                import re
                if model_name == 'especies':
                    # Buscar SPECIES = [...] (la variable estática, no la función)
                    match = re.search(r'^SPECIES\s*=\s*\[(.*?)\]', config_content, re.DOTALL | re.MULTILINE)
                    if not match:
                        # Intentar buscar en cualquier lugar (fallback)
                        match = re.search(r'SPECIES\s*=\s*\[(.*?)\]', config_content, re.DOTALL)
                    if match:
                        try:
                            # Evaluar la lista de forma segura
                            config_classes = ast.literal_eval('[' + match.group(1) + ']')
                        except:
                            pass
                
                elif model_name == 'formas':
                    match = re.search(r'^SHAPES\s*=\s*\[(.*?)\]', config_content, re.DOTALL | re.MULTILINE)
                    if not match:
                        match = re.search(r'SHAPES\s*=\s*\[(.*?)\]', config_content, re.DOTALL)
                    if match:
                        try:
                            config_classes = ast.literal_eval('[' + match.group(1) + ']')
                        except:
                            pass
                
                elif model_name == 'plantas':
                    match = re.search(r'^PLANTS\s*=\s*\[(.*?)\]', config_content, re.DOTALL | re.MULTILINE)
                    if not match:
                        match = re.search(r'PLANTS\s*=\s*\[(.*?)\]', config_content, re.DOTALL)
                    if match:
                        try:
                            config_classes = ast.literal_eval('[' + match.group(1) + ']')
                        except:
                            pass
            except Exception as e:
                print(f"⚠️  Error leyendo config.py: {e}")
        
        # Obtener número de clases del modelo
        num_classes = None
        if model_name == 'especies' and especies_model is not None:
            if isinstance(especies_model.output, list):
                num_classes = especies_model.output[0].shape[-1]
            else:
                num_classes = especies_model.output.shape[-1]
        elif model_name == 'formas' and formas_model is not None:
            if isinstance(formas_model.output, list):
                num_classes = formas_model.output[0].shape[-1]
            else:
                num_classes = formas_model.output.shape[-1]
        elif model_name == 'plantas' and plantas_model is not None:
            if isinstance(plantas_model.output, list):
                num_classes = plantas_model.output[0].shape[-1]
            else:
                num_classes = plantas_model.output.shape[-1]
        
        # Validar que las clases de config.py coinciden con el modelo
        if config_classes is not None and num_classes is not None:
            if len(config_classes) == num_classes:
                return config_classes
            else:
                print(f"⚠️  Advertencia: config.py tiene {len(config_classes)} clases pero el modelo tiene {num_classes}")
        
        # Si no coincide o no se pudo leer, obtener desde los datos y validar
        if model_name == 'especies':
            data_path = os.path.join(DATA_DIR, 'especies')
            classes = detect_classes_in_data(data_path)
            if classes and num_classes is not None:
                if len(classes) == num_classes:
                    return sorted(classes)
            # Fallback a clases por defecto
            if num_classes and num_classes <= len(_DEFAULT_SPECIES):
                return _DEFAULT_SPECIES[:num_classes]
            return _DEFAULT_SPECIES.copy() if config_classes is None else config_classes
        
        elif model_name == 'formas':
            data_path = os.path.join(DATA_DIR, 'hojas')
            classes = detect_classes_in_data(data_path)
            if classes and num_classes is not None:
                if len(classes) == num_classes:
                    return sorted(classes)
            if num_classes and num_classes <= len(_DEFAULT_SHAPES):
                return _DEFAULT_SHAPES[:num_classes]
            return _DEFAULT_SHAPES.copy() if config_classes is None else config_classes
        
        elif model_name == 'plantas':
            if num_classes and num_classes <= len(_DEFAULT_PLANTS):
                return _DEFAULT_PLANTS[:num_classes]
            return _DEFAULT_PLANTS.copy() if config_classes is None else config_classes
        
    except Exception as e:
        print(f"⚠️  Error obteniendo clases del modelo {model_name}: {e}")
    
    # Fallback final a clases por defecto
    if model_name == 'especies':
        return _DEFAULT_SPECIES.copy()
    elif model_name == 'formas':
        return _DEFAULT_SHAPES.copy()
    elif model_name == 'plantas':
        return _DEFAULT_PLANTS.copy()
    return []


def get_SPECIES():
    """Obtiene dinámicamente las clases de especies del modelo cargado"""
    return get_classes_from_model('especies')


def get_SHAPES():
    """Obtiene dinámicamente las clases de formas del modelo cargado"""
    return get_classes_from_model('formas')


def get_PLANTS():
    """Obtiene dinámicamente las clases de plantas del modelo cargado"""
    return get_classes_from_model('plantas')


def backup_classes():
    """Hace backup de las clases actuales antes de actualizar config.py"""
    try:
        _CLASSES_BACKUP['SPECIES'] = get_SPECIES().copy()
        _CLASSES_BACKUP['SHAPES'] = get_SHAPES().copy()
        _CLASSES_BACKUP['PLANTS'] = get_PLANTS().copy()
        print(f"✅ Backup de clases realizado: SPECIES={len(_CLASSES_BACKUP['SPECIES'])}, SHAPES={len(_CLASSES_BACKUP['SHAPES'])}, PLANTS={len(_CLASSES_BACKUP['PLANTS'])}")
        return True
    except Exception as e:
        print(f"⚠️  Error haciendo backup de clases: {e}")
        return False


def restore_classes():
    """Restaura las clases desde el backup si existe"""
    if _CLASSES_BACKUP['SPECIES'] is None:
        print("ℹ️  No hay backup de clases para restaurar")
        return False
    
    try:
        from .utils.label_detector import update_config_with_detected_classes
        
        if _CLASSES_BACKUP['SPECIES']:
            update_config_with_detected_classes('especies', _CLASSES_BACKUP['SPECIES'])
        if _CLASSES_BACKUP['SHAPES']:
            update_config_with_detected_classes('hojas', _CLASSES_BACKUP['SHAPES'])
        if _CLASSES_BACKUP['PLANTS']:
            update_config_with_detected_classes('plantas', _CLASSES_BACKUP['PLANTS'])
        
        from .utils.label_detector import reload_config
        reload_config()
        
        print(f"✅ Clases restauradas desde backup: SPECIES={len(_CLASSES_BACKUP['SPECIES'])}, SHAPES={len(_CLASSES_BACKUP['SHAPES'])}, PLANTS={len(_CLASSES_BACKUP['PLANTS'])}")
        return True
    except Exception as e:
        print(f"⚠️  Error restaurando clases: {e}")
        return False


def clear_classes_backup():
    """Limpia el backup de clases después de actualización exitosa"""
    _CLASSES_BACKUP['SPECIES'] = None
    _CLASSES_BACKUP['SHAPES'] = None
    _CLASSES_BACKUP['PLANTS'] = None


# Variables estáticas que se actualizan dinámicamente mediante update_config_with_detected_classes
# Las funciones SPECIES(), SHAPES(), PLANTS() leen estas variables dinámicamente
# Estas son las clases iniciales/por defecto que se usan antes de entrenar
SPECIES = _DEFAULT_SPECIES.copy()
SHAPES = _DEFAULT_SHAPES.copy()
PLANTS = _DEFAULT_PLANTS.copy()


# Configuración de Cloudflare R2 para descarga de imágenes
# Estas credenciales se pueden obtener desde el dashboard de Cloudflare R2
# Soporta tanto R2_* como CLOUDFLARE_R2_* para compatibilidad
CLOUDFLARE_R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME') or os.getenv('CLOUDFLARE_R2_BUCKET_NAME')
CLOUDFLARE_R2_ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID') or os.getenv('CLOUDFLARE_R2_ACCOUNT_ID')
CLOUDFLARE_R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID') or os.getenv('CLOUDFLARE_R2_ACCESS_KEY_ID')
CLOUDFLARE_R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY') or os.getenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
CLOUDFLARE_R2_PREFIX = os.getenv('R2_PREFIX') or os.getenv('CLOUDFLARE_R2_PREFIX', '')  # Prefijo opcional para filtrar archivos
R2_PUBLIC_BASE_URL = os.getenv('R2_PUBLIC_BASE_URL', None)  # URL pública del bucket (opcional)
