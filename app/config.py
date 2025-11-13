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

PLANTS = [
    False, True
    ]

SHAPES = [
    'Elliptic', 'Imparipinnate', 'Lanceolate', 'Obovate', 'Ovate', 'Palmeate', 'Trifoliate'
    ]

SPECIES = [
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

# Configuración de Cloudflare R2 para descarga de imágenes
# Estas credenciales se pueden obtener desde el dashboard de Cloudflare R2
# Soporta tanto R2_* como CLOUDFLARE_R2_* para compatibilidad
CLOUDFLARE_R2_BUCKET_NAME = os.getenv('R2_BUCKET_NAME') or os.getenv('CLOUDFLARE_R2_BUCKET_NAME')
CLOUDFLARE_R2_ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID') or os.getenv('CLOUDFLARE_R2_ACCOUNT_ID')
CLOUDFLARE_R2_ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID') or os.getenv('CLOUDFLARE_R2_ACCESS_KEY_ID')
CLOUDFLARE_R2_SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY') or os.getenv('CLOUDFLARE_R2_SECRET_ACCESS_KEY')
CLOUDFLARE_R2_PREFIX = os.getenv('R2_PREFIX') or os.getenv('CLOUDFLARE_R2_PREFIX', '')  # Prefijo opcional para filtrar archivos
R2_PUBLIC_BASE_URL = os.getenv('R2_PUBLIC_BASE_URL', None)  # URL pública del bucket (opcional)
