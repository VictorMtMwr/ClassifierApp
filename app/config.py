import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')
BACKUP_DIR = os.path.join(BASE_DIR, '..', 'backups')
MAX_BACKUPS = 3

MAX_CONTENT_LENGTH = 20 * 1024 * 1024
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

PLANTS = [False, True]
SHAPES = [
    'Eliptica',
    'Imparipinnada',
    'Lanceolada',
    'Obovada',
    'Ovada',
    'Palmeada',
    'Trifoliada',
    'Elíptica',
    'Redonda'
]
SPECIES = [
    'cassava_deseased',
    'cassava_healthy',
    'corn_deseased',
    'corn_healthy',
    'cucumber_deseased',
    'cucumber_healthy',
    'eggplant_deseased',
    'eggplant_healthy',
    'yam_deseased',
    'yam_healthy',
    'tomato_deseased',
    'tomato_healthy'
]
