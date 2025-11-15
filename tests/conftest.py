"""
Configuración y fixtures compartidas para las pruebas
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import tensorflow as tf
from fastapi.testclient import TestClient

# Configurar variables de entorno de prueba antes de importar la app
os.environ['TESTING'] = '1'


@pytest.fixture
def temp_dir():
    """Crea un directorio temporal para pruebas"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_model():
    """Crea un modelo mock de TensorFlow"""
    model = MagicMock()
    # Simular predicción: retorna un tensor con probabilidades
    mock_output = tf.constant([[0.1, 0.2, 0.7]], dtype=tf.float32)
    model.return_value = mock_output
    model.__call__ = MagicMock(return_value=mock_output)
    return model


@pytest.fixture
def mock_models(mock_model):
    """Crea tres modelos mock"""
    return mock_model, mock_model, mock_model


@pytest.fixture
def sample_image_bytes():
    """Crea bytes de una imagen de prueba (1x1 PNG)"""
    # PNG mínimo válido (1x1 píxel rojo)
    return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'


@pytest.fixture
def mock_image_file(sample_image_bytes):
    """Crea un archivo de imagen mock para FastAPI"""
    from io import BytesIO
    from fastapi import UploadFile
    
    file = BytesIO(sample_image_bytes)
    return UploadFile(
        file=file,
        filename="test_image.png",
        headers={"content-type": "image/png"}
    )


@pytest.fixture
def app_with_mocked_models(mock_models):
    """Crea una instancia de la app con modelos mockeados"""
    with patch('app.models_loader.load_models', return_value=mock_models):
        from app import create_app
        app = create_app()
        # También mockear get_models para que devuelva los modelos mock
        with patch('app.models_loader.get_models', return_value=mock_models):
            yield app


@pytest.fixture
def client(app_with_mocked_models):
    """Cliente de prueba para la API"""
    return TestClient(app_with_mocked_models)


@pytest.fixture
def mock_backup_dir(temp_dir):
    """Crea un directorio de backups temporal"""
    backup_dir = os.path.join(temp_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    return backup_dir


@pytest.fixture
def mock_model_dir(temp_dir):
    """Crea un directorio de modelos temporal"""
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


@pytest.fixture
def sample_model_file(mock_model_dir):
    """Crea un archivo de modelo de prueba"""
    # Crear un modelo simple de TensorFlow y guardarlo
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='softmax', input_shape=(128, 128, 3))
    ])
    model_path = os.path.join(mock_model_dir, "modelo_especies.h5")
    model.save(model_path)
    return model_path


@pytest.fixture
def mock_config(temp_dir, mock_model_dir, mock_backup_dir):
    """Mock de configuración con directorios temporales"""
    with patch('app.config.MODEL_DIR', mock_model_dir), \
         patch('app.config.BACKUP_DIR', mock_backup_dir), \
         patch('app.config.DATA_DIR', os.path.join(temp_dir, "data")):
        yield {
            'MODEL_DIR': mock_model_dir,
            'BACKUP_DIR': mock_backup_dir,
            'DATA_DIR': os.path.join(temp_dir, "data")
        }

