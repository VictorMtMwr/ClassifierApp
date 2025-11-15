"""
Pruebas unitarias para el endpoint de predicción
"""
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
from io import BytesIO


@pytest.fixture
def mock_prediction_output():
    """Crea una salida de predicción mock"""
    # Simular salidas de los tres modelos
    pred1 = np.array([0.1, 0.2, 0.7])  # Especies
    pred2 = np.array([0.05, 0.15, 0.3, 0.5])  # Formas
    pred3 = np.array([0.2, 0.8])  # Plantas
    return pred1, pred2, pred3


def test_predict_endpoint_success(client, sample_image_bytes, mock_prediction_output):
    """Test que el endpoint de predicción funciona correctamente con una imagen válida"""
    pred1, pred2, pred3 = mock_prediction_output
    
    # Mockear get_models y las predicciones
    with patch('app.routes.predict.get_models') as mock_get_models, \
         patch('app.routes.predict.safe_predict') as mock_safe_predict, \
         patch('app.routes.predict.preprocess_image') as mock_preprocess:
        
        # Configurar mocks
        mock_model = MagicMock()
        mock_get_models.return_value = (mock_model, mock_model, mock_model)
        
        # Simular preprocesamiento
        mock_preprocess.return_value = np.zeros((1, 128, 128, 3))
        
        # Simular predicciones
        mock_safe_predict.side_effect = [
            tf.constant([pred1], dtype=tf.float32),
            tf.constant([pred2], dtype=tf.float32),
            tf.constant([pred3], dtype=tf.float32)
        ]
        
        # Hacer la petición
        response = client.post(
            "/predict",
            files={"image": ("test.png", sample_image_bytes, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'model1' in data
        assert 'model2' in data
        assert 'model3' in data
        assert 'class' in data['model1']
        assert 'class_name' in data['model1']
        assert 'probability' in data['model1']


def test_predict_endpoint_invalid_file_type(client):
    """Test que el endpoint rechaza archivos que no son imágenes"""
    response = client.post(
        "/predict",
        files={"image": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "imagen" in response.json()["detail"].lower()


def test_predict_endpoint_missing_file(client):
    """Test que el endpoint requiere un archivo"""
    response = client.post("/predict")
    
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_class_index_validation(client, sample_image_bytes):
    """Test que el endpoint valida los índices de clase"""
    # Crear predicciones que generen índices fuera de rango
    with patch('app.routes.predict.get_models') as mock_get_models, \
         patch('app.routes.predict.safe_predict') as mock_safe_predict, \
         patch('app.routes.predict.preprocess_image'):
        
        mock_model = MagicMock()
        mock_get_models.return_value = (mock_model, mock_model, mock_model)
        
        # Simular predicción con índice fuera de rango
        # Crear un array donde el máximo está en un índice fuera de rango
        # SPECIES tiene 10 clases (índices 0-9), así que usamos índice 10
        invalid_pred1 = np.zeros(100)
        invalid_pred1[10] = 1.0  # Índice 10 está fuera de rango para SPECIES (0-9)
        
        # Para los otros modelos, usar predicciones válidas
        valid_pred2 = np.array([0.1, 0.2, 0.3, 0.4])  # 4 clases, válido para SHAPES
        valid_pred3 = np.array([0.3, 0.7])  # 2 clases, válido para PLANTS
        
        mock_safe_predict.side_effect = [
            tf.constant([invalid_pred1], dtype=tf.float32),
            tf.constant([valid_pred2], dtype=tf.float32),
            tf.constant([valid_pred3], dtype=tf.float32)
        ]
        
        response = client.post(
            "/predict",
            files={"image": ("test.png", sample_image_bytes, "image/png")}
        )
        
        # Debería fallar con error 500 por índice fuera de rango
        assert response.status_code == 500
        assert "fuera de rango" in response.json()["detail"].lower() or "índice" in response.json()["detail"].lower()

