"""
Pruebas unitarias para los endpoints de reentrenamiento y versionado
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_retrain_endpoint_invalid_model(client):
    """Test que el endpoint rechaza nombres de modelo inválidos"""
    response = client.post("/retrain?model=invalid")
    
    assert response.status_code == 400
    assert "especies" in response.json()["detail"].lower() or "hojas" in response.json()["detail"].lower()


def test_retrain_endpoint_valid_model(client):
    """Test que el endpoint acepta nombres de modelo válidos"""
    with patch('app.routes.retrain.detect_new_classes') as mock_detect, \
         patch('app.routes.retrain.threading.Thread') as mock_thread:
        
        mock_detect.return_value = {
            'detected_classes': ['class1', 'class2'],
            'current_classes': ['class1'],
            'new_classes': ['class2'],
            'removed_classes': [],
            'has_changes': True
        }
        
        response = client.post("/retrain?model=especies")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == "Entrenamiento iniciado"
        assert data['model'] == "especies"
        # Verificar que se inició el thread
        mock_thread.assert_called_once()


def test_list_versions_endpoint(client, mock_config):
    """Test que el endpoint lista versiones correctamente"""
    with patch('app.routes.retrain.list_model_versions') as mock_list:
        mock_list.return_value = [
            {'version': 2, 'timestamp': '2024-01-02T12:00:00', 'filename': 'modelo_especies_v0002.h5'},
            {'version': 1, 'timestamp': '2024-01-01T12:00:00', 'filename': 'modelo_especies_v0001.h5'}
        ]
        
        response = client.get("/retrain/versions?model=especies")
        
        assert response.status_code == 200
        data = response.json()
        assert data['model'] == 'especies'
        assert data['total_versions'] == 2
        assert len(data['versions']) == 2


def test_list_versions_endpoint_invalid_model(client):
    """Test que el endpoint rechaza nombres de modelo inválidos"""
    response = client.get("/retrain/versions?model=invalid")
    
    assert response.status_code == 400


def test_version_info_endpoint(client):
    """Test que el endpoint obtiene información de versión"""
    with patch('app.routes.retrain.get_version_info') as mock_get:
        mock_get.return_value = {
            'version': 1,
            'timestamp': '2024-01-01T12:00:00',
            'filename': 'modelo_especies_v0001.h5',
            'notes': 'Test version'
        }
        
        response = client.get("/retrain/version-info?model=especies&version=1")
        
        assert response.status_code == 200
        data = response.json()
        assert data['model'] == 'especies'
        assert data['version'] == 1
        assert 'version_info' in data


def test_version_info_endpoint_not_found(client):
    """Test que el endpoint retorna 404 para versión inexistente"""
    with patch('app.routes.retrain.get_version_info') as mock_get:
        mock_get.return_value = None
        
        response = client.get("/retrain/version-info?model=especies&version=999")
        
        assert response.status_code == 404


def test_restore_version_endpoint(client):
    """Test que el endpoint restaura una versión correctamente"""
    with patch('app.routes.retrain.restore_model_version') as mock_restore, \
         patch('app.routes.retrain.reload_model') as mock_reload:
        
        mock_restore.return_value = {
            'model_name': 'especies',
            'version': 1,
            'restored_at': '2024-01-01T12:00:00',
            'version_info': {'version': 1}
        }
        
        response = client.post("/retrain/restore-version?model=especies&version=1")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['model_name'] == 'especies'
        assert data['version'] == 1
        # Verificar que se recargó el modelo
        mock_reload.assert_called_once_with('especies')


def test_restore_version_endpoint_not_found(client):
    """Test que el endpoint retorna 404 para versión inexistente"""
    with patch('app.routes.retrain.restore_model_version') as mock_restore:
        mock_restore.side_effect = FileNotFoundError("Version not found")
        
        response = client.post("/retrain/restore-version?model=especies&version=999")
        
        assert response.status_code == 404


def test_check_classes_endpoint(client):
    """Test que el endpoint verifica clases correctamente"""
    with patch('app.routes.retrain.detect_new_classes') as mock_detect:
        mock_detect.return_value = {
            'detected_classes': ['class1', 'class2'],
            'current_classes': ['class1'],
            'new_classes': ['class2'],
            'removed_classes': [],
            'has_changes': True
        }
        
        response = client.get("/retrain/check-classes?model=especies")
        
        assert response.status_code == 200
        data = response.json()
        assert data['model'] == 'especies'
        assert 'classes_detected' in data
        assert 'has_changes' in data


def test_gpu_status_endpoint(client):
    """Test que el endpoint obtiene información de GPU"""
    response = client.get("/retrain/gpu-status")
    
    assert response.status_code == 200
    data = response.json()
    assert 'tensorflow_gpu_info' in data
    assert 'gpus_available' in data['tensorflow_gpu_info']

