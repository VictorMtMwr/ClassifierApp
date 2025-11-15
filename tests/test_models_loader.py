"""
Pruebas unitarias para el módulo de carga de modelos
"""
import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
import tensorflow as tf
from app.models_loader import load_models, get_models, reload_model, load_model_version


def test_get_models_returns_models(mock_models):
    """Test que get_models retorna los modelos correctamente"""
    with patch('app.models_loader._models', {
        'especies': mock_models[0],
        'formas': mock_models[1],
        'plantas': mock_models[2]
    }):
        especies, formas, plantas = get_models()
        
        assert especies is not None
        assert formas is not None
        assert plantas is not None


def test_reload_model_invalid_name():
    """Test que reload_model valida el nombre del modelo"""
    with pytest.raises(ValueError):
        reload_model('invalid_model')


def test_reload_model_file_not_found(mock_config):
    """Test que reload_model maneja correctamente archivos inexistentes"""
    # Asegurarse de que el archivo no existe en el directorio mock
    model_path = os.path.join(mock_config['MODEL_DIR'], 'modelo_especies.h5')
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Usar patch para asegurar que se use el MODEL_DIR del mock
    # Necesitamos parchear tanto en config como en models_loader
    with patch('app.config.MODEL_DIR', mock_config['MODEL_DIR']), \
         patch('app.models_loader.MODEL_DIR', mock_config['MODEL_DIR']):
        with pytest.raises(FileNotFoundError) as exc_info:
            reload_model('especies')
        assert 'No se encontró el modelo' in str(exc_info.value)


def test_load_model_version_invalid_name():
    """Test que load_model_version valida el nombre del modelo"""
    with pytest.raises(ValueError):
        load_model_version('invalid_model', 1)


def test_load_model_version_not_found(mock_config):
    """Test que load_model_version maneja versiones inexistentes"""
    with patch('app.models_loader.get_model_version_path', return_value=None):
        with pytest.raises(FileNotFoundError):
            load_model_version('especies', 999)

