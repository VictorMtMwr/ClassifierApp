"""
Pruebas unitarias para el sistema de versionado de modelos
"""
import pytest
import os
import json
import shutil
import glob
from unittest.mock import patch, MagicMock
from app.utils.model_versioning import (
    create_model_version,
    list_model_versions,
    get_version_info,
    restore_model_version,
    get_model_version_path,
    load_version_metadata,
    save_version_metadata
)


@pytest.fixture
def setup_versioning_test(mock_config, sample_model_file):
    """Configura el entorno para pruebas de versionado"""
    # Limpiar metadatos y versiones existentes antes de la prueba
    metadata_path = os.path.join(mock_config['BACKUP_DIR'], 'especies_versions.json')
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    # Limpiar archivos de versión existentes
    version_files = glob.glob(os.path.join(mock_config['BACKUP_DIR'], 'modelo_especies_v*.h5'))
    for f in version_files:
        try:
            os.remove(f)
        except:
            pass
    
    # También limpiar el directorio real de backups por si acaso
    # (esto puede pasar si los tests anteriores no limpiaron correctamente)
    real_backup_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'backups')
    if os.path.exists(real_backup_dir):
        real_metadata = os.path.join(real_backup_dir, 'especies_versions.json')
        if os.path.exists(real_metadata):
            # Leer y limpiar solo las versiones de este test
            try:
                with open(real_metadata, 'r') as f:
                    data = json.load(f)
                    # Resetear versiones
                    data['versions'] = []
                    data['current_version'] = None
                with open(real_metadata, 'w') as f:
                    json.dump(data, f)
            except:
                pass
    
    # Crear un modelo de prueba en el directorio de modelos
    model_path = sample_model_file
    yield {
        'model_path': model_path,
        'model_name': 'especies'
    }
    
    # Limpiar metadatos después de la prueba
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    # Limpiar archivos de versión después de la prueba
    version_files = glob.glob(os.path.join(mock_config['BACKUP_DIR'], 'modelo_especies_v*.h5'))
    for f in version_files:
        try:
            os.remove(f)
        except:
            pass


def test_create_model_version(mock_config, setup_versioning_test):
    """Test que create_model_version crea correctamente una versión"""
    model_name = setup_versioning_test['model_name']
    model_path = setup_versioning_test['model_path']
    
    version_info = create_model_version(model_name, version_notes="Test version")
    
    assert version_info is not None
    assert version_info['version'] == 1
    assert 'timestamp' in version_info
    assert 'filename' in version_info
    assert 'path' in version_info
    assert version_info['notes'] == "Test version"
    assert os.path.exists(version_info['path'])


def test_create_model_version_sequential(mock_config, setup_versioning_test):
    """Test que las versiones se numeran secuencialmente"""
    model_name = setup_versioning_test['model_name']
    
    # Asegurarse de que estamos usando el mock_config
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']):
        # Limpiar cualquier versión previa
        metadata_path = os.path.join(mock_config['BACKUP_DIR'], 'especies_versions.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        v1 = create_model_version(model_name, "Version 1")
        v2 = create_model_version(model_name, "Version 2")
        v3 = create_model_version(model_name, "Version 3")
        
        assert v1['version'] == 1
        assert v2['version'] == 2
        assert v3['version'] == 3


def test_list_model_versions(mock_config, setup_versioning_test):
    """Test que list_model_versions lista correctamente las versiones"""
    model_name = setup_versioning_test['model_name']
    
    # Asegurarse de que estamos usando el mock_config
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']):
        # Limpiar cualquier versión previa
        metadata_path = os.path.join(mock_config['BACKUP_DIR'], 'especies_versions.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Crear algunas versiones
        create_model_version(model_name, "Version 1")
        create_model_version(model_name, "Version 2")
        
        versions = list_model_versions(model_name)
        
        assert len(versions) == 2
        assert versions[0]['version'] == 2  # Más reciente primero
        assert versions[1]['version'] == 1


def test_get_version_info(mock_config, setup_versioning_test):
    """Test que get_version_info obtiene información correcta de una versión"""
    model_name = setup_versioning_test['model_name']
    
    version_info = create_model_version(model_name, "Test version")
    version_num = version_info['version']
    
    info = get_version_info(model_name, version_num)
    
    assert info is not None
    assert info['version'] == version_num
    assert info['notes'] == "Test version"


def test_get_version_info_not_found(mock_config, setup_versioning_test):
    """Test que get_version_info retorna None para versión inexistente"""
    model_name = setup_versioning_test['model_name']
    
    info = get_version_info(model_name, 999)
    
    assert info is None


def test_get_model_version_path(mock_config, setup_versioning_test):
    """Test que get_model_version_path retorna la ruta correcta"""
    model_name = setup_versioning_test['model_name']
    
    version_info = create_model_version(model_name, "Test")
    version_path = get_model_version_path(model_name, version_info['version'])
    
    assert version_path == version_info['path']
    assert os.path.exists(version_path)


def test_get_model_version_path_latest(mock_config, setup_versioning_test):
    """Test que get_model_version_path retorna la más reciente si no se especifica versión"""
    model_name = setup_versioning_test['model_name']
    
    v1 = create_model_version(model_name, "Version 1")
    v2 = create_model_version(model_name, "Version 2")
    
    latest_path = get_model_version_path(model_name, None)
    
    assert latest_path == v2['path']


def test_restore_model_version(mock_config, setup_versioning_test):
    """Test que restore_model_version restaura correctamente una versión"""
    model_name = setup_versioning_test['model_name']
    model_path = setup_versioning_test['model_path']
    
    # Crear una versión
    version_info = create_model_version(model_name, "Test version")
    
    # Restaurar la versión
    with patch('tensorflow.keras.models.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        restore_result = restore_model_version(model_name, version_info['version'])
    
    assert restore_result is not None
    assert restore_result['version'] == version_info['version']
    assert restore_result['model_name'] == model_name


def test_create_model_version_invalid_name(mock_config):
    """Test que create_model_version valida el nombre del modelo"""
    with pytest.raises(ValueError):
        create_model_version('invalid_model', "Test")


def test_load_and_save_version_metadata(mock_config):
    """Test que load_version_metadata y save_version_metadata funcionan correctamente"""
    model_name = 'especies'
    metadata = {
        'versions': [],
        'current_version': None
    }
    
    save_version_metadata(model_name, metadata)
    loaded = load_version_metadata(model_name)
    
    assert loaded == metadata

