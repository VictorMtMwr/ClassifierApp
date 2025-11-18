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
    """Configura el entorno para pruebas de versionado - solo especies"""
    model_name = 'especies'
    _cleanup_versioning_test(mock_config, model_name)
    
    # Crear un modelo de prueba en el directorio de modelos
    model_path = sample_model_file
    yield {
        'model_path': model_path,
        'model_name': model_name
    }
    
    # Limpiar después de la prueba
    _cleanup_versioning_test(mock_config, model_name)


@pytest.fixture
def setup_versioning_test_formas(mock_config, sample_model_file_formas):
    """Configura el entorno para pruebas de versionado - formas"""
    model_name = 'formas'
    _cleanup_versioning_test(mock_config, model_name)
    
    # Usar el modelo creado por el fixture
    model_path = sample_model_file_formas
    yield {
        'model_path': model_path,
        'model_name': model_name
    }
    
    # Limpiar después de la prueba
    _cleanup_versioning_test(mock_config, model_name)


@pytest.fixture
def setup_versioning_test_plantas(mock_config, sample_model_file_plantas):
    """Configura el entorno para pruebas de versionado - plantas"""
    model_name = 'plantas'
    _cleanup_versioning_test(mock_config, model_name)
    
    # Usar el modelo creado por el fixture
    model_path = sample_model_file_plantas
    yield {
        'model_path': model_path,
        'model_name': model_name
    }
    
    # Limpiar después de la prueba
    _cleanup_versioning_test(mock_config, model_name)


def _cleanup_versioning_test(mock_config, model_name):
    """Función auxiliar para limpiar versiones de un modelo"""
    metadata_path = os.path.join(mock_config['BACKUP_DIR'], f'{model_name}_versions.json')
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    # Limpiar archivos de versión existentes - usar el mapeo correcto de nombres
    from app.utils.model_versioning import get_model_file_map
    model_file_map = get_model_file_map()
    if model_name in model_file_map:
        # Obtener el nombre base del archivo (sin extensión)
        model_file_base = os.path.splitext(model_file_map[model_name])[0]
        # Buscar archivos de versión con el nombre correcto
        version_files = glob.glob(os.path.join(mock_config['BACKUP_DIR'], f'{model_file_base}_v*.h5'))
        for f in version_files:
            try:
                os.remove(f)
            except:
                pass
        # También limpiar backups pre-restore
        pre_restore_files = glob.glob(os.path.join(mock_config['BACKUP_DIR'], f'{model_file_base}_pre_restore_*.h5'))
        for f in pre_restore_files:
            try:
                os.remove(f)
            except:
                pass


def test_create_model_version(mock_config, setup_versioning_test):
    """Test que create_model_version crea correctamente una versión"""
    model_name = setup_versioning_test['model_name']
    model_path = setup_versioning_test['model_path']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        version_info = create_model_version(model_name, version_notes="Test version")
        
        assert version_info is not None
        assert version_info['version'] == 1
        assert 'timestamp' in version_info
        assert 'filename' in version_info
        assert 'path' in version_info
        assert version_info['notes'] == "Test version"
        # Verificar que la ruta es absoluta y normalizada
        assert os.path.isabs(version_info['path'])
        assert os.path.exists(version_info['path'])
        # Verificar que el nombre del archivo es correcto
        assert 'modelo_especies' in version_info['filename']


def test_create_model_version_sequential(mock_config, setup_versioning_test):
    """Test que las versiones se numeran secuencialmente"""
    model_name = setup_versioning_test['model_name']
    
    # Asegurarse de que estamos usando el mock_config
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
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
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
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
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        version_info = create_model_version(model_name, "Test version")
        version_num = version_info['version']
        
        info = get_version_info(model_name, version_num)
        
        assert info is not None
        assert info['version'] == version_num
        assert info['notes'] == "Test version"
        # Verificar que la ruta es absoluta
        assert os.path.isabs(info['path'])


def test_get_version_info_not_found(mock_config, setup_versioning_test):
    """Test que get_version_info retorna None para versión inexistente"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']):
        info = get_version_info(model_name, 999)
        
        assert info is None


def test_get_model_version_path(mock_config, setup_versioning_test):
    """Test que get_model_version_path retorna la ruta correcta"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        version_info = create_model_version(model_name, "Test")
        version_path = get_model_version_path(model_name, version_info['version'])
        
        # Las rutas deben ser absolutas y normalizadas
        assert os.path.isabs(version_path)
        assert os.path.isabs(version_info['path'])
        # Normalizar ambas rutas para comparar
        assert os.path.abspath(version_path) == os.path.abspath(version_info['path'])
        assert os.path.exists(version_path)


def test_get_model_version_path_latest(mock_config, setup_versioning_test):
    """Test que get_model_version_path retorna la más reciente si no se especifica versión"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        v1 = create_model_version(model_name, "Version 1")
        v2 = create_model_version(model_name, "Version 2")
        
        latest_path = get_model_version_path(model_name, None)
        
        # Normalizar rutas para comparar
        assert os.path.abspath(latest_path) == os.path.abspath(v2['path'])
        assert os.path.isabs(latest_path)


def test_restore_model_version(mock_config, setup_versioning_test):
    """Test que restore_model_version restaura correctamente una versión"""
    model_name = setup_versioning_test['model_name']
    model_path = setup_versioning_test['model_path']
    
    # Crear una versión
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        version_info = create_model_version(model_name, "Test version")
    
    # Restaurar la versión
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']), \
         patch('tensorflow.keras.models.load_model') as mock_load:
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


# Tests para todos los modelos
def test_create_model_version_especies(mock_config, setup_versioning_test):
    """Test que create_model_version funciona para especies"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        version_info = create_model_version(model_name, "Test especies")
        assert version_info['version'] == 1
        assert 'modelo_especies' in version_info['filename']


def test_create_model_version_formas(mock_config, setup_versioning_test_formas):
    """Test que create_model_version funciona para formas"""
    model_name = setup_versioning_test_formas['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        version_info = create_model_version(model_name, "Test formas")
        assert version_info['version'] == 1
        assert 'modelo_hojas' in version_info['filename']  # El archivo se llama modelo_hojas.h5


def test_create_model_version_plantas(mock_config, setup_versioning_test_plantas):
    """Test que create_model_version funciona para plantas"""
    model_name = setup_versioning_test_plantas['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        version_info = create_model_version(model_name, "Test plantas")
        assert version_info['version'] == 1
        assert 'modelo_plantas' in version_info['filename']


def test_list_versions_all_models_especies(mock_config, setup_versioning_test):
    """Test que list_model_versions funciona para especies"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        create_model_version(model_name, f"Version 1 - {model_name}")
        create_model_version(model_name, f"Version 2 - {model_name}")
        
        versions = list_model_versions(model_name)
        assert len(versions) == 2
        assert versions[0]['version'] == 2
        assert versions[1]['version'] == 1


def test_list_versions_all_models_formas(mock_config, setup_versioning_test_formas):
    """Test que list_model_versions funciona para formas"""
    model_name = setup_versioning_test_formas['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        create_model_version(model_name, f"Version 1 - {model_name}")
        create_model_version(model_name, f"Version 2 - {model_name}")
        
        versions = list_model_versions(model_name)
        assert len(versions) == 2
        assert versions[0]['version'] == 2
        assert versions[1]['version'] == 1


def test_list_versions_all_models_plantas(mock_config, setup_versioning_test_plantas):
    """Test que list_model_versions funciona para plantas"""
    model_name = setup_versioning_test_plantas['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        create_model_version(model_name, f"Version 1 - {model_name}")
        create_model_version(model_name, f"Version 2 - {model_name}")
        
        versions = list_model_versions(model_name)
        assert len(versions) == 2
        assert versions[0]['version'] == 2
        assert versions[1]['version'] == 1


def test_restore_version_all_models_especies(mock_config, setup_versioning_test):
    """Test que restore_model_version funciona para especies"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']), \
         patch('tensorflow.keras.models.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        
        version_info = create_model_version(model_name, f"Test {model_name}")
        restore_result = restore_model_version(model_name, version_info['version'])
        
        assert restore_result is not None
        assert restore_result['version'] == version_info['version']
        assert restore_result['model_name'] == model_name


def test_restore_version_all_models_formas(mock_config, setup_versioning_test_formas):
    """Test que restore_model_version funciona para formas"""
    model_name = setup_versioning_test_formas['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']), \
         patch('tensorflow.keras.models.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        
        version_info = create_model_version(model_name, f"Test {model_name}")
        restore_result = restore_model_version(model_name, version_info['version'])
        
        assert restore_result is not None
        assert restore_result['version'] == version_info['version']
        assert restore_result['model_name'] == model_name


def test_restore_version_all_models_plantas(mock_config, setup_versioning_test_plantas):
    """Test que restore_model_version funciona para plantas"""
    model_name = setup_versioning_test_plantas['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']), \
         patch('tensorflow.keras.models.load_model') as mock_load:
        mock_load.return_value = MagicMock()
        
        version_info = create_model_version(model_name, f"Test {model_name}")
        restore_result = restore_model_version(model_name, version_info['version'])
        
        assert restore_result is not None
        assert restore_result['version'] == version_info['version']
        assert restore_result['model_name'] == model_name
        # Verificar que el archivo de versión existe y es accesible
        assert os.path.exists(version_info['path'])
        assert os.path.isabs(version_info['path'])


# Tests adicionales para verificar normalización de rutas y versionado completo
def test_version_path_normalization(mock_config, setup_versioning_test):
    """Test que las rutas se normalizan correctamente en todas las operaciones"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        # Crear versión
        version_info = create_model_version(model_name, "Test normalization")
        
        # Verificar que la ruta es absoluta
        assert os.path.isabs(version_info['path'])
        
        # Listar versiones y verificar normalización
        versions = list_model_versions(model_name)
        assert len(versions) == 1
        assert os.path.isabs(versions[0]['path'])
        assert os.path.abspath(versions[0]['path']) == os.path.abspath(version_info['path'])
        
        # Obtener path y verificar normalización
        version_path = get_model_version_path(model_name, version_info['version'])
        assert os.path.isabs(version_path)
        assert os.path.abspath(version_path) == os.path.abspath(version_info['path'])


def test_versioning_all_models_complete_flow(mock_config, setup_versioning_test, setup_versioning_test_formas, setup_versioning_test_plantas):
    """Test que el flujo completo de versionado funciona para todos los modelos"""
    models = [
        ('especies', setup_versioning_test),
        ('formas', setup_versioning_test_formas),
        ('plantas', setup_versioning_test_plantas)
    ]
    
    for model_name, setup in models:
        with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
             patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
            
            # 1. Crear versión
            version_info = create_model_version(model_name, f"Test {model_name}")
            assert version_info['version'] == 1
            assert os.path.exists(version_info['path'])
            assert os.path.isabs(version_info['path'])
            
            # 2. Listar versiones
            versions = list_model_versions(model_name)
            assert len(versions) == 1
            assert versions[0]['version'] == 1
            
            # 3. Obtener información de versión
            info = get_version_info(model_name, 1)
            assert info is not None
            assert info['version'] == 1
            
            # 4. Obtener path de versión
            version_path = get_model_version_path(model_name, 1)
            assert os.path.exists(version_path)
            assert os.path.isabs(version_path)
            
            # 5. Restaurar versión (mock load_model)
            with patch('tensorflow.keras.models.load_model') as mock_load:
                mock_load.return_value = MagicMock()
                restore_result = restore_model_version(model_name, 1)
                assert restore_result is not None
                assert restore_result['version'] == 1
                assert restore_result['model_name'] == model_name


def test_restore_version_path_handling(mock_config, setup_versioning_test):
    """Test que restore_model_version maneja correctamente las rutas normalizadas"""
    model_name = setup_versioning_test['model_name']
    
    with patch('app.utils.model_versioning.BACKUP_DIR', mock_config['BACKUP_DIR']), \
         patch('app.utils.model_versioning.MODEL_DIR', mock_config['MODEL_DIR']):
        
        # Crear versión
        version_info = create_model_version(model_name, "Test restore path")
        
        # Verificar que la ruta de versión es absoluta
        assert os.path.isabs(version_info['path'])
        assert os.path.exists(version_info['path'])
        
        # Restaurar versión
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_load.return_value = MagicMock()
            restore_result = restore_model_version(model_name, version_info['version'])
            
            assert restore_result is not None
            # Verificar que el modelo fue restaurado (mock_load fue llamado)
            assert mock_load.called

