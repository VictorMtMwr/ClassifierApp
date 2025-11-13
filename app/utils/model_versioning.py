"""
Sistema de versionado de modelos
Permite guardar, listar y restaurar versiones de modelos
"""
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tensorflow as tf
from ..config import MODEL_DIR, BACKUP_DIR, MAX_BACKUPS


def get_version_metadata_path(model_name: str) -> str:
    """Obtiene la ruta del archivo de metadatos de versiones para un modelo"""
    return os.path.join(BACKUP_DIR, f"{model_name}_versions.json")


def get_model_file_map() -> Dict[str, str]:
    """Mapea nombres de modelos a archivos"""
    return {
        'especies': 'modelo_especies.h5',
        'formas': 'modelo_hojas.h5',
        'plantas': 'modelo_plantas.h5'
    }


def load_version_metadata(model_name: str) -> Dict:
    """Carga los metadatos de versiones de un modelo"""
    metadata_path = get_version_metadata_path(model_name)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando metadatos de versiones para {model_name}: {e}")
            return {'versions': [], 'current_version': None}
    return {'versions': [], 'current_version': None}


def save_version_metadata(model_name: str, metadata: Dict):
    """Guarda los metadatos de versiones de un modelo"""
    metadata_path = get_version_metadata_path(model_name)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error guardando metadatos de versiones para {model_name}: {e}")


def create_model_version(model_name: str, version_notes: Optional[str] = None) -> Dict:
    """
    Crea una nueva versión del modelo guardándolo en backups con versionado
    
    Args:
        model_name: Nombre del modelo ('especies', 'formas', 'plantas')
        version_notes: Notas opcionales sobre esta versión
    
    Returns:
        Dict con información de la versión creada
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    model_file_map = get_model_file_map()
    model_path = os.path.join(MODEL_DIR, model_file_map[model_name])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    
    # Cargar metadatos existentes
    metadata = load_version_metadata(model_name)
    versions = metadata.get('versions', [])
    
    # Generar número de versión
    if versions:
        next_version = max(v['version'] for v in versions) + 1
    else:
        next_version = 1
    
    # Crear timestamp
    timestamp = datetime.utcnow()
    timestamp_str = timestamp.strftime('%Y%m%dT%H%M%S')
    
    # Nombre del archivo de versión
    version_filename = f"modelo_{model_name}_v{next_version:04d}_{timestamp_str}.h5"
    version_path = os.path.join(BACKUP_DIR, version_filename)
    
    # Copiar modelo a backup con versionado
    shutil.copy2(model_path, version_path)
    
    # Crear información de versión
    version_info = {
        'version': next_version,
        'timestamp': timestamp.isoformat(),
        'timestamp_str': timestamp_str,
        'filename': version_filename,
        'path': version_path,
        'notes': version_notes or '',
        'size_bytes': os.path.getsize(version_path)
    }
    
    # Agregar a la lista de versiones
    versions.append(version_info)
    
    # Ordenar por versión descendente (más reciente primero)
    versions.sort(key=lambda v: v['version'], reverse=True)
    
    # Mantener solo las MAX_BACKUPS versiones más recientes
    if len(versions) > MAX_BACKUPS:
        versions_to_remove = versions[MAX_BACKUPS:]
        for old_version in versions_to_remove:
            try:
                if os.path.exists(old_version['path']):
                    os.remove(old_version['path'])
                    print(f"Eliminada versión antigua: {old_version['filename']}")
            except Exception as e:
                print(f"Error eliminando versión antigua {old_version['filename']}: {e}")
        versions = versions[:MAX_BACKUPS]
    
    # Actualizar metadatos
    metadata['versions'] = versions
    metadata['current_version'] = next_version
    metadata['last_updated'] = timestamp.isoformat()
    
    # Guardar metadatos
    save_version_metadata(model_name, metadata)
    
    print(f"✅ Versión {next_version} del modelo {model_name} creada: {version_filename}")
    return version_info


def list_model_versions(model_name: str) -> List[Dict]:
    """
    Lista todas las versiones disponibles de un modelo
    
    Args:
        model_name: Nombre del modelo
    
    Returns:
        Lista de diccionarios con información de cada versión
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    metadata = load_version_metadata(model_name)
    versions = metadata.get('versions', [])
    
    # Verificar que los archivos aún existen
    valid_versions = []
    for version in versions:
        if os.path.exists(version['path']):
            # Actualizar tamaño si cambió
            version['size_bytes'] = os.path.getsize(version['path'])
            valid_versions.append(version)
        else:
            print(f"⚠️  Archivo de versión no encontrado: {version['path']}")
    
    # Si hay versiones inválidas, actualizar metadatos
    if len(valid_versions) != len(versions):
        metadata['versions'] = valid_versions
        if valid_versions:
            metadata['current_version'] = max(v['version'] for v in valid_versions)
        save_version_metadata(model_name, metadata)
    
    return valid_versions


def get_model_version_path(model_name: str, version: Optional[int] = None) -> Optional[str]:
    """
    Obtiene la ruta de una versión específica del modelo
    
    Args:
        model_name: Nombre del modelo
        version: Número de versión (None para la más reciente)
    
    Returns:
        Ruta al archivo del modelo o None si no existe
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    versions = list_model_versions(model_name)
    
    if not versions:
        return None
    
    if version is None:
        # Devolver la versión más reciente
        return versions[0]['path']
    
    # Buscar versión específica
    for v in versions:
        if v['version'] == version:
            return v['path']
    
    return None


def restore_model_version(model_name: str, version: int) -> Dict:
    """
    Restaura una versión específica del modelo, reemplazando el modelo actual
    
    Args:
        model_name: Nombre del modelo
        version: Número de versión a restaurar
    
    Returns:
        Dict con información de la versión restaurada
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    model_file_map = get_model_file_map()
    model_path = os.path.join(MODEL_DIR, model_file_map[model_name])
    version_path = get_model_version_path(model_name, version)
    
    if version_path is None:
        raise FileNotFoundError(f"No se encontró la versión {version} del modelo {model_name}")
    
    # Crear backup del modelo actual antes de restaurar
    if os.path.exists(model_path):
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        pre_restore_backup = os.path.join(BACKUP_DIR, f"modelo_{model_name}_pre_restore_{timestamp}.h5")
        shutil.copy2(model_path, pre_restore_backup)
        print(f"Backup del modelo actual creado: {pre_restore_backup}")
    
    # Restaurar la versión
    shutil.copy2(version_path, model_path)
    
    # Validar que el modelo restaurado se puede cargar
    try:
        _ = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo {model_name} restaurado a la versión {version}")
    except Exception as e:
        # Si falla, restaurar desde el backup
        if os.path.exists(pre_restore_backup):
            shutil.copy2(pre_restore_backup, model_path)
            raise RuntimeError(f"Error validando modelo restaurado: {e}. Modelo original restaurado.")
        raise
    
    # Obtener información de la versión restaurada
    versions = list_model_versions(model_name)
    restored_version = next((v for v in versions if v['version'] == version), None)
    
    return {
        'model_name': model_name,
        'version': version,
        'restored_at': datetime.utcnow().isoformat(),
        'version_info': restored_version
    }


def get_version_info(model_name: str, version: int) -> Optional[Dict]:
    """Obtiene información detallada de una versión específica"""
    versions = list_model_versions(model_name)
    return next((v for v in versions if v['version'] == version), None)

