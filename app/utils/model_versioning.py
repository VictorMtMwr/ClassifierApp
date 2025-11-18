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
    Crea una nueva versión del modelo guardándolo en backups con versionado.
    Busca primero modelos versionados en MODEL_DIR, si no encuentra busca sin versión.
    Después de copiar a backups, elimina versiones antiguas de MODEL_DIR.
    
    Args:
        model_name: Nombre del modelo ('especies', 'formas', 'plantas')
        version_notes: Notas opcionales sobre esta versión
    
    Returns:
        Dict con información de la versión creada (incluye la versión copiada y si se eliminó de MODEL_DIR)
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    import glob
    
    model_file_map = get_model_file_map()
    model_base_name = os.path.splitext(model_file_map[model_name])[0]  # 'modelo_especies', 'modelo_hojas', etc.
    
    # Función auxiliar para ordenar por timestamp (más reciente primero)
    def get_timestamp_from_filename(filename):
        """Extrae el timestamp del nombre del archivo para ordenar"""
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) >= 3:
            timestamp = parts[-1].replace('.h5', '')
            return timestamp
        return ''
    
    # Buscar modelo versionado en MODEL_DIR primero
    pattern_model = os.path.join(MODEL_DIR, f"{model_base_name}_v*_*.h5")
    versioned_models = glob.glob(pattern_model)
    current_model_path = None
    model_in_models_dir = False
    
    if versioned_models:
        # Ordenar por timestamp (más reciente primero)
        versioned_models.sort(key=get_timestamp_from_filename, reverse=True)
        current_model_path = versioned_models[0]
        model_in_models_dir = True
        print(f"📋 Encontrado modelo versionado en {MODEL_DIR}: {os.path.basename(current_model_path)}")
    else:
        # Buscar modelo sin versión en MODEL_DIR
        model_path = os.path.join(MODEL_DIR, model_file_map[model_name])
        if os.path.exists(model_path):
            current_model_path = model_path
            model_in_models_dir = True
            print(f"📋 Encontrado modelo sin versión en {MODEL_DIR}: {os.path.basename(model_path)}")
    
    if not current_model_path or not os.path.exists(current_model_path):
        raise FileNotFoundError(f"No se encontró el modelo {model_name} en {MODEL_DIR}")
    
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
    version_filename = f"{model_base_name}_v{next_version:04d}_{timestamp_str}.h5"
    version_path = os.path.abspath(os.path.join(BACKUP_DIR, version_filename))
    
    # Asegurar que el directorio de backups existe
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Copiar modelo a backup con versionado
    shutil.copy2(current_model_path, version_path)
    print(f"📋 Copiado a {BACKUP_DIR}: {version_filename}")
    
    # Crear información de versión
    version_info = {
        'version': next_version,
        'timestamp': timestamp.isoformat(),
        'timestamp_str': timestamp_str,
        'filename': version_filename,
        'path': version_path,  # Ruta absoluta normalizada
        'notes': version_notes or '',
        'size_bytes': os.path.getsize(version_path),
        'source_path': current_model_path
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
                    print(f"🗑️  Eliminada versión antigua de {BACKUP_DIR}: {old_version['filename']}")
            except Exception as e:
                print(f"⚠️  Error eliminando versión antigua {old_version['filename']}: {e}")
        versions = versions[:MAX_BACKUPS]
    
    # Actualizar metadatos
    metadata['versions'] = versions
    metadata['current_version'] = next_version
    metadata['last_updated'] = timestamp.isoformat()
    
    # Guardar metadatos
    save_version_metadata(model_name, metadata)
    
    # Eliminar versiones antiguas de MODEL_DIR (mantener solo la más reciente si hay múltiples)
    if model_in_models_dir and versioned_models:
        # Si hay múltiples versiones en MODEL_DIR, eliminar todas excepto la que acabamos de copiar
        for old_versioned_model in versioned_models:
            if old_versioned_model != current_model_path:
                try:
                    os.remove(old_versioned_model)
                    print(f"🗑️  Eliminada versión antigua de {MODEL_DIR}: {os.path.basename(old_versioned_model)}")
                except Exception as e:
                    print(f"⚠️  Error eliminando versión antigua de {MODEL_DIR}: {e}")
        # Eliminar también el modelo versionado que acabamos de copiar (se guardará uno nuevo después)
        try:
            os.remove(current_model_path)
            print(f"🗑️  Eliminado modelo versionado anterior de {MODEL_DIR}: {os.path.basename(current_model_path)}")
        except Exception as e:
            print(f"⚠️  Error eliminando modelo anterior de {MODEL_DIR}: {e}")
    
    print(f"✅ Versión {next_version} del modelo {model_name} creada en {BACKUP_DIR}: {version_filename}")
    return version_info


def list_model_versions(model_name: str) -> List[Dict]:
    """
    Lista todas las versiones disponibles de un modelo.
    Detecta automáticamente archivos físicos que no estén en el JSON y los agrega.
    
    Args:
        model_name: Nombre del modelo
    
    Returns:
        Lista de diccionarios con información de cada versión
    """
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    import glob
    import re
    
    metadata = load_version_metadata(model_name)
    versions = metadata.get('versions', [])
    
    # Obtener nombres de archivos ya registrados
    registered_filenames = {v.get('filename', '') for v in versions}
    
    # Buscar archivos físicos de versiones en BACKUP_DIR que no estén registrados
    model_file_map = get_model_file_map()
    model_base_name = os.path.splitext(model_file_map[model_name])[0]  # 'modelo_especies', etc.
    pattern = os.path.join(BACKUP_DIR, f"{model_base_name}_v*_*.h5")
    physical_files = glob.glob(pattern)
    
    # Función para extraer información de versión desde el nombre del archivo
    def parse_version_from_filename(filename):
        """Extrae número de versión y timestamp del nombre del archivo"""
        basename = os.path.basename(filename)
        # Patrón: modelo_especies_v0001_20251118T150737.h5
        match = re.match(rf"{model_base_name}_v(\d+)_(\d{{8}}T\d{{6}})\.h5", basename)
        if match:
            version_num = int(match.group(1))
            timestamp_str = match.group(2)
            # Convertir timestamp_str a datetime
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
                return {
                    'version': version_num,
                    'timestamp_str': timestamp_str,
                    'filename': basename,
                    'path': os.path.abspath(filename),
                    'timestamp': timestamp.isoformat()
                }
            except:
                return None
        return None
    
    # Detectar archivos físicos no registrados y agregarlos
    new_versions = []
    for physical_file in physical_files:
        filename = os.path.basename(physical_file)
        if filename not in registered_filenames:
            version_info = parse_version_from_filename(physical_file)
            if version_info:
                # Agregar información adicional
                version_info['size_bytes'] = os.path.getsize(physical_file)
                version_info['notes'] = f"Versión detectada automáticamente - {version_info['timestamp']}"
                new_versions.append(version_info)
    
    # Agregar nuevas versiones detectadas
    if new_versions:
        versions.extend(new_versions)
        print(f"ℹ️  Detectadas {len(new_versions)} versión(es) no registrada(s) del modelo {model_name}, agregadas automáticamente")
    
    # Verificar que los archivos aún existen
    valid_versions = []
    removed_count = 0
    for version in versions:
        # Normalizar la ruta (puede ser relativa o absoluta)
        version_path = version.get('path', '')
        if not version_path:
            # Si no hay path, construir desde filename
            version_path = os.path.join(BACKUP_DIR, version.get('filename', ''))
        
        if not os.path.isabs(version_path):
            # Si es relativa, intentar construirla desde BACKUP_DIR
            version_path = os.path.join(BACKUP_DIR, os.path.basename(version_path))
        else:
            version_path = os.path.abspath(version_path)
        
        if os.path.exists(version_path):
            # Actualizar tamaño y ruta normalizada
            version['size_bytes'] = os.path.getsize(version_path)
            version['path'] = version_path  # Actualizar con ruta normalizada
            valid_versions.append(version)
        else:
            # Archivo no encontrado - será removido silenciosamente
            removed_count += 1
    
    # Ordenar por versión descendente (más reciente primero)
    valid_versions.sort(key=lambda v: v['version'], reverse=True)
    
    # Si hubo cambios (versiones agregadas o removidas), actualizar metadatos automáticamente
    if len(valid_versions) != len(versions) or new_versions:
        metadata['versions'] = valid_versions
        if valid_versions:
            metadata['current_version'] = max(v['version'] for v in valid_versions)
        else:
            # Si no hay versiones válidas, limpiar current_version
            metadata['current_version'] = None
        metadata['last_updated'] = datetime.utcnow().isoformat()
        save_version_metadata(model_name, metadata)
        if removed_count > 0:
            print(f"ℹ️  Limpiadas {removed_count} versión(es) huérfana(s) del modelo {model_name}")
    
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
    Restaura una versión específica del modelo en MODEL_DIR con nombre versionado.
    Elimina versiones antiguas de MODEL_DIR y guarda la versión restaurada con su nombre versionado.
    
    Args:
        model_name: Nombre del modelo
        version: Número de versión a restaurar
    
    Returns:
        Dict con información de la versión restaurada
    """
    import glob
    
    if model_name not in ['especies', 'formas', 'plantas']:
        raise ValueError(f"Nombre de modelo inválido: {model_name}")
    
    model_file_map = get_model_file_map()
    model_base_name = os.path.splitext(model_file_map[model_name])[0]
    version_path = get_model_version_path(model_name, version)
    
    if version_path is None:
        raise FileNotFoundError(f"No se encontró la versión {version} del modelo {model_name}")
    
    # Normalizar la ruta de versión
    if not os.path.isabs(version_path):
        version_path = os.path.abspath(os.path.join(BACKUP_DIR, os.path.basename(version_path)))
    else:
        version_path = os.path.abspath(version_path)
    
    if not os.path.exists(version_path):
        raise FileNotFoundError(f"El archivo de versión no existe: {version_path}")
    
    # Función auxiliar para ordenar por timestamp
    def get_timestamp_from_filename(filename):
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) >= 3:
            timestamp = parts[-1].replace('.h5', '')
            return timestamp
        return ''
    
    # Buscar versiones antiguas en MODEL_DIR para eliminarlas
    pattern_model = os.path.join(MODEL_DIR, f"{model_base_name}_v*_*.h5")
    versioned_models_in_models = glob.glob(pattern_model)
    
    # También buscar modelo sin versión
    model_path_no_version = os.path.join(MODEL_DIR, model_file_map[model_name])
    existing_models = list(versioned_models_in_models)
    if os.path.exists(model_path_no_version):
        existing_models.append(model_path_no_version)
    
    # Guardar referencias para restauración en caso de error
    backup_source_path = None
    
    # Crear backup del modelo actual antes de restaurar (si existe)
    pre_restore_backup = None
    if existing_models:
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        pre_restore_backup = os.path.join(BACKUP_DIR, f"{model_base_name}_pre_restore_{timestamp}.h5")
        # Copiar el más reciente como backup
        if versioned_models_in_models:
            versioned_models_in_models.sort(key=get_timestamp_from_filename, reverse=True)
            backup_source_path = versioned_models_in_models[0]
            shutil.copy2(backup_source_path, pre_restore_backup)
        else:
            backup_source_path = model_path_no_version
            shutil.copy2(model_path_no_version, pre_restore_backup)
        print(f"📋 Backup del modelo actual creado: {os.path.basename(pre_restore_backup)}")
    
    # Eliminar versiones antiguas de MODEL_DIR
    for old_model in existing_models:
        try:
            os.remove(old_model)
            print(f"🗑️  Eliminada versión antigua de {MODEL_DIR}: {os.path.basename(old_model)}")
        except Exception as e:
            print(f"⚠️  Error eliminando versión antigua de {MODEL_DIR}: {e}")
    
    # Obtener el nombre del archivo versionado desde BACKUP_DIR
    version_filename = os.path.basename(version_path)
    restored_model_path = os.path.join(MODEL_DIR, version_filename)
    
    # Copiar la versión restaurada a MODEL_DIR con su nombre versionado
    shutil.copy2(version_path, restored_model_path)
    print(f"📋 Versión {version} restaurada en {MODEL_DIR}: {version_filename}")
    
    # Validar que el modelo restaurado se puede cargar
    try:
        _ = tf.keras.models.load_model(restored_model_path)
        print(f"✅ Modelo {model_name} restaurado a la versión {version}")
    except Exception as e:
        # Si falla, intentar restaurar desde el backup
        if pre_restore_backup and os.path.exists(pre_restore_backup) and backup_source_path:
            # Eliminar el modelo que falló
            try:
                os.remove(restored_model_path)
            except:
                pass
            # Restaurar desde el backup a la ruta original
            shutil.copy2(pre_restore_backup, backup_source_path)
            print(f"🔄 Modelo original restaurado desde backup")
            raise RuntimeError(f"Error validando modelo restaurado: {e}. Modelo original restaurado.")
        raise
    
    # Obtener información de la versión restaurada
    versions = list_model_versions(model_name)
    restored_version = next((v for v in versions if v['version'] == version), None)
    
    return {
        'model_name': model_name,
        'version': version,
        'restored_at': datetime.utcnow().isoformat(),
        'version_info': restored_version,
        'restored_path': restored_model_path
    }


def get_version_info(model_name: str, version: int) -> Optional[Dict]:
    """Obtiene información detallada de una versión específica"""
    versions = list_model_versions(model_name)
    return next((v for v in versions if v['version'] == version), None)

