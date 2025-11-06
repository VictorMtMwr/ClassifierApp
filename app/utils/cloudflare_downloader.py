import os
import re
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Optional
from ..config import DATA_DIR


def parse_image_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parsea el nombre de una imagen para extraer especie y estado.
    
    Ejemplos:
    - 'solanum-melongena_healthy_Lobulate_verified.jpg'
      -> {'especie': 'solanum-melongena', 'estado': 'healthy', 'verified': True}
    - 'solanum-melongena_healthy_Palmeate_verified_981d16c3.jpg'
      -> {'especie': 'solanum-melongena', 'estado': 'healthy', 'verified': True}
    
    Args:
        filename: Nombre del archivo de imagen
        
    Returns:
        Dict con 'especie', 'estado' y 'verified', o None si no coincide con el patrón
    """
    # Remover extensión del archivo
    name_without_ext = os.path.splitext(filename)[0]
    
    # Verificar que contenga 'verified'
    if 'verified' not in name_without_ext.lower():
        return None
    
    # Buscar patrón: especie_estado_resto_verified[opcional_contenido_adicional]
    # Ejemplos:
    # - solanum-melongena_healthy_Lobulate_verified
    # - solanum-melongena_healthy_Palmeate_verified_981d16c3
    # El patrón permite contenido opcional después de _verified
    pattern = r'^([a-z-]+)_(healthy|deseased)_.+?_verified.*$'
    match = re.match(pattern, name_without_ext.lower())
    
    if match:
        especie = match.group(1)
        estado = match.group(2)
        return {
            'especie': especie,
            'estado': estado,
            'verified': True,
            'original_filename': filename
        }
    
    return None


def setup_r2_client(
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    endpoint_url: Optional[str] = None
) -> boto3.client:
    """
    Configura y retorna un cliente de boto3 para Cloudflare R2.
    
    Args:
        account_id: ID de cuenta de Cloudflare
        access_key_id: Clave de acceso de R2
        secret_access_key: Clave secreta de R2
        endpoint_url: URL del endpoint de R2 (opcional, se genera automáticamente si no se proporciona)
        
    Returns:
        Cliente de boto3 configurado para R2
    """
    if endpoint_url is None:
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key
    )
    
    return s3_client


def list_verified_images(
    s3_client: boto3.client,
    bucket_name: str,
    prefix: str = ""
) -> List[str]:
    """
    Lista todas las imágenes verificadas en el bucket de Cloudflare R2.
    
    Args:
        s3_client: Cliente de boto3 configurado
        bucket_name: Nombre del bucket de R2
        prefix: Prefijo para filtrar archivos (opcional)
        
    Returns:
        Lista de nombres de archivos verificados
    """
    verified_images = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Solo incluir si tiene 'verified' en el nombre
                    if 'verified' in key.lower():
                        verified_images.append(key)
        
        print(f"✅ Encontradas {len(verified_images)} imágenes verificadas en el bucket")
        return verified_images
    
    except ClientError as e:
        print(f"❌ Error al listar imágenes del bucket: {e}")
        return []


def download_and_organize_images(
    s3_client: boto3.client,
    bucket_name: str,
    account_id: str = None,
    access_key_id: str = None,
    secret_access_key: str = None,
    prefix: str = "",
    base_dir: str = None
) -> Dict[str, int]:
    """
    Descarga imágenes verificadas del bucket de Cloudflare R2 y las organiza
    en directorios según especie y estado.
    
    Args:
        s3_client: Cliente de boto3 configurado (o None si se proporcionan credenciales)
        bucket_name: Nombre del bucket de R2
        account_id: ID de cuenta de Cloudflare (necesario si s3_client es None)
        access_key_id: Clave de acceso de R2 (necesario si s3_client es None)
        secret_access_key: Clave secreta de R2 (necesario si s3_client es None)
        prefix: Prefijo para filtrar archivos (opcional)
        base_dir: Directorio base donde guardar las imágenes (por defecto usa DATA_DIR/especies/train)
        
    Returns:
        Dict con estadísticas de descarga: {'downloaded': int, 'skipped': int, 'errors': int}
    """
    # Si no se proporciona s3_client, crear uno
    if s3_client is None:
        if not all([account_id, access_key_id, secret_access_key]):
            raise ValueError("Debe proporcionar s3_client o todas las credenciales (account_id, access_key_id, secret_access_key)")
        s3_client = setup_r2_client(account_id, access_key_id, secret_access_key)
    
    # Configurar directorio base
    if base_dir is None:
        base_dir = os.path.join(DATA_DIR, 'especies', 'train')
    os.makedirs(base_dir, exist_ok=True)
    
    # Obtener lista de imágenes verificadas
    verified_images = list_verified_images(s3_client, bucket_name, prefix)
    
    if not verified_images:
        print("⚠️  No se encontraron imágenes verificadas para descargar")
        return {'downloaded': 0, 'skipped': 0, 'errors': 0}
    
    stats = {'downloaded': 0, 'skipped': 0, 'errors': 0}
    
    print(f"\n📥 Iniciando descarga de {len(verified_images)} imágenes...")
    
    for image_key in verified_images:
        try:
            # Parsear nombre del archivo
            parsed = parse_image_filename(image_key)
            
            if not parsed:
                print(f"⚠️  No se pudo parsear: {image_key}")
                stats['skipped'] += 1
                continue
            
            especie = parsed['especie']
            estado = parsed['estado']
            original_filename = parsed['original_filename']
            
            # Crear directorio destino: data/especies/train/especie_estado/
            dest_dir = os.path.join(base_dir, f"{especie}_{estado}")
            os.makedirs(dest_dir, exist_ok=True)
            
            # Ruta completa del archivo destino
            dest_path = os.path.join(dest_dir, original_filename)
            
            # Verificar si el archivo ya existe
            if os.path.exists(dest_path):
                stats['skipped'] += 1
                continue
            
            # Descargar imagen
            print(f"⬇️  Descargando: {original_filename} -> {especie}_{estado}/")
            s3_client.download_file(bucket_name, image_key, dest_path)
            
            stats['downloaded'] += 1
            
            # Mostrar progreso cada 50 imágenes
            if stats['downloaded'] % 50 == 0:
                print(f"📊 Progreso: {stats['downloaded']} descargadas, {stats['skipped']} omitidas, {stats['errors']} errores")
        
        except ClientError as e:
            print(f"❌ Error descargando {image_key}: {e}")
            stats['errors'] += 1
        except Exception as e:
            print(f"❌ Error inesperado con {image_key}: {e}")
            stats['errors'] += 1
    
    print(f"\n✅ Descarga completada:")
    print(f"   - Descargadas: {stats['downloaded']}")
    print(f"   - Omitidas: {stats['skipped']}")
    print(f"   - Errores: {stats['errors']}")
    
    return stats


def download_verified_images_from_r2(
    bucket_name: str,
    account_id: str,
    access_key_id: str,
    secret_access_key: str,
    prefix: str = "",
    base_dir: str = None
) -> Dict[str, int]:
    """
    Función principal para descargar imágenes verificadas de Cloudflare R2.
    
    Esta es la función que deberías llamar desde tu código.
    
    Args:
        bucket_name: Nombre del bucket de Cloudflare R2
        account_id: ID de cuenta de Cloudflare
        access_key_id: Clave de acceso de R2
        secret_access_key: Clave secreta de R2
        prefix: Prefijo para filtrar archivos en el bucket (opcional)
        base_dir: Directorio base donde guardar las imágenes (opcional, por defecto usa DATA_DIR/especies/train)
        
    Returns:
        Dict con estadísticas de descarga
        
    Example:
        >>> stats = download_verified_images_from_r2(
        ...     bucket_name="mi-bucket",
        ...     account_id="tu-account-id",
        ...     access_key_id="tu-access-key",
        ...     secret_access_key="tu-secret-key"
        ... )
        >>> print(f"Descargadas: {stats['downloaded']}")
    """
    s3_client = setup_r2_client(account_id, access_key_id, secret_access_key)
    return download_and_organize_images(
        s3_client=s3_client,
        bucket_name=bucket_name,
        prefix=prefix,
        base_dir=base_dir
    )

