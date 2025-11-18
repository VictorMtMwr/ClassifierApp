from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import threading
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ..config import (
    MODEL_DIR, DATA_DIR, BACKUP_DIR, MAX_BACKUPS,
    CLOUDFLARE_R2_BUCKET_NAME, CLOUDFLARE_R2_ACCOUNT_ID,
    CLOUDFLARE_R2_ACCESS_KEY_ID, CLOUDFLARE_R2_SECRET_ACCESS_KEY,
    CLOUDFLARE_R2_PREFIX
)
from ..utils.label_detector import detect_new_classes, update_config_with_new_classes, update_config_with_detected_classes, adjust_model_for_new_classes, reload_config
from ..utils.cloudflare_downloader import download_verified_images_from_r2
from ..utils.model_versioning import create_model_version, list_model_versions, restore_model_version, get_version_info
from ..models_loader import reload_model

def get_gpu_info():
    """Obtiene información sobre las GPUs disponibles y su configuración"""
    gpus = tf.config.list_physical_devices('GPU')
    info = {
        'gpus_available': len(gpus),
        'gpus': []
    }
    
    for i, gpu in enumerate(gpus):
        try:
            details = tf.config.experimental.get_device_details(gpu)
            try:
                memory_info = tf.config.experimental.get_memory_info(gpu)
                memory_usage = {
                    'current_gb': memory_info['current'] / (1024**3),
                    'peak_gb': memory_info['peak'] / (1024**3)
                }
            except Exception:
                memory_usage = None
            
            info['gpus'].append({
                'index': i,
                'name': gpu.name,
                'details': details if details else {},
                'memory_usage': memory_usage
            })
        except Exception as e:
            info['gpus'].append({
                'index': i,
                'name': gpu.name,
                'error': str(e)
            })
    
    return info

def init_retrain_route():
    bp = APIRouter(prefix="/retrain", tags=["retrain"])

    @bp.post(
        "",
        summary="Reentrenar modelo",
        description="""
        Inicia el reentrenamiento de un modelo específico. El entrenamiento se ejecuta en un hilo separado
        para no bloquear la API.
        
        **Proceso automático:**
        1. Se crea automáticamente una versión del modelo actual antes de sobrescribir
        2. Se detectan nuevas clases en los datos de entrenamiento
        3. Se ajusta el modelo si hay cambios en las clases
        4. Se entrena el modelo (usa GPU si está disponible)
        5. Se guarda el nuevo modelo
        6. Se recarga automáticamente y queda disponible para predicciones
        
        **Sistema de versionado:**
        - Cada reentrenamiento crea una nueva versión numerada del modelo anterior
        - Las versiones se guardan en `backups/` con nombres como `modelo_especies_v0001_20240101T120000.h5`
        - Se mantienen automáticamente las 3 versiones más recientes
        - Puedes restaurar versiones anteriores usando `/retrain/restore-version`
        
        **Notas:**
        - El entrenamiento puede tardar varios minutos dependiendo del tamaño del dataset
        - El modelo reentrenado queda disponible inmediatamente sin reiniciar la aplicación
        - Si hay error, el modelo anterior ya está guardado en versiones y puede restaurarse
        """,
        response_description="Estado del entrenamiento iniciado"
    )
    def retrain_model(model: str = Query(..., description="Modelo a reentrenar: especies, hojas o plantas")):
        if model not in ['especies', 'hojas', 'plantas']:
            raise HTTPException(
                status_code=400,
                detail="Debes especificar ?model=especies | hojas | plantas"
            )

        def train_thread(model_name):
            # Descargar imágenes verificadas de Cloudflare R2 ANTES del entrenamiento
            # Solo para el modelo de especies
            if model_name == 'especies':
                print(f"\n{'='*60}")
                print(f"Descargando imágenes verificadas de Cloudflare R2...")
                print(f"{'='*60}")
                
                # Verificar que las credenciales estén configuradas
                if all([
                    CLOUDFLARE_R2_BUCKET_NAME,
                    CLOUDFLARE_R2_ACCOUNT_ID,
                    CLOUDFLARE_R2_ACCESS_KEY_ID,
                    CLOUDFLARE_R2_SECRET_ACCESS_KEY
                ]):
                    try:
                        print(f"📥 Iniciando descarga desde bucket: {CLOUDFLARE_R2_BUCKET_NAME}")
                        stats = download_verified_images_from_r2(
                            bucket_name=CLOUDFLARE_R2_BUCKET_NAME,
                            account_id=CLOUDFLARE_R2_ACCOUNT_ID,
                            access_key_id=CLOUDFLARE_R2_ACCESS_KEY_ID,
                            secret_access_key=CLOUDFLARE_R2_SECRET_ACCESS_KEY,
                            prefix=CLOUDFLARE_R2_PREFIX,
                            base_dir=None  # Usa DATA_DIR/especies/train por defecto
                        )
                        print(f"✅ Descarga completada: {stats['downloaded']} nuevas, {stats['skipped']} omitidas, {stats['errors']} errores")
                        print(f"{'='*60}\n")
                    except Exception as e:
                        print(f"⚠️  Error al descargar imágenes de Cloudflare R2: {e}")
                        print(f"   Continuando con el entrenamiento con los datos existentes...")
                        print(f"{'='*60}\n")
                else:
                    print(f"⚠️  Credenciales de Cloudflare R2 no configuradas.")
                    print(f"   Para habilitar la descarga automática, configura las variables de entorno:")
                    print(f"   - CLOUDFLARE_R2_BUCKET_NAME")
                    print(f"   - CLOUDFLARE_R2_ACCOUNT_ID")
                    print(f"   - CLOUDFLARE_R2_ACCESS_KEY_ID")
                    print(f"   - CLOUDFLARE_R2_SECRET_ACCESS_KEY")
                    print(f"   Continuando con el entrenamiento con los datos existentes...")
                    print(f"{'='*60}\n")
            
            # Habilitar memory growth para todas las GPUs disponibles
            gpus = tf.config.list_physical_devices('GPU')
            print(f"\n{'='*60}")
            print(f"Información de GPU para entrenamiento de {model_name}")
            print(f"{'='*60}")
            
            if gpus:
                print(f"GPUs detectadas: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        details = tf.config.experimental.get_device_details(gpu)
                        print(f"  GPU {i}: {gpu.name}")
                        if details:
                            for key, value in details.items():
                                print(f"    {key}: {value}")
                    except Exception as e:
                        print(f"  GPU {i}: {gpu.name} - Error: {e}")
                print(f"{'='*60}\n")
                print("💡 TIP: Abre otra terminal y ejecuta 'watch -n 1 nvidia-smi' para monitorear el uso de GPU en tiempo real")
                print(f"{'='*60}\n")
                
                # CRÍTICO: Asegurar que TensorFlow use GPU por defecto
                # Configurar variables de entorno para forzar uso de GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                # Establecer que TensorFlow debe usar GPU
                tf.config.set_visible_devices(gpus[0], 'GPU')
                print("✅ GPU configurada como dispositivo visible para TensorFlow")
            else:
                print("⚠️  No se detectaron GPUs. Se usará CPU para entrenamiento.")
                print(f"{'='*60}\n")
            
            print(f"Reentrenando modelo {model_name}...")
            
            # Buscar el modelo actual (puede estar versionado o sin versión)
            from ..models_loader import find_model_path
            try:
                model_path = find_model_path('formas' if model_name == 'hojas' else model_name)
                print(f"📋 Modelo actual encontrado: {os.path.basename(model_path)}")
            except FileNotFoundError:
                # Si no existe modelo versionado, usar ruta sin versión (primera vez)
                model_file_map = {
                    'especies': 'modelo_especies.h5',
                    'hojas': 'modelo_hojas.h5',
                    'plantas': 'modelo_plantas.h5'
                }
                model_path = os.path.join(MODEL_DIR, model_file_map[model_name])
                print(f"📋 Modelo no encontrado, se creará nuevo: {os.path.basename(model_path)}")
            
            data_path = os.path.join(DATA_DIR, model_name)

            # Detectar clases en los datos DESPUÉS de descargar (si se descargaron)
            print(f"Detectando clases en los datos para {model_name}...")
            class_info = detect_new_classes(model_name)
            detected_classes = class_info['detected_classes']
            num_detected_classes = len(detected_classes)
            
            print(f"Clases detectadas en los datos: {num_detected_classes} ({detected_classes})")
            
            # Hacer backup de las clases actuales ANTES de actualizar
            from ..config import backup_classes, restore_classes, clear_classes_backup
            backup_created = backup_classes()
            
            # SIEMPRE actualizar la configuración con todas las clases detectadas en orden alfabético
            # Esto asegura que el orden en config.py coincida con el orden que usa ImageDataGenerator
            print(f"Actualizando configuración con todas las clases detectadas en orden alfabético...")
            config_updated = False
            try:
                if update_config_with_detected_classes(model_name, detected_classes):
                    # Recargar la configuración para aplicar los cambios
                    reload_config()
                    config_updated = True
                    print("✅ Configuración actualizada y recargada exitosamente.")
                else:
                    print("⚠️  Error al actualizar la configuración.")
            except Exception as config_error:
                print(f"⚠️  Error al actualizar la configuración: {config_error}")
                # Restaurar clases si hay error
                if backup_created:
                    restore_classes()
            
            if class_info['has_changes']:
                print(f"Nuevas clases detectadas: {class_info['new_classes']}")
                if class_info['removed_classes']:
                    print(f"Clases removidas: {class_info['removed_classes']}")
            else:
                print(f"No se detectaron cambios en las clases para {model_name} (solo actualización de orden)")

            # Detectar dispositivo automáticamente (GPU o CPU)
            gpus = tf.config.list_physical_devices('GPU')
            use_gpu = bool(gpus)
            
            if use_gpu:
                device = '/GPU:0'
                print(f"\n🖥️  Dispositivo detectado: GPU")
                print("✅ GPU disponible - El entrenamiento se ejecutará en GPU")
                # Asegurar que GPU es visible y configurada
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    print("   ✅ Memory growth habilitado para GPU")
                except Exception as e:
                    print(f"   ⚠️  Advertencia al configurar memory growth: {e}")
            else:
                device = '/CPU:0'
                print(f"\n🖥️  Dispositivo detectado: CPU")
                print("⚠️  No se detectó GPU - El entrenamiento se ejecutará en CPU")
                print("   💡 El entrenamiento será más lento pero funcionará correctamente")

            # Cargar y entrenar en GPU si hay, de lo contrario en CPU
            # Envolver todo el proceso de entrenamiento en try-except para restaurar clases si falla
            try:
                with tf.device(device):
                    model = tf.keras.models.load_model(model_path)
                    
                    # Verificar el número de clases en el modelo actual
                    # Manejar casos donde output puede ser una lista o un tensor
                    if isinstance(model.output, list):
                        current_model_classes = model.output[0].shape[-1]
                    else:
                        current_model_classes = model.output.shape[-1]
                    print(f"Clases en el modelo actual: {current_model_classes}")
                    print(f"Clases detectadas en los datos: {num_detected_classes}")
                    
                    # Ajustar modelo si el número de clases no coincide
                    if current_model_classes != num_detected_classes:
                        print(f"Ajustando modelo: de {current_model_classes} a {num_detected_classes} clases...")
                        
                        # Calcular cuántas clases faltan
                        classes_to_add = num_detected_classes - current_model_classes
                        
                        if classes_to_add > 0:
                            # Obtener las clases nuevas detectadas directamente
                            new_classes_from_detection = class_info.get('new_classes', [])
                            
                            # Crear un generador temporal para obtener el orden exacto de clases que usa ImageDataGenerator
                            datagen = ImageDataGenerator(rescale=1./255)
                            temp_gen = datagen.flow_from_directory(
                                os.path.join(data_path, "train"),
                                target_size=(128, 128),
                                batch_size=1,
                                class_mode='categorical',
                                shuffle=False
                            )
                            # Obtener todas las clases en el orden que el generador las tiene (orden alfabético)
                            all_classes_sorted = sorted(temp_gen.class_indices.keys(), key=lambda x: temp_gen.class_indices[x])
                            
                            # Obtener las clases actuales del modelo desde la configuración
                            current_classes_from_config = class_info.get('current_classes', [])
                            
                            # Las clases nuevas son las que están en detected_classes pero no en current_classes
                            # Usar el orden del generador para mantener consistencia
                            new_classes_needed = [cls for cls in all_classes_sorted if cls not in current_classes_from_config]
                            
                            # Validar que tenemos exactamente classes_to_add clases
                            if len(new_classes_needed) != classes_to_add:
                                print(f"⚠️  Advertencia: Discrepancia en número de clases nuevas.")
                                print(f"   Esperadas: {classes_to_add}, Calculadas: {len(new_classes_needed)}")
                                print(f"   Clases nuevas detectadas: {new_classes_from_detection}")
                                print(f"   Clases nuevas calculadas: {new_classes_needed}")
                                # Si hay discrepancia, usar las clases nuevas detectadas directamente
                                # pero asegurarse de que estén en el orden del generador
                                if new_classes_from_detection:
                                    new_classes_needed = [cls for cls in all_classes_sorted if cls in new_classes_from_detection]
                            
                            print(f"Agregando {len(new_classes_needed)} clases al modelo: {new_classes_needed}")
                            model = adjust_model_for_new_classes(model, model_name, new_classes_needed)
                            print("Modelo ajustado exitosamente.")
                            
                            # Limpiar el generador temporal
                            del temp_gen
                        else:
                            # El modelo tiene más clases que las detectadas - esto es problemático
                            # Por ahora, creamos un nuevo modelo con el número correcto de clases
                            print(f"ADVERTENCIA: El modelo tiene más clases ({current_model_classes}) que las detectadas ({num_detected_classes})")
                            print("Reconstruyendo la capa de salida...")
                            
                            # Obtener la penúltima capa
                            penultimate_layer = model.layers[-2]
                            
                            # Crear nueva capa de salida con el número correcto de clases
                            if model_name == 'plantas':
                                new_output = tf.keras.layers.Dense(
                                    num_detected_classes,
                                    activation='sigmoid',
                                    name='new_output'
                                )(penultimate_layer.output)
                            else:
                                new_output = tf.keras.layers.Dense(
                                    num_detected_classes,
                                    activation='softmax',
                                    name='new_output'
                                )(penultimate_layer.output)
                            
                            # Crear nuevo modelo
                            model = tf.keras.Model(inputs=model.input, outputs=new_output)
                            print("Modelo reconstruido con el número correcto de clases.")
                    else:
                        print("El modelo ya tiene el número correcto de clases.")

                    # Detectar recursos disponibles y ajustar batch size automáticamente
                    if use_gpu:
                        # Obtener información de VRAM disponible
                        available_memory_gb = None
                        total_memory = None
                        
                        try:
                            gpu_devices = tf.config.list_physical_devices('GPU')
                            if gpu_devices:
                                # Intentar obtener información de memoria de GPU usando TensorFlow
                                try:
                                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                                    total_memory = memory_info['limit'] / (1024**3)  # Convertir a GB
                                    # Estimar memoria disponible (restar ~500MB para sistema y otros procesos)
                                    available_memory_gb = total_memory - 0.5
                                    print(f"   💾 VRAM total detectada (TensorFlow): {total_memory:.2f} GB")
                                    print(f"   💾 VRAM disponible estimada: {available_memory_gb:.2f} GB")
                                except:
                                    # Fallback: usar detalles de dispositivo
                                    try:
                                        details = tf.config.experimental.get_device_details(gpu_devices[0])
                                        if 'device_memory_size' in details:
                                            total_memory = details['device_memory_size'] / (1024**3)
                                            available_memory_gb = total_memory - 0.5
                                            print(f"   💾 VRAM total detectada (TensorFlow details): {total_memory:.2f} GB")
                                            print(f"   💾 VRAM disponible estimada: {available_memory_gb:.2f} GB")
                                    except:
                                        pass
                                
                                # Si TensorFlow no pudo obtener la info, intentar con nvidia-smi
                                if available_memory_gb is None:
                                    try:
                                        import subprocess
                                        result = subprocess.run(
                                            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                            capture_output=True,
                                            text=True,
                                            timeout=3
                                        )
                                        if result.returncode == 0:
                                            memory_total_mb = float(result.stdout.strip().split('\n')[0])
                                            total_memory = memory_total_mb / 1024  # Convertir MB a GB
                                            available_memory_gb = total_memory - 0.5
                                            print(f"   💾 VRAM total detectada (nvidia-smi): {total_memory:.2f} GB")
                                            print(f"   💾 VRAM disponible estimada: {available_memory_gb:.2f} GB")
                                    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                                        pass
                                
                                # Si aún no tenemos info, usar valores conservadores
                                if available_memory_gb is None:
                                    available_memory_gb = 1.5  # Asumir ~1.5GB disponible
                                    print(f"   ⚠️  No se pudo detectar VRAM exacta, usando estimación conservadora: {available_memory_gb:.2f} GB")
                                    print(f"   💡 Puedes verificar manualmente con: nvidia-smi")
                            else:
                                available_memory_gb = 1.5
                                print(f"   ⚠️  No se detectaron GPUs, usando estimación conservadora: {available_memory_gb:.2f} GB")
                        except Exception as e:
                            available_memory_gb = 1.5
                            print(f"   ⚠️  Error detectando VRAM: {e}")
                            print(f"   📦 Usando estimación conservadora: {available_memory_gb:.2f} GB")
                        
                        # Calcular batch_size óptimo según VRAM disponible
                        # Estimación: cada imagen 128x128x3 usa ~0.2MB en memoria (con gradientes y overhead)
                        # Con modelo cargado, reservamos ~2GB, así que usamos el resto para batches
                        # batch_size * 0.2MB * 4 (overhead) = memoria por batch
                        # Queremos usar ~60% de la memoria disponible para batches
                        memory_for_batches_gb = available_memory_gb * 0.6
                        memory_for_batches_mb = memory_for_batches_gb * 1024
                        
                        # Cada batch usa aproximadamente: batch_size * 0.2MB * 4 = batch_size * 0.8MB
                        # batch_size = memory_for_batches_mb / 0.8
                        estimated_batch_size = int(memory_for_batches_mb / 0.8)
                        
                        # Ajustar según rangos de VRAM conocidos
                        if available_memory_gb >= 8:
                            # GPU con 8GB+ (RTX 3070, 3080, etc.)
                            batch_size = min(estimated_batch_size, 256)
                        elif available_memory_gb >= 4:
                            # GPU con 4-8GB (GTX 1650 Ti, RTX 2060, etc.)
                            batch_size = min(estimated_batch_size, 128)
                        elif available_memory_gb >= 2:
                            # GPU con 2-4GB
                            batch_size = min(estimated_batch_size, 64)
                        else:
                            # GPU con menos de 2GB disponible
                            batch_size = min(estimated_batch_size, 32)
                        
                        # Asegurar batch_size mínimo
                        if batch_size < 8:
                            batch_size = 8
                        
                        print(f"   📦 Batch size calculado automáticamente: {batch_size} (basado en {available_memory_gb:.2f} GB VRAM disponible)")
                        print(f"   ⚠️  Si hay OOM, se reducirá automáticamente")
                    else:
                        # CPU: usar batch size más pequeño
                        batch_size = 32
                        print(f"   📦 Batch size para CPU: {batch_size}")
                        
                    print(f"   📦 Configurando batch size: {batch_size} {'(GPU optimizado)' if use_gpu else '(CPU)'}")
                    
                    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
                    train_gen = datagen.flow_from_directory(
                        os.path.join(data_path, "train"),
                        target_size=(128, 128),
                        batch_size=batch_size,
                        subset='training',
                        class_mode='categorical'
                    )
                    val_gen = datagen.flow_from_directory(
                        os.path.join(data_path, "train"),
                        target_size=(128, 128),
                        batch_size=batch_size,
                        subset='validation',
                        class_mode='categorical'
                    )
                    
                    # Verificar que el número de clases del generador coincide con el modelo
                    train_num_classes = train_gen.num_classes
                    # Manejar casos donde output puede ser una lista o un tensor
                    if isinstance(model.output, list):
                        model_output_classes = model.output[0].shape[-1]
                    else:
                        model_output_classes = model.output.shape[-1]
                    print(f"Verificación final: Generador tiene {train_num_classes} clases, Modelo tiene {model_output_classes} clases")
                    
                    if train_num_classes != model_output_classes:
                        raise ValueError(
                            f"Incompatibilidad: El generador tiene {train_num_classes} clases pero el modelo tiene {model_output_classes} clases. "
                            f"Ajuste el modelo antes de continuar."
                        )

                # Verificar en qué dispositivo está el modelo antes de compilar
                print(f"\n📍 Verificando dispositivo del modelo...")
                if hasattr(model, 'layers') and len(model.layers) > 0:
                    # Verificar la primera capa para ver dónde está el modelo
                    first_layer = model.layers[0]
                    if hasattr(first_layer, 'weights') and len(first_layer.weights) > 0:
                        device_location = first_layer.weights[0].device
                        print(f"   Dispositivo del modelo: {device_location}")
                        if '/GPU' in str(device_location):
                            print("   ✅ Modelo está en GPU")
                        else:
                            print("   ⚠️  Modelo está en CPU")
                
                # Forzar uso de GPU explícitamente si está disponible
                if use_gpu:
                    print(f"\n🔧 Configurando para forzar uso de GPU...")
                    # Verificar que TensorFlow vea la GPU
                    available_gpus = tf.config.list_physical_devices('GPU')
                    print(f"   GPUs físicas: {len(available_gpus)}")
                    print(f"   GPUs lógicas: {len(tf.config.list_logical_devices('GPU'))}")
                    
                    if available_gpus:
                        # Asegurar que todas las operaciones se ejecuten en GPU
                        # Configurar para que TensorFlow use GPU por defecto
                        try:
                            tf.config.experimental.set_memory_growth(available_gpus[0], True)
                            print("   ✅ GPU configurada con memory growth")
                        except Exception as e:
                            print(f"   ⚠️  Error configurando memory growth: {e}")
                
                # Compilar modelo - esto debe hacerse dentro del contexto de GPU
                print(f"\n⚙️  Compilando modelo...")
                model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

                # Verificar dispositivo después de compilar
                print(f"   Verificando dispositivo después de compilar...")
                if hasattr(model, 'layers') and len(model.layers) > 0:
                    first_layer = model.layers[0]
                    if hasattr(first_layer, 'weights') and len(first_layer.weights) > 0:
                        device_location = first_layer.weights[0].device
                        print(f"   Dispositivo: {device_location}")
                        if '/GPU' in str(device_location):
                            print("   ✅ Modelo compilado en GPU")
                        else:
                            print("   ⚠️  Modelo compilado en CPU - esto puede afectar el rendimiento")
                
                # Ejecutar entrenamiento dentro del contexto de GPU
                print(f"\n🚀 Iniciando entrenamiento en {device}...")
                print(f"   Batch size: {batch_size}")
                print(f"   Epochs: 5")
                print(f"   ⚠️  IMPORTANTE: Verifica nvidia-smi - deberías ver 80-100% de utilización\n")
                
                # Configurar para maximizar uso de GPU
                if use_gpu:
                    print(f"   🔧 Optimizaciones para GPU:")
                    
                    # IMPORTANTE: Deshabilitar mixed precision si causa problemas de memoria
                    # Mixed precision puede causar OOM en GPUs con poca memoria
                    use_mixed_precision = False  # Deshabilitado por defecto para evitar OOM
                    
                    if use_mixed_precision:
                        try:
                            policy = tf.keras.mixed_precision.Policy('mixed_float16')
                            tf.keras.mixed_precision.set_global_policy(policy)
                            print("      ✅ Mixed precision habilitado (puede aumentar memoria)")
                        except Exception as e:
                            print(f"      ℹ️  Mixed precision no disponible")
                    else:
                        print("      ℹ️  Mixed precision deshabilitado (para evitar OOM en 4GB VRAM)")
                    
                    # Configurar para usar GPU agresivamente
                    try:
                        tf.config.optimizer.set_jit(True)  # Habilitar XLA JIT
                        print("      ✅ XLA JIT habilitado")
                    except Exception as e:
                        print(f"      ℹ️  XLA JIT no disponible")
                    
                    # Nota: Los threads ya están configurados en app/config.py al inicio
                    print("      ✅ Threads ya configurados al inicio")
                
                # Ejecutar entrenamiento - ya estamos dentro del contexto with tf.device(device)
                # por lo que todas las operaciones se ejecutarán en el dispositivo correcto
                print(f"\n   ⚡ Ejecutando entrenamiento...")
                print(f"   📊 Monitorea nvidia-smi - deberías ver 80-100% de utilización")
                print(f"   💡 Si ves baja utilización, el modelo puede ser pequeño - considera aumentar batch_size\n")
                
                # Intentar entrenar con manejo de errores de memoria
                try:
                    # NO limpiar sesión aquí - el modelo ya está cargado y compilado
                    # Solo hacer garbage collection para liberar memoria del sistema
                    if use_gpu:
                        import gc
                        gc.collect()
                        print(f"   🧹 Memoria del sistema limpiada")
                    
                    model.fit(
                        train_gen, 
                        epochs=5, 
                        validation_data=val_gen, 
                        verbose=1
                    )
                except (tf.errors.ResourceExhaustedError, RuntimeError, Exception) as e:
                    error_msg = str(e)
                    # Detectar errores de memoria de GPU
                    is_oom = ('OOM' in error_msg or 
                             'out of memory' in error_msg.lower() or 
                             'ResourceExhaustedError' in error_msg or 
                             'ran out of memory' in error_msg.lower() or
                             'Allocator' in error_msg and 'memory' in error_msg.lower())
                    
                    if is_oom:
                        # Si hay error de memoria, reducir batch size y reintentar
                        print(f"\n   ⚠️  ERROR: Memoria de GPU insuficiente con batch_size={batch_size}")
                        print(f"   🔄 Reduciendo batch_size a {batch_size // 2} y reintentando...\n")
                        
                        # Limpiar memoria de GPU pero NO usar clear_session (destruye el modelo)
                        import gc
                        gc.collect()
                        
                        # Reducir batch size
                        batch_size = batch_size // 2
                        if batch_size < 8:
                            print(f"   ❌ Batch size muy pequeño ({batch_size}). El entrenamiento puede ser muy lento.")
                            print(f"   💡 Considera reducir el tamaño del modelo o usar CPU.")
                            raise RuntimeError(f"Batch size demasiado pequeño después de reducir por OOM: {batch_size}")
                        
                        # Recrear generadores con nuevo batch size
                        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
                        train_gen = datagen.flow_from_directory(
                            os.path.join(data_path, "train"),
                            target_size=(128, 128),
                            batch_size=batch_size,
                            subset='training',
                            class_mode='categorical'
                        )
                        val_gen = datagen.flow_from_directory(
                            os.path.join(data_path, "train"),
                            target_size=(128, 128),
                            batch_size=batch_size,
                            subset='validation',
                            class_mode='categorical'
                        )
                        
                        # Recompilar modelo para asegurar que está en GPU
                        print(f"   🔄 Recompilando modelo con batch_size={batch_size}...")
                        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                                      loss='categorical_crossentropy',
                                      metrics=['accuracy'])
                        
                        print(f"   ✅ Batch size reducido a {batch_size}, reintentando entrenamiento en GPU...\n")
                        # Asegurar que estamos en GPU al reintentar
                        with tf.device('/GPU:0'):
                            model.fit(
                                train_gen, 
                                epochs=5, 
                                validation_data=val_gen, 
                                verbose=1
                            )
                    else:
                        # Re-lanzar el error si no es de memoria
                        raise
                
                    # Si llegamos aquí, el entrenamiento fue exitoso
                    print(f"\n✅ Entrenamiento completado exitosamente para {model_name}")

            except Exception as training_error:
                # Si hay cualquier error durante el entrenamiento, restaurar clases
                print(f"\n❌ ERROR durante el entrenamiento del modelo {model_name}: {training_error}")
                print(f"   Tipo de error: {type(training_error).__name__}")
                import traceback
                traceback.print_exc()
                
                # Restaurar clases desde backup si existe
                if backup_created:
                    print("🔄 Restaurando clases desde backup debido al error en el entrenamiento...")
                    restore_classes()
                    print("✅ Clases restauradas a su estado anterior.")
                
                # Re-lanzar el error para que se maneje en el bloque except externo
                raise

            # Variable para almacenar la versión creada (para restauración en caso de error)
            previous_version_info = None
            version_created = False
            
            try:
                # Convertir 'hojas' a 'formas' para el sistema de versionado
                # (el endpoint usa 'hojas' pero internamente se usa 'formas')
                versioning_model_name = 'formas' if model_name == 'hojas' else model_name
                
                # Buscar si existe un modelo en MODEL_DIR (versionado o sin versión)
                model_file_map = {
                    'especies': 'modelo_especies.h5',
                    'hojas': 'modelo_hojas.h5',
                    'plantas': 'modelo_plantas.h5'
                }
                model_base_name = os.path.splitext(model_file_map[model_name])[0]
                
                import glob
                
                # Función auxiliar para ordenar por timestamp (más reciente primero)
                def get_timestamp_from_filename(filename):
                    basename = os.path.basename(filename)
                    parts = basename.split('_')
                    if len(parts) >= 3:
                        timestamp = parts[-1].replace('.h5', '')
                        return timestamp
                    return ''
                
                # Buscar modelo versionado en MODEL_DIR
                pattern_model = os.path.join(MODEL_DIR, f"{model_base_name}_v*_*.h5")
                versioned_models = glob.glob(pattern_model)
                if versioned_models:
                    versioned_models.sort(key=get_timestamp_from_filename, reverse=True)
                    existing_model = versioned_models[0]
                else:
                    existing_model = None
                
                # Si no hay modelo versionado, buscar sin versión
                if not existing_model:
                    existing_model = os.path.join(MODEL_DIR, model_file_map[model_name])
                    if not os.path.exists(existing_model):
                        existing_model = None
                
                # Crear versión del modelo actual ANTES de guardar el nuevo
                # Esto preserva el modelo anterior en backups
                if existing_model and os.path.exists(existing_model):
                    try:
                        from datetime import datetime
                        previous_version_info = create_model_version(
                            versioning_model_name, 
                            version_notes=f"Reentrenamiento automático - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        version_created = True
                        print(f"✅ Versión {previous_version_info['version']} del modelo {model_name} anterior guardada en backups: {previous_version_info['filename']}")
                    except FileNotFoundError as version_error:
                        print(f"⚠️  No se pudo crear versión del modelo {model_name} anterior: {version_error}")
                        print(f"   Esto puede ocurrir si es la primera vez que se entrena este modelo.")
                        print(f"   Continuando con el guardado del nuevo modelo...")
                    except Exception as version_error:
                        print(f"⚠️  Error creando versión del modelo {model_name} anterior: {version_error}")
                        print(f"   Continuando con el guardado del nuevo modelo...")
                else:
                    print(f"ℹ️  Modelo {model_name} no existe aún, se creará la versión 1 sin backup previo.")
                
                # Generar nombre versionado para el nuevo modelo
                from datetime import datetime
                from ..utils.model_versioning import load_version_metadata
                metadata = load_version_metadata(versioning_model_name)
                versions = metadata.get('versions', [])
                
                # Calcular siguiente versión (si no se creó versión antes, será 1)
                if version_created and previous_version_info:
                    next_version = previous_version_info['version'] + 1
                elif versions:
                    next_version = max(v['version'] for v in versions) + 1
                else:
                    next_version = 1
                
                timestamp = datetime.utcnow()
                timestamp_str = timestamp.strftime('%Y%m%dT%H%M%S')
                new_version_filename = f"{model_base_name}_v{next_version:04d}_{timestamp_str}.h5"
                new_model_path = os.path.join(MODEL_DIR, new_version_filename)
                
                # Guardar nuevo modelo con nombre versionado
                print(f"💾 Guardando nuevo modelo entrenado como versión {next_version} en {MODEL_DIR}: {new_version_filename}")
                model.save(new_model_path)

                # Validar que el modelo guardado se puede cargar
                _ = tf.keras.models.load_model(new_model_path)
                print(f"✅ Modelo {model_name} versión {next_version} guardado y validado: {new_version_filename}")
                
                # Recargar el modelo en el sistema global para que esté disponible para predicciones
                print(f"Recargando modelo {model_name} en el sistema de predicción...")
                try:
                    # Convertir 'hojas' a 'formas' para reload_model
                    # (el endpoint usa 'hojas' pero internamente se usa 'formas')
                    reload_model_name = 'formas' if model_name == 'hojas' else model_name
                    reload_model(reload_model_name)
                    print(f"✅ Modelo {model_name} recargado y disponible para predicciones.")
                    
                    # Si todo fue exitoso, limpiar el backup de clases
                    from ..config import clear_classes_backup
                    clear_classes_backup()
                except Exception as reload_error:
                    print(f"⚠️  Error al recargar modelo {model_name} para predicciones: {reload_error}")
                    print(f"   El modelo fue guardado correctamente, pero será necesario reiniciar la aplicación para usarlo.")
                    # Restaurar clases si hay error en recarga
                    if backup_created:
                        restore_classes()
                        raise  # Re-lanzar el error para que se maneje en el bloque except externo
                    
            except Exception as save_error:
                # Si hay error al guardar o validar el modelo, restaurar la versión anterior
                print(f"\n❌ ERROR al guardar o validar el modelo {model_name}: {save_error}")
                
                # Restaurar clases desde backup si existe
                from ..config import restore_classes
                if backup_created:
                    print("🔄 Restaurando clases desde backup debido al error...")
                    restore_classes()
                
                if version_created and previous_version_info:
                    try:
                        from ..utils.model_versioning import restore_model_version
                        versioning_model_name = 'formas' if model_name == 'hojas' else model_name
                        print(f"🔄 Restaurando versión anterior (v{previous_version_info['version']}) del modelo {model_name}...")
                        restore_result = restore_model_version(versioning_model_name, previous_version_info['version'])
                        
                        # Recargar el modelo restaurado
                        reload_model_name = 'formas' if model_name == 'hojas' else model_name
                        reload_model(reload_model_name)
                        print(f"✅ Modelo {model_name} restaurado a la versión {previous_version_info['version']} y recargado.")
                        print(f"   El modelo anterior está disponible para predicciones.")
                    except Exception as restore_error:
                        print(f"❌ ERROR crítico: No se pudo restaurar la versión anterior: {restore_error}")
                        print(f"   Por favor, restaura manualmente usando: POST /retrain/restore-version?model={model_name}&version={previous_version_info['version']}")
                else:
                    print(f"⚠️  No se pudo restaurar automáticamente porque no se creó versión previa.")
                    print(f"   El modelo original debería estar intacto si no se guardó el nuevo.")
                
                # Re-lanzar el error para que se registre en el estado del entrenamiento
                raise

        # Detectar clases antes de iniciar el entrenamiento para mostrar información inmediata
        class_info = detect_new_classes(model)
        
        threading.Thread(target=train_thread, args=(model,)).start()

        response = {
            "status": "Entrenamiento iniciado",
            "model": model,
            "classes_detected": class_info['detected_classes'],
            "current_classes": class_info['current_classes'],
            "new_classes": class_info['new_classes'],
            "removed_classes": class_info['removed_classes'],
            "has_changes": class_info['has_changes']
        }
        
        if class_info['has_changes']:
            response["message"] = f"Se detectaron {len(class_info['new_classes'])} nuevas clases y {len(class_info['removed_classes'])} clases removidas. El modelo será ajustado automáticamente."
        else:
            response["message"] = "No se detectaron cambios en las clases. El modelo será reentrenado con las clases existentes."

        return response
    
    @bp.get(
        "/check-classes",
        summary="Verificar clases disponibles",
        description="""
        Verifica las clases disponibles en los datos de entrenamiento sin iniciar el entrenamiento.
        Útil para ver qué clases se detectarían antes de reentrenar.
        """
    )
    def check_classes(model: str = Query(..., description="Modelo a verificar: especies, hojas o plantas")):
        if model not in ['especies', 'hojas', 'plantas']:
            raise HTTPException(
                status_code=400,
                detail="Debes especificar ?model=especies | hojas | plantas"
            )
        
        class_info = detect_new_classes(model)
        
        return {
            "model": model,
            "classes_detected": class_info['detected_classes'],
            "current_classes": class_info['current_classes'],
            "new_classes": class_info['new_classes'],
            "removed_classes": class_info['removed_classes'],
            "has_changes": class_info['has_changes'],
            "message": f"Clases detectadas: {len(class_info['detected_classes'])}, Clases actuales: {len(class_info['current_classes'])}"
        }
    
    @bp.post(
        "/update-config",
        summary="Actualizar configuración",
        description="""
        Actualiza la configuración de la aplicación con nuevas clases detectadas en los datos.
        Esto actualiza las listas de clases en `config.py` sin necesidad de reentrenar.
        """
    )
    def update_config(model: str = Query(..., description="Modelo a actualizar: especies, hojas o plantas")):
        if model not in ['especies', 'hojas', 'plantas']:
            raise HTTPException(
                status_code=400,
                detail="Debes especificar ?model=especies | hojas | plantas"
            )
        
        class_info = detect_new_classes(model)
        
        if class_info['new_classes']:
            if update_config_with_new_classes(model, class_info['new_classes']):
                reload_config()
                return {
                    "status": "success",
                    "model": model,
                    "new_classes_added": class_info['new_classes'],
                    "message": f"Configuración actualizada con {len(class_info['new_classes'])} nuevas clases"
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Error al actualizar la configuración"
                )
        else:
            return {
                "status": "info",
                "message": "No hay nuevas clases para actualizar"
            }
    
    @bp.get(
        "/gpu-status",
        summary="Estado de GPU",
        description="""
        Obtiene información detallada sobre el estado y uso de las GPUs disponibles.
        Incluye información de TensorFlow y nvidia-smi (si está disponible).
        """
    )
    def gpu_status():
        try:
            import subprocess
            import json
            
            gpu_info = get_gpu_info()
            
            # Intentar obtener información detallada con nvidia-smi si está disponible
            nvidia_smi_info = None
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    nvidia_smi_info = []
                    for line in lines:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            nvidia_smi_info.append({
                                'index': parts[0],
                                'name': parts[1],
                                'gpu_utilization_percent': parts[2],
                                'memory_utilization_percent': parts[3],
                                'memory_used_mb': parts[4],
                                'memory_total_mb': parts[5],
                                'temperature_celsius': parts[6]
                            })
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                pass  # nvidia-smi no está disponible o falló
            
            return {
                "tensorflow_gpu_info": gpu_info,
                "nvidia_smi_info": nvidia_smi_info,
                "message": "Usa 'watch -n 1 nvidia-smi' en terminal para monitoreo en tiempo real" if nvidia_smi_info else "nvidia-smi no disponible. Instálalo para ver uso detallado."
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error obteniendo información de GPU: {str(e)}")
    
    @bp.get(
        "/versions",
        summary="Listar versiones de modelo",
        description="""
        Lista todas las versiones disponibles de un modelo guardadas en backups.
        
        **Información de cada versión:**
        - Número de versión (secuencial)
        - Timestamp de creación
        - Nombre del archivo
        - Notas/descripción
        - Tamaño del archivo
        
        **Notas:**
        - Se mantienen automáticamente las 3 versiones más recientes (configurable)
        - Las versiones se ordenan de más reciente a más antigua
        - Solo se muestran versiones cuyos archivos aún existen
        """,
        response_description="Lista de versiones disponibles del modelo"
    )
    def list_versions(model: str = Query(..., description="Modelo a consultar: especies, hojas o plantas")):
        if model not in ['especies', 'hojas', 'plantas']:
            raise HTTPException(
                status_code=400,
                detail="Debes especificar ?model=especies | hojas | plantas"
            )
        
        try:
            # Convertir 'hojas' a 'formas' para el sistema de versionado
            versioning_model_name = 'formas' if model == 'hojas' else model
            versions = list_model_versions(versioning_model_name)
            return {
                "model": model,
                "total_versions": len(versions),
                "versions": versions,
                "message": f"Se encontraron {len(versions)} versiones del modelo {model}"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listando versiones: {str(e)}")
    
    @bp.get(
        "/version-info",
        summary="Información de versión específica",
        description="""
        Obtiene información detallada de una versión específica de un modelo.
        
        **Información incluida:**
        - Número de versión
        - Timestamp de creación
        - Ruta del archivo
        - Notas/descripción
        - Tamaño del archivo en bytes
        """
    )
    def get_version(model: str = Query(..., description="Modelo a consultar: especies, hojas o plantas"),
                    version: int = Query(..., description="Número de versión a consultar")):
        if model not in ['especies', 'hojas', 'plantas']:
            raise HTTPException(
                status_code=400,
                detail="Debes especificar ?model=especies | hojas | plantas"
            )
        
        try:
            # Convertir 'hojas' a 'formas' para el sistema de versionado
            versioning_model_name = 'formas' if model == 'hojas' else model
            version_info = get_version_info(versioning_model_name, version)
            if version_info is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No se encontró la versión {version} del modelo {model}"
                )
            return {
                "model": model,
                "version": version,
                "version_info": version_info
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error obteniendo información de versión: {str(e)}")
    
    @bp.post(
        "/restore-version",
        summary="Restaurar versión de modelo",
        description="""
        Restaura una versión específica de un modelo, reemplazando el modelo actual.
        
        **Proceso de restauración:**
        1. Se crea un backup del modelo actual antes de restaurar
        2. Se copia la versión seleccionada al directorio de modelos activos
        3. Se valida que el modelo restaurado se puede cargar correctamente
        4. Se recarga automáticamente en el sistema de predicción
        5. El modelo restaurado queda disponible inmediatamente para predicciones
        
        **Notas:**
        - El modelo actual se reemplaza por la versión seleccionada
        - Si hay error durante la validación, se restaura automáticamente el modelo original
        - El modelo restaurado se recarga sin necesidad de reiniciar la aplicación
        - Útil para revertir a una versión anterior si el nuevo modelo no funciona bien
        """,
        response_description="Resultado de la restauración con información de la versión"
    )
    def restore_version(model: str = Query(..., description="Modelo a restaurar: especies, hojas o plantas"),
                       version: int = Query(..., description="Número de versión a restaurar")):
        if model not in ['especies', 'hojas', 'plantas']:
            raise HTTPException(
                status_code=400,
                detail="Debes especificar ?model=especies | hojas | plantas"
            )
        
        try:
            # Convertir 'hojas' a 'formas' para el sistema de versionado
            versioning_model_name = 'formas' if model == 'hojas' else model
            restore_result = restore_model_version(versioning_model_name, version)
            
            # Recargar el modelo restaurado en el sistema global
            print(f"Recargando modelo {model} restaurado en el sistema de predicción...")
            try:
                # Convertir 'hojas' a 'formas' para reload_model
                reload_model_name = 'formas' if model == 'hojas' else model
                reload_model(reload_model_name)
                print(f"✅ Modelo {model} restaurado y recargado correctamente.")
                restore_result['reloaded'] = True
            except Exception as reload_error:
                print(f"⚠️  Error al recargar modelo {model} restaurado: {reload_error}")
                restore_result['reloaded'] = False
                restore_result['reload_error'] = str(reload_error)
            
            return {
                "status": "success",
                "message": f"Modelo {model} restaurado a la versión {version}",
                **restore_result
            }
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error restaurando versión: {str(e)}")
    
    return bp
