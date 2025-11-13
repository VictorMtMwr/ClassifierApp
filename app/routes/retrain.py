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

    @bp.post("")
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
            model_path = os.path.join(MODEL_DIR, f"modelo_{model_name}.h5")
            data_path = os.path.join(DATA_DIR, model_name)

            # Detectar clases en los datos DESPUÉS de descargar (si se descargaron)
            print(f"Detectando clases en los datos para {model_name}...")
            class_info = detect_new_classes(model_name)
            detected_classes = class_info['detected_classes']
            num_detected_classes = len(detected_classes)
            
            print(f"Clases detectadas en los datos: {num_detected_classes} ({detected_classes})")
            
            # SIEMPRE actualizar la configuración con todas las clases detectadas en orden alfabético
            # Esto asegura que el orden en config.py coincida con el orden que usa ImageDataGenerator
            print(f"Actualizando configuración con todas las clases detectadas en orden alfabético...")
            if update_config_with_detected_classes(model_name, detected_classes):
                # Recargar la configuración para aplicar los cambios
                reload_config()
                print("✅ Configuración actualizada y recargada exitosamente.")
            else:
                print("⚠️  Error al actualizar la configuración.")
            
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
                        # Obtener las clases nuevas detectadas
                        new_classes_from_config = class_info.get('new_classes', [])
                        
                        # Crear un generador temporal para obtener el orden exacto de clases
                        datagen = ImageDataGenerator(rescale=1./255)
                        temp_gen = datagen.flow_from_directory(
                            os.path.join(data_path, "train"),
                            target_size=(128, 128),
                            batch_size=1,
                            class_mode='categorical',
                            shuffle=False
                        )
                        # Obtener todas las clases en el orden que el generador las tiene
                        all_classes_sorted = sorted(temp_gen.class_indices.keys(), key=lambda x: temp_gen.class_indices[x])
                        
                        # Las clases que el modelo tiene actualmente son las primeras current_model_classes
                        model_current_classes = all_classes_sorted[:current_model_classes]
                        
                        # Las clases que faltan son las que están en all_classes_sorted pero no en model_current_classes
                        new_classes_needed = [cls for cls in all_classes_sorted if cls not in model_current_classes]
                        
                        # Asegurarse de que tenemos exactamente classes_to_add clases
                        if len(new_classes_needed) != classes_to_add:
                            # Si hay discrepancia, tomar las últimas classes_to_add clases
                            new_classes_needed = all_classes_sorted[-classes_to_add:]
                        
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
                    try:
                        gpu_devices = tf.config.list_physical_devices('GPU')
                        if gpu_devices:
                            # Intentar obtener información de memoria de GPU
                            try:
                                memory_info = tf.config.experimental.get_memory_info('GPU:0')
                                total_memory = memory_info['limit'] / (1024**3)  # Convertir a GB
                                # Estimar memoria disponible (restar ~500MB para sistema y otros procesos)
                                available_memory_gb = total_memory - 0.5
                                print(f"   💾 VRAM total detectada: {total_memory:.2f} GB")
                                print(f"   💾 VRAM disponible estimada: {available_memory_gb:.2f} GB")
                            except:
                                # Fallback: usar detalles de dispositivo
                                try:
                                    details = tf.config.experimental.get_device_details(gpu_devices[0])
                                    if 'device_memory_size' in details:
                                        total_memory = details['device_memory_size'] / (1024**3)
                                        available_memory_gb = total_memory - 0.5
                                        print(f"   💾 VRAM total detectada: {total_memory:.2f} GB")
                                        print(f"   💾 VRAM disponible estimada: {available_memory_gb:.2f} GB")
                                    else:
                                        # Si no podemos obtener info, usar valores conservadores
                                        available_memory_gb = 1.5  # Asumir ~1.5GB disponible
                                        print(f"   ⚠️  No se pudo detectar VRAM exacta, usando estimación conservadora: {available_memory_gb:.2f} GB")
                                except:
                                    available_memory_gb = 1.5
                                    print(f"   ⚠️  No se pudo detectar VRAM, usando estimación conservadora: {available_memory_gb:.2f} GB")
                            
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
                            batch_size = 64
                            print(f"   📦 Batch size por defecto: {batch_size} (no se detectó GPU)")
                    except Exception as e:
                        print(f"   ⚠️  Error detectando VRAM: {e}")
                        batch_size = 64  # Fallback conservador
                        print(f"   📦 Usando batch size conservador: {batch_size}")
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

            # Preparar respaldo antes de sobrescribir
            os.makedirs(BACKUP_DIR, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
            backup_filename = f"modelo_{model_name}.{timestamp}.h5.bak"
            backup_path = os.path.join(BACKUP_DIR, backup_filename)

            try:
                # Copia de seguridad del modelo actual
                if os.path.exists(model_path):
                    import shutil
                    shutil.copy2(model_path, backup_path)

                # Guardar nuevo modelo
                model.save(model_path)

                # Validar que el modelo guardado se puede cargar
                _ = tf.keras.models.load_model(model_path)
                print(f"Modelo {model_name} actualizado y validado.")
                # Rotar backups: mantener solo los MAX_BACKUPS más recientes por modelo
                try:
                    import glob
                    import re
                    pattern = os.path.join(BACKUP_DIR, f"modelo_{model_name}.*.h5.bak")
                    files = glob.glob(pattern)
                    # Ordenar por mtime descendente
                    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    for old in files[MAX_BACKUPS:]:
                        try:
                            os.remove(old)
                        except Exception:
                            pass
                except Exception:
                    pass

            except Exception as e:
                print(f"Error al actualizar modelo {model_name}: {e}")
                # Restaurar desde backup si existe
                try:
                    if os.path.exists(backup_path):
                        import shutil
                        shutil.copy2(backup_path, model_path)
                        print(f"Restaurado modelo {model_name} desde backup.")
                except Exception as re:
                    print(f"Error al restaurar backup de {model_name}: {re}")

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
    
    @bp.get("/check-classes")
    def check_classes(model: str = Query(..., description="Modelo a verificar: especies, hojas o plantas")):
        """Endpoint para verificar las clases disponibles sin iniciar entrenamiento"""
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
    
    @bp.post("/update-config")
    def update_config(model: str = Query(..., description="Modelo a actualizar: especies, hojas o plantas")):
        """Endpoint para actualizar la configuración con nuevas clases detectadas"""
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
    
    @bp.get("/gpu-status")
    def gpu_status():
        """Endpoint para verificar el estado y uso de las GPUs"""
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
    
    return bp
