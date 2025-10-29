from flask import Blueprint, jsonify, request
import threading
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ..config import MODEL_DIR, DATA_DIR, BACKUP_DIR, MAX_BACKUPS
from ..utils.label_detector import detect_new_classes, update_config_with_new_classes, adjust_model_for_new_classes, reload_config

bp = Blueprint('retrain', __name__)

def init_retrain_route():
    @bp.route('/retrain', methods=['POST'])
    def retrain_model():
        model_name = request.args.get('model')
        if model_name not in ['especies', 'hojas', 'plantas']:
            return jsonify({
                "error": "Debes especificar ?model=especies | hojas | plantas"
            }), 400

        def train_thread(model_name):
            # Habilitar memory growth para todas las GPUs disponibles
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            print(f"Reentrenando modelo {model_name}...")
            model_path = os.path.join(MODEL_DIR, f"modelo_{model_name}.h5")
            data_path = os.path.join(DATA_DIR, model_name)

            # Detectar nuevas clases en los datos
            print(f"Detectando clases en los datos para {model_name}...")
            class_info = detect_new_classes(model_name)
            
            if class_info['has_changes']:
                print(f"Nuevas clases detectadas: {class_info['new_classes']}")
                if class_info['removed_classes']:
                    print(f"Clases removidas: {class_info['removed_classes']}")
                
                # Actualizar configuración con nuevas clases
                if class_info['new_classes']:
                    print(f"Actualizando configuración con nuevas clases...")
                    if update_config_with_new_classes(model_name, class_info['new_classes']):
                        # Recargar la configuración para aplicar los cambios
                        reload_config()
                        print("Configuración actualizada y recargada exitosamente.")
                    else:
                        print("Error al actualizar la configuración.")
            else:
                print(f"No se detectaron cambios en las clases para {model_name}")

            # Cargar y entrenar en GPU si hay, de lo contrario en CPU
            with tf.device('/GPU:0') if tf.config.list_physical_devices('GPU') else tf.device('/CPU:0'):
                model = tf.keras.models.load_model(model_path)
                
                # Ajustar modelo si hay nuevas clases
                if class_info['has_changes'] and class_info['new_classes']:
                    print(f"Ajustando modelo para {len(class_info['new_classes'])} nuevas clases...")
                    model = adjust_model_for_new_classes(model, model_name, class_info['new_classes'])
                    print("Modelo ajustado exitosamente.")

            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_gen = datagen.flow_from_directory(
                os.path.join(data_path, "train"),
                target_size=(128, 128),
                batch_size=32,
                subset='training'
            )
            val_gen = datagen.flow_from_directory(
                os.path.join(data_path, "train"),
                target_size=(128, 128),
                batch_size=32,
                subset='validation'
            )

            model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            model.fit(train_gen, epochs=5, validation_data=val_gen)

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
        class_info = detect_new_classes(model_name)
        
        threading.Thread(target=train_thread, args=(model_name,)).start()

        response = {
            "status": "Entrenamiento iniciado",
            "model": model_name,
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

        return jsonify(response)
    
    @bp.route('/check-classes', methods=['GET'])
    def check_classes():
        """Endpoint para verificar las clases disponibles sin iniciar entrenamiento"""
        model_name = request.args.get('model')
        if model_name not in ['especies', 'hojas', 'plantas']:
            return jsonify({
                "error": "Debes especificar ?model=especies | hojas | plantas"
            }), 400
        
        class_info = detect_new_classes(model_name)
        
        return jsonify({
            "model": model_name,
            "classes_detected": class_info['detected_classes'],
            "current_classes": class_info['current_classes'],
            "new_classes": class_info['new_classes'],
            "removed_classes": class_info['removed_classes'],
            "has_changes": class_info['has_changes'],
            "message": f"Clases detectadas: {len(class_info['detected_classes'])}, Clases actuales: {len(class_info['current_classes'])}"
        })
    
    @bp.route('/update-config', methods=['POST'])
    def update_config():
        """Endpoint para actualizar la configuración con nuevas clases detectadas"""
        model_name = request.args.get('model')
        if model_name not in ['especies', 'hojas', 'plantas']:
            return jsonify({
                "error": "Debes especificar ?model=especies | hojas | plantas"
            }), 400
        
        class_info = detect_new_classes(model_name)
        
        if class_info['new_classes']:
            if update_config_with_new_classes(model_name, class_info['new_classes']):
                reload_config()
                return jsonify({
                    "status": "success",
                    "model": model_name,
                    "new_classes_added": class_info['new_classes'],
                    "message": f"Configuración actualizada con {len(class_info['new_classes'])} nuevas clases"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Error al actualizar la configuración"
                }), 500
        else:
            return jsonify({
                "status": "info",
                "message": "No hay nuevas clases para actualizar"
            })
    
    return bp
