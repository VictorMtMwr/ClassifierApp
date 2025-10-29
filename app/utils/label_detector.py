import os
import json
from ..config import DATA_DIR, MODEL_DIR


def detect_classes_in_data(data_path):
    """
    Detecta las clases disponibles en los datos de entrenamiento.
    
    Args:
        data_path (str): Ruta al directorio de datos de entrenamiento
        
    Returns:
        list: Lista de clases encontradas ordenadas alfabéticamente
    """
    train_path = os.path.join(data_path, "train")
    if not os.path.exists(train_path):
        return []
    
    # Obtener todas las carpetas (clases) en el directorio de entrenamiento
    classes = []
    for item in os.listdir(train_path):
        item_path = os.path.join(train_path, item)
        if os.path.isdir(item_path):
            classes.append(item)
    
    return sorted(classes)


def get_current_classes(model_type):
    """
    Obtiene las clases actuales según el tipo de modelo.
    
    Args:
        model_type (str): Tipo de modelo ('especies', 'hojas', 'plantas')
        
    Returns:
        list: Lista de clases actuales
    """
    from ..config import SPECIES, SHAPES, PLANTS
    
    if model_type == 'especies':
        return SPECIES
    elif model_type == 'hojas':
        return SHAPES
    elif model_type == 'plantas':
        return [str(x) for x in PLANTS]  # Convertir booleanos a strings
    else:
        return []


def detect_new_classes(model_type):
    """
    Detecta si hay nuevas clases en los datos comparado con la configuración actual.
    
    Args:
        model_type (str): Tipo de modelo ('especies', 'hojas', 'plantas')
        
    Returns:
        dict: Información sobre las clases detectadas
    """
    data_path = os.path.join(DATA_DIR, model_type)
    detected_classes = detect_classes_in_data(data_path)
    current_classes = get_current_classes(model_type)
    
    new_classes = [cls for cls in detected_classes if cls not in current_classes]
    removed_classes = [cls for cls in current_classes if cls not in detected_classes]
    
    return {
        'detected_classes': detected_classes,
        'current_classes': current_classes,
        'new_classes': new_classes,
        'removed_classes': removed_classes,
        'has_changes': len(new_classes) > 0 or len(removed_classes) > 0
    }


def update_config_with_new_classes(model_type, new_classes):
    """
    Actualiza la configuración con las nuevas clases detectadas.
    
    Args:
        model_type (str): Tipo de modelo ('especies', 'hojas', 'plantas')
        new_classes (list): Lista de nuevas clases a agregar
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')
    
    # Leer el archivo de configuración actual
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Obtener las clases actuales
    current_classes = get_current_classes(model_type)
    
    # Crear la nueva lista de clases
    updated_classes = current_classes + new_classes
    
    # Determinar el nombre de la variable según el tipo de modelo
    if model_type == 'especies':
        var_name = 'SPECIES'
        # Formatear como lista de strings
        new_content = str(updated_classes)
    elif model_type == 'hojas':
        var_name = 'SHAPES'
        # Formatear como lista de strings
        new_content = str(updated_classes)
    elif model_type == 'plantas':
        var_name = 'PLANTS'
        # Convertir strings de vuelta a booleanos
        bool_classes = []
        for cls in updated_classes:
            if cls.lower() in ['true', '1', 'yes']:
                bool_classes.append(True)
            elif cls.lower() in ['false', '0', 'no']:
                bool_classes.append(False)
            else:
                # Si no es un booleano, mantener como string
                bool_classes.append(cls)
        new_content = str(bool_classes)
    else:
        return False
    
    # Reemplazar la línea correspondiente en el archivo de configuración
    import re
    pattern = rf"^{var_name}\s*=\s*\[.*?\]"
    replacement = f"{var_name} = {new_content}"
    
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Escribir el archivo actualizado
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True


def adjust_model_for_new_classes(model, model_type, new_classes):
    """
    Ajusta la última capa del modelo para incluir las nuevas clases.
    
    Args:
        model: Modelo de TensorFlow/Keras
        model_type (str): Tipo de modelo ('especies', 'hojas', 'plantas')
        new_classes (list): Lista de nuevas clases a agregar
        
    Returns:
        model: Modelo ajustado
    """
    if not new_classes:
        return model
    
    # Obtener el número actual de clases
    current_classes = get_current_classes(model_type)
    old_num_classes = len(current_classes)
    new_num_classes = old_num_classes + len(new_classes)
    
    # Obtener la última capa del modelo
    last_layer = model.layers[-1]
    
    # Crear una nueva capa de salida con el número correcto de clases
    if model_type == 'plantas':
        # Para plantas, usar sigmoid ya que es binario
        new_output = tf.keras.layers.Dense(
            new_num_classes, 
            activation='sigmoid',
            name='new_output'
        )(model.layers[-2].output)
    else:
        # Para especies y hojas, usar softmax
        new_output = tf.keras.layers.Dense(
            new_num_classes, 
            activation='softmax',
            name='new_output'
        )(model.layers[-2].output)
    
    # Crear un nuevo modelo con la nueva capa de salida
    new_model = tf.keras.Model(inputs=model.input, outputs=new_output)
    
    # Copiar los pesos de las clases existentes
    old_weights = last_layer.get_weights()
    if old_weights:
        new_weights = new_model.layers[-1].get_weights()
        new_weights[0][:old_num_classes] = old_weights[0]
        new_weights[1][:old_num_classes] = old_weights[1]
        new_model.layers[-1].set_weights(new_weights)
    
    return new_model
