import os
import json
import importlib
import sys
import re
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
    Usa las variables estáticas que se actualizan dinámicamente mediante reload_config().
    
    Args:
        model_type (str): Tipo de modelo ('especies', 'hojas', 'plantas')
        
    Returns:
        list: Lista de clases actuales
    """
    from ..config import SPECIES, SHAPES, PLANTS
    
    if model_type == 'especies':
        return SPECIES.copy() if hasattr(SPECIES, 'copy') else list(SPECIES)  # Variable estática (lista)
    elif model_type == 'hojas':
        return SHAPES.copy() if hasattr(SHAPES, 'copy') else list(SHAPES)  # Variable estática (lista)
    elif model_type == 'plantas':
        plants = PLANTS.copy() if hasattr(PLANTS, 'copy') else list(PLANTS)  # Variable estática (lista)
        return [str(x) for x in plants]  # Convertir booleanos a strings
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


def update_config_with_detected_classes(model_type, detected_classes):
    """
    Actualiza la configuración con todas las clases detectadas en orden alfabético.
    Esto asegura que el orden en config.py coincida con el orden que usa ImageDataGenerator.
    
    Args:
        model_type (str): Tipo de modelo ('especies', 'hojas', 'plantas')
        detected_classes (list): Lista de todas las clases detectadas (ya ordenadas alfabéticamente)
    
    Returns:
        bool: True si la actualización fue exitosa, False en caso contrario
    """
    if not detected_classes:
        return False
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')
    
    # Leer el archivo de configuración actual
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Usar las clases detectadas directamente (ya están en orden alfabético)
    updated_classes = sorted(detected_classes)  # Asegurar orden alfabético
    
    # Determinar el nombre de la variable según el tipo de modelo
    # IMPORTANTE: Actualizar _DEFAULT_* porque esas son las variables que contienen las listas
    if model_type == 'especies':
        default_var_name = '_DEFAULT_SPECIES'
        var_name = 'SPECIES'
        # Formatear como lista de strings con saltos de línea para legibilidad
        formatted_classes = []
        for cls in updated_classes:
            formatted_classes.append(f"    '{cls}'")
        new_content = "[\n" + ",\n".join(formatted_classes) + "\n]"
        
    elif model_type == 'hojas':
        default_var_name = '_DEFAULT_SHAPES'
        var_name = 'SHAPES'
        # Formatear como lista de strings con saltos de línea para legibilidad
        formatted_classes = []
        for cls in updated_classes:
            formatted_classes.append(f"    '{cls}'")
        new_content = "[\n" + ",\n".join(formatted_classes) + "\n]"
        
    elif model_type == 'plantas':
        default_var_name = '_DEFAULT_PLANTS'
        var_name = 'PLANTS'
        # Convertir strings de vuelta a booleanos
        bool_classes = []
        for cls in updated_classes:
            if isinstance(cls, bool):
                bool_classes.append(cls)
            elif cls.lower() in ['true', '1', 'yes']:
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
    
    # Método mejorado: buscar línea por línea para manejar listas multilínea correctamente
    # Buscar _DEFAULT_* que es donde realmente está la lista
    lines = content.split('\n')
    new_lines = []
    i = 0
    found_default = False
    
    while i < len(lines):
        line = lines[i]
        # Buscar la línea que define _DEFAULT_* (donde está la lista real)
        if re.match(rf"^{default_var_name}\s*=\s*\[", line):
            found_default = True
            # Agregar la nueva definición
            new_lines.append(f"{default_var_name} = {new_content}")
            # Saltar todas las líneas hasta encontrar el corchete de cierre
            bracket_count = line.count('[') - line.count(']')
            i += 1
            while i < len(lines) and bracket_count > 0:
                bracket_count += lines[i].count('[') - lines[i].count(']')
                i += 1
            # No agregar la línea actual (ya la saltamos)
            continue
        new_lines.append(line)
        i += 1
    
    if not found_default:
        print(f"Advertencia: No se pudo encontrar la definición de {default_var_name} en config.py")
        return False
    
    # También actualizar SPECIES/SHAPES/PLANTS para que apunten a la nueva lista
    # Buscar la línea SPECIES = _DEFAULT_SPECIES.copy() y actualizarla (si existe)
    lines = new_lines  # Continuar con las líneas ya procesadas
    new_lines = []
    i = 0
    found_var = False
    
    while i < len(lines):
        line = lines[i]
        # Buscar la línea que define SPECIES/SHAPES/PLANTS
        # El patrón busca: SPECIES = _DEFAULT_SPECIES.copy()
        pattern = rf"^{var_name}\s*=\s*{default_var_name}\.copy\(\)"
        if re.match(pattern, line):
            found_var = True
            # Mantener la línea tal cual (ya apunta a _DEFAULT_*)
            new_lines.append(line)
            i += 1
            continue
        new_lines.append(line)
        i += 1
    
    # Si no se encuentra la línea SPECIES = _DEFAULT_SPECIES.copy(), no es crítico
    # porque se recargará cuando se recargue el módulo
    
    new_file_content = '\n'.join(new_lines)
    
    # Escribir el archivo actualizado
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_file_content)
    
    print(f"✅ Configuración actualizada: {var_name} ahora tiene {len(updated_classes)} clases en orden alfabético")
    return True


def update_config_with_new_classes(model_type, new_classes):
    """
    Actualiza la configuración con las nuevas clases detectadas.
    DEPRECATED: Usar update_config_with_detected_classes en su lugar.
    
    Args:
        model_type (str): Tipo de modelo ('especies', 'hojas', 'plantas')
        new_classes (list): Lista de nuevas clases a agregar
    """
    if not new_classes:
        return True
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')
    
    # Leer el archivo de configuración actual
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Obtener las clases actuales
    current_classes = get_current_classes(model_type)
    
    # Crear la nueva lista de clases (evitar duplicados)
    updated_classes = list(current_classes)
    for new_class in new_classes:
        if new_class not in updated_classes:
            updated_classes.append(new_class)
    
    # Ordenar alfabéticamente para mantener consistencia
    updated_classes = sorted(updated_classes)
    
    # Determinar el nombre de la variable según el tipo de modelo
    if model_type == 'especies':
        var_name = 'SPECIES'
        # Formatear como lista de strings con saltos de línea para legibilidad
        formatted_classes = []
        for i, cls in enumerate(updated_classes):
            if i == 0:
                formatted_classes.append(f"    '{cls}'")
            else:
                formatted_classes.append(f"    '{cls}'")
        new_content = "[\n" + ",\n".join(formatted_classes) + "\n]"
        
    elif model_type == 'hojas':
        var_name = 'SHAPES'
        # Formatear como lista de strings con saltos de línea para legibilidad
        formatted_classes = []
        for i, cls in enumerate(updated_classes):
            if i == 0:
                formatted_classes.append(f"    '{cls}'")
            else:
                formatted_classes.append(f"    '{cls}'")
        new_content = "[\n" + ",\n".join(formatted_classes) + "\n]"
        
    elif model_type == 'plantas':
        var_name = 'PLANTS'
        # Convertir strings de vuelta a booleanos
        bool_classes = []
        for cls in updated_classes:
            if isinstance(cls, bool):
                bool_classes.append(cls)
            elif cls.lower() in ['true', '1', 'yes']:
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
    
    # Patrón más específico para capturar la variable completa
    if model_type in ['especies', 'hojas']:
        pattern = rf"^{var_name}\s*=\s*\[.*?\]"
        replacement = f"{var_name} = {new_content}"
    else:  # plantas
        pattern = rf"^{var_name}\s*=\s*\[.*?\]"
        replacement = f"{var_name} = {new_content}"
    
    new_file_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    # Verificar que el reemplazo fue exitoso
    if new_file_content == content:
        print(f"Advertencia: No se pudo actualizar {var_name} en config.py")
        return False
    
    # Escribir el archivo actualizado
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_file_content)
    
    print(f"Configuración actualizada: {var_name} ahora incluye {len(updated_classes)} clases")
    return True


def reload_config():
    """
    Recarga el módulo de configuración para aplicar los cambios sin reiniciar la aplicación.
    También actualiza las variables estáticas SPECIES, SHAPES, PLANTS desde el archivo.
    """
    try:
        # Recargar el módulo de configuración
        if 'app.config' in sys.modules:
            importlib.reload(sys.modules['app.config'])
            
            # Leer las clases actualizadas directamente desde el archivo config.py
            # para asegurar que las variables estáticas se actualicen correctamente
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_lines = f.readlines()
                
                import ast
                
                # Función auxiliar para extraer lista desde líneas
                def extract_list_from_lines(lines, var_name):
                    """Extrae una lista de una variable en config.py manejando listas multilínea"""
                    result_lines = []
                    in_var = False
                    bracket_count = 0
                    for i, line in enumerate(lines):
                        if re.match(rf"^{var_name}\s*=\s*\[", line):
                            in_var = True
                            result_lines.append(line)
                            bracket_count = line.count('[') - line.count(']')
                            continue
                        if in_var:
                            result_lines.append(line)
                            bracket_count += line.count('[') - line.count(']')
                            if bracket_count == 0:
                                break
                    if result_lines:
                        var_content = ''.join(result_lines)
                        # Extraer solo la parte de la lista
                        match = re.search(r'\[(.*)\]', var_content, re.DOTALL)
                        if match:
                            try:
                                return ast.literal_eval('[' + match.group(1) + ']')
                            except:
                                pass
                    return None
                
                # Actualizar _DEFAULT_SPECIES y SPECIES
                species_list = extract_list_from_lines(config_lines, '_DEFAULT_SPECIES')
                if species_list is not None:
                    sys.modules['app.config']._DEFAULT_SPECIES = species_list
                    sys.modules['app.config'].SPECIES = species_list.copy()
                
                # Actualizar _DEFAULT_SHAPES y SHAPES
                shapes_list = extract_list_from_lines(config_lines, '_DEFAULT_SHAPES')
                if shapes_list is not None:
                    sys.modules['app.config']._DEFAULT_SHAPES = shapes_list
                    sys.modules['app.config'].SHAPES = shapes_list.copy()
                
                # Actualizar _DEFAULT_PLANTS y PLANTS
                plants_list = extract_list_from_lines(config_lines, '_DEFAULT_PLANTS')
                if plants_list is not None:
                    sys.modules['app.config']._DEFAULT_PLANTS = plants_list
                    sys.modules['app.config'].PLANTS = plants_list.copy()
            
            print("✅ Configuración recargada exitosamente")
            return True
    except Exception as e:
        print(f"⚠️  Error al recargar configuración: {e}")
        import traceback
        traceback.print_exc()
        return False


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
    import tensorflow as tf
    
    if not new_classes:
        return model
    
    # Obtener el número actual de clases del modelo (no de la configuración)
    # Manejar casos donde output puede ser una lista o un tensor
    if isinstance(model.output, list):
        old_num_classes = model.output[0].shape[-1]
    else:
        old_num_classes = model.output.shape[-1]
    new_num_classes = old_num_classes + len(new_classes)
    
    print(f"Ajustando modelo: {old_num_classes} -> {new_num_classes} clases")
    
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
        # Copiar pesos de las clases originales
        new_weights[0][:, :old_num_classes] = old_weights[0]
        new_weights[1][:old_num_classes] = old_weights[1]
        # Las nuevas clases tendrán pesos inicializados aleatoriamente
        new_model.layers[-1].set_weights(new_weights)
    
    return new_model
