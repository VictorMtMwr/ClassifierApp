# Detección Automática de Etiquetas

## Descripción

El sistema ahora incluye funcionalidad para detectar automáticamente nuevas etiquetas en los datos de entrenamiento y ajustar la última capa de los modelos de manera dinámica.

## Funcionalidades Implementadas

### 1. Detección Automática de Clases
- **Función**: `detect_classes_in_data()`
- **Propósito**: Escanea el directorio de datos de entrenamiento para detectar todas las clases disponibles
- **Ubicación**: `app/utils/label_detector.py`

### 2. Comparación de Clases
- **Función**: `detect_new_classes()`
- **Propósito**: Compara las clases detectadas en los datos con las clases actuales en la configuración
- **Retorna**: Información sobre nuevas clases, clases removidas y si hay cambios

### 3. Ajuste Automático del Modelo
- **Función**: `adjust_model_for_new_classes()`
- **Propósito**: Ajusta la última capa del modelo para incluir nuevas clases
- **Características**:
  - Preserva los pesos de las clases existentes
  - Inicializa pesos aleatorios para nuevas clases
  - Usa activación apropiada según el tipo de modelo

### 4. Actualización de Configuración
- **Función**: `update_config_with_new_classes()`
- **Propósito**: Actualiza automáticamente el archivo `config.py` con las nuevas clases
- **Tipos de modelos soportados**:
  - `especies`: Lista de strings
  - `hojas`: Lista de strings
  - `plantas`: Lista de booleanos

## Endpoints Disponibles

### 1. Verificar Clases (GET)
```
GET /check-classes?model={especies|hojas|plantas}
```

**Respuesta:**
```json
{
  "model": "especies",
  "classes_detected": ["nueva_especie"],
  "current_classes": ["cassava_deseased", "cassava_healthy", ...],
  "new_classes": ["nueva_especie"],
  "removed_classes": [],
  "has_changes": true,
  "message": "Clases detectadas: 1, Clases actuales: 10"
}
```

### 2. Reentrenar con Detección Automática (POST)
```
POST /retrain?model={especies|hojas|plantas}
```

**Respuesta:**
```json
{
  "status": "Entrenamiento iniciado",
  "model": "especies",
  "classes_detected": ["nueva_especie"],
  "current_classes": ["cassava_deseased", "cassava_healthy", ...],
  "new_classes": ["nueva_especie"],
  "removed_classes": [],
  "has_changes": true,
  "message": "Se detectaron 1 nuevas clases y 0 clases removidas. El modelo será ajustado automáticamente."
}
```

## Flujo de Trabajo

1. **Detección**: El sistema escanea `data/{modelo}/train/` para detectar clases
2. **Comparación**: Compara con las clases actuales en `config.py`
3. **Actualización**: Si hay nuevas clases, actualiza la configuración
4. **Ajuste del Modelo**: Ajusta la última capa del modelo para incluir nuevas clases
5. **Entrenamiento**: Reentrena el modelo con los datos actualizados

## Estructura de Datos Esperada

```
data/
├── especies/
│   └── train/
│       ├── cassava_deseased/
│       ├── cassava_healthy/
│       └── nueva_especie/          # Nueva clase detectada automáticamente
├── hojas/
│   └── train/
│       ├── Eliptica/
│       ├── Lanceolada/
│       └── nueva_forma/            # Nueva clase detectada automáticamente
└── plantas/
    └── train/
        ├── False/
        └── True/
```

## Ventajas

- **Automático**: No requiere intervención manual para agregar nuevas clases
- **Preservación**: Mantiene el conocimiento aprendido de clases existentes
- **Flexible**: Funciona con diferentes tipos de modelos (especies, hojas, plantas)
- **Seguro**: Crea respaldos automáticos antes de modificar modelos
- **Informativo**: Proporciona información detallada sobre los cambios detectados

## Uso Recomendado

1. **Agregar nuevos datos**: Coloca las nuevas clases en el directorio `data/{modelo}/train/`
2. **Verificar cambios**: Usa `/check-classes` para ver qué se detectó
3. **Reentrenar**: Usa `/retrain` para ajustar automáticamente el modelo
4. **Verificar**: El sistema actualizará automáticamente la configuración y el modelo

## Notas Técnicas

- Los pesos de las clases existentes se preservan durante el ajuste
- Las nuevas clases se inicializan con pesos aleatorios
- El sistema usa activaciones apropiadas según el tipo de modelo
- Se mantienen respaldos automáticos de los modelos originales
- La configuración se actualiza dinámicamente sin reiniciar la aplicación
