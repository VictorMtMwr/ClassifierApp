# ClassifierApp API

API REST para clasificación de plantas usando modelos de aprendizaje profundo con TensorFlow/Keras.

## Características

- 🧠 **Tres modelos de clasificación**: Especies, Formas de hojas, y Plantas
- 🔄 **Reentrenamiento dinámico**: Reentrena modelos sin reiniciar la aplicación
- 📦 **Sistema de versionado**: Gestión automática de versiones de modelos con backups
- 🔁 **Recarga automática**: Los modelos reentrenados quedan disponibles inmediatamente
- 🖥️ **Soporte GPU/CPU**: Entrenamiento en GPU, predicciones en CPU
- ☁️ **Integración Cloudflare R2**: Descarga automática de imágenes verificadas

## Estructura del Proyecto

```
ClassifierApp/
│
├── app/
│   ├── __init__.py              # Inicializa FastAPI, configura app y modelos
│   ├── config.py                # Configuración general (paths, límites, etc.)
│   ├── models_loader.py         # Carga y gestión de los modelos TensorFlow
│   ├── preprocess.py            # Funciones de preprocesamiento de imágenes
│   ├── schemas.py               # Esquemas Pydantic para validación
│   ├── routes/
│   │   ├── predict.py           # Endpoint /predict
│   │   └── retrain.py           # Endpoint /retrain y gestión de versiones
│   └── utils/
│       ├── locks.py             # Manejadores de threading y seguridad
│       ├── label_detector.py    # Detección automática de nuevas clases
│       ├── cloudflare_downloader.py  # Descarga de imágenes desde R2
│       └── model_versioning.py  # Sistema de versionado de modelos
│
├── data/                        # Dataset de entrenamiento
│   ├── especies/train/          # Imágenes por especie y estado
│   ├── hojas/train/             # Imágenes por forma de hoja
│   └── plantas/train/           # Imágenes por tipo de planta
│
├── models/                      # Modelos actuales (.h5)
│   ├── modelo_especies.h5
│   ├── modelo_hojas.h5
│   └── modelo_plantas.h5
│
├── backups/                     # Versiones de modelos con versionado
│   ├── modelo_especies_v0001_20240101T120000.h5
│   ├── modelo_especies_v0002_20240102T150000.h5
│   ├── especies_versions.json   # Metadatos de versiones
│   └── ...
│
├── logs/                        # Logs de predicción y entrenamiento
├── main.py                      # Punto de entrada (run server)
└── requirements.txt             # Dependencias Python
```

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd ClassifierApp
```

2. Crear entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar variables de entorno (opcional):
Crear un archivo `.env` en la raíz del proyecto:
```env
# Cloudflare R2 (opcional, solo para descarga automática de imágenes)
R2_BUCKET_NAME=tu-bucket
R2_ACCOUNT_ID=tu-account-id
R2_ACCESS_KEY_ID=tu-access-key
R2_SECRET_ACCESS_KEY=tu-secret-key
R2_PREFIX=prefijo/opcional
```

5. Ejecutar la aplicación:
```bash
python main.py
```

La API estará disponible en `http://localhost:8000`

Documentación interactiva disponible en:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints de la API

### 1. Predicción

#### `POST /predict`

Clasifica una imagen usando los tres modelos.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Archivo de imagen (JPG, PNG, etc.)

**Response:**
```json
{
  "model1": {
    "class": 0,
    "class_name": "cucumis-sativus_deseased",
    "probability": 0.95,
    "all_probabilities": {
      "cucumis-sativus_deseased": 0.95,
      "cucumis-sativus_healthy": 0.05,
      ...
    }
  },
  "model2": {
    "class": 5,
    "class_name": "Palmeate",
    "probability": 0.88,
    "all_probabilities": {
      "Elliptic": 0.02,
      "Imparipinnate": 0.01,
      ...
      "Palmeate": 0.88,
      ...
    }
  },
  "model3": {
    "class": 1,
    "class_name": "True",
    "probability": 0.92,
    "all_probabilities": {
      "False": 0.08,
      "True": 0.92
    }
  }
}
```

**Ejemplo con curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@ruta/a/imagen.jpg"
```

### 2. Reentrenamiento

#### `POST /retrain?model={modelo}`

Inicia el reentrenamiento de un modelo específico.

**Parámetros:**
- `model` (query, requerido): `especies`, `hojas`, o `plantas`

**Response:**
```json
{
  "status": "Entrenamiento iniciado",
  "model": "especies",
  "classes_detected": ["cucumis-sativus_deseased", "cucumis-sativus_healthy", ...],
  "current_classes": [...],
  "new_classes": [...],
  "removed_classes": [...],
  "has_changes": true,
  "message": "Se detectaron 2 nuevas clases..."
}
```

**Notas:**
- El entrenamiento se ejecuta en un hilo separado (no bloquea la API)
- Se crea automáticamente una versión del modelo anterior antes de sobrescribir
- El modelo reentrenado se recarga automáticamente y queda disponible para predicciones
- El entrenamiento usa GPU si está disponible, CPU en caso contrario

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/retrain?model=especies"
```

### 3. Gestión de Versiones

#### `GET /retrain/versions?model={modelo}`

Lista todas las versiones disponibles de un modelo.

**Parámetros:**
- `model` (query, requerido): `especies`, `hojas`, o `plantas`

**Response:**
```json
{
  "model": "especies",
  "total_versions": 3,
  "versions": [
    {
      "version": 3,
      "timestamp": "2024-01-03T18:00:00",
      "timestamp_str": "20240103T180000",
      "filename": "modelo_especies_v0003_20240103T180000.h5",
      "path": "/path/to/backups/modelo_especies_v0003_20240103T180000.h5",
      "notes": "Reentrenamiento automático - 2024-01-03 18:00:00",
      "size_bytes": 15728640
    },
    {
      "version": 2,
      ...
    },
    {
      "version": 1,
      ...
    }
  ],
  "message": "Se encontraron 3 versiones del modelo especies"
}
```

**Ejemplo:**
```bash
curl -X GET "http://localhost:8000/retrain/versions?model=especies"
```

#### `GET /retrain/version-info?model={modelo}&version={version}`

Obtiene información detallada de una versión específica.

**Parámetros:**
- `model` (query, requerido): `especies`, `hojas`, o `plantas`
- `version` (query, requerido): Número de versión (ej: 1, 2, 3)

**Response:**
```json
{
  "model": "especies",
  "version": 2,
  "version_info": {
    "version": 2,
    "timestamp": "2024-01-02T15:00:00",
    "filename": "modelo_especies_v0002_20240102T150000.h5",
    "notes": "Reentrenamiento automático - 2024-01-02 15:00:00",
    "size_bytes": 15728640
  }
}
```

**Ejemplo:**
```bash
curl -X GET "http://localhost:8000/retrain/version-info?model=especies&version=2"
```

#### `POST /retrain/restore-version?model={modelo}&version={version}`

Restaura una versión específica de un modelo.

**Parámetros:**
- `model` (query, requerido): `especies`, `hojas`, o `plantas`
- `version` (query, requerido): Número de versión a restaurar

**Response:**
```json
{
  "status": "success",
  "message": "Modelo especies restaurado a la versión 2",
  "model_name": "especies",
  "version": 2,
  "restored_at": "2024-01-04T10:00:00",
  "reloaded": true,
  "version_info": {
    "version": 2,
    "timestamp": "2024-01-02T15:00:00",
    ...
  }
}
```

**Notas:**
- El modelo restaurado reemplaza el modelo actual en `models/`
- Se recarga automáticamente y queda disponible para predicciones
- Se crea un backup del modelo actual antes de restaurar

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/retrain/restore-version?model=especies&version=2"
```

### 4. Utilidades

#### `GET /retrain/check-classes?model={modelo}`

Verifica las clases disponibles sin iniciar entrenamiento.

#### `POST /retrain/update-config?model={modelo}`

Actualiza la configuración con nuevas clases detectadas.

#### `GET /retrain/gpu-status`

Obtiene información sobre el estado y uso de las GPUs.

## Sistema de Versionado de Modelos

### Características

- **Versionado automático**: Cada vez que se reentrena un modelo, se crea automáticamente una versión del modelo anterior
- **Numeración secuencial**: Las versiones se numeran automáticamente (v0001, v0002, v0003, ...)
- **Metadatos**: Cada versión incluye timestamp, notas y tamaño del archivo
- **Rotación automática**: Se mantienen solo las 3 versiones más recientes (configurable en `config.py`)
- **Almacenamiento**: Las versiones se guardan en `backups/` con nombres descriptivos

### Estructura de Versiones

```
backups/
├── modelo_especies_v0001_20240101T120000.h5
├── modelo_especies_v0002_20240102T150000.h5
├── modelo_especies_v0003_20240103T180000.h5
├── especies_versions.json
├── formas_versions.json
└── plantas_versions.json
```

### Metadatos de Versiones

Cada modelo tiene un archivo JSON con metadatos:
```json
{
  "versions": [
    {
      "version": 3,
      "timestamp": "2024-01-03T18:00:00",
      "timestamp_str": "20240103T180000",
      "filename": "modelo_especies_v0003_20240103T180000.h5",
      "path": "/path/to/backups/modelo_especies_v0003_20240103T180000.h5",
      "notes": "Reentrenamiento automático - 2024-01-03 18:00:00",
      "size_bytes": 15728640
    }
  ],
  "current_version": 3,
  "last_updated": "2024-01-03T18:00:00"
}
```

### Flujo de Versionado

1. **Antes de reentrenar**: Se crea una versión del modelo actual
2. **Durante el reentrenamiento**: Se entrena el nuevo modelo
3. **Después de guardar**: El nuevo modelo reemplaza al anterior
4. **Recarga automática**: El modelo nuevo se recarga y queda disponible

### Restauración de Versiones

Puedes restaurar cualquier versión disponible:
- Las versiones se listan con `GET /retrain/versions`
- Se restauran con `POST /retrain/restore-version`
- El modelo restaurado se recarga automáticamente

## Configuración

### Variables de Configuración (`app/config.py`)

- `MODEL_DIR`: Directorio donde se guardan los modelos actuales
- `BACKUP_DIR`: Directorio donde se guardan las versiones
- `DATA_DIR`: Directorio con los datasets de entrenamiento
- `MAX_BACKUPS`: Número máximo de versiones a mantener (por defecto: 3)
- `HOST`: Host del servidor (por defecto: '0.0.0.0')
- `PORT`: Puerto del servidor (por defecto: 8000)
- `DEBUG`: Modo debug (por defecto: True)

### Clases de Clasificación

Los modelos clasifican en las siguientes categorías:

**Especies** (`SPECIES`):
- cucumis-sativus_deseased/healthy
- discorea-alata-l_deseased/healthy
- manihot-esculenta_deseased/healthy
- solanum-lycopersicum_deseased/healthy
- solanum-melongena_deseased/healthy
- zea-mays_deseased/healthy

**Formas de Hojas** (`SHAPES`):
- Elliptic, Imparipinnate, Lanceolate, Obovate, Ovate, Palmeate, Trifoliate

**Plantas** (`PLANTS`):
- False, True

## Características Técnicas

### Gestión de Modelos

- **Carga inicial**: Los modelos se cargan una vez al iniciar la aplicación
- **Recarga dinámica**: Los modelos se recargan automáticamente después de reentrenar
- **Thread-safe**: Sistema de locks para evitar conflictos entre predicciones y entrenamiento
- **Memoria**: Limpieza automática de modelos antiguos al recargar

### Entrenamiento

- **GPU automática**: Detecta y usa GPU si está disponible
- **Batch size adaptativo**: Calcula automáticamente el batch size según VRAM disponible
- **Manejo de OOM**: Reduce automáticamente el batch size si hay errores de memoria
- **Validación**: 20% de los datos se usan para validación

### Predicciones

- **Ejecución en CPU**: Las predicciones se ejecutan en CPU para no interferir con el entrenamiento
- **Concurrencia**: Múltiples predicciones pueden ejecutarse simultáneamente
- **Preprocesamiento**: Las imágenes se redimensionan a 128x128 y normalizan

## Desarrollo

### Estructura de Código

- **FastAPI**: Framework web moderno y rápido
- **TensorFlow/Keras**: Para los modelos de aprendizaje profundo
- **Pydantic**: Para validación de datos
- **Threading**: Para ejecución asíncrona de entrenamientos

### Extensión

Para agregar nuevos modelos o funcionalidades:

1. Agregar el modelo en `models/`
2. Actualizar `models_loader.py` para cargarlo
3. Agregar las clases en `config.py`
4. Actualizar los endpoints según sea necesario

## Troubleshooting

### Error: "No se encontró el modelo"
- Verificar que los archivos `.h5` existan en `models/`
- Verificar permisos de lectura

### Error: "OOM (Out of Memory)"
- El sistema reduce automáticamente el batch size
- Si persiste, reducir `MAX_BACKUPS` o usar CPU

### Modelos no se actualizan después de reentrenar
- Verificar logs para errores de recarga
- Los modelos se recargan automáticamente, pero puedes reiniciar la aplicación si es necesario

### Versiones no aparecen
- Verificar que `backups/` tenga permisos de escritura
- Verificar que los archivos JSON de metadatos se creen correctamente

## Test
pytest --cov=app --cov-report=html
