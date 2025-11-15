# Tests

Este directorio contiene las pruebas unitarias y de integración para ClassifierApp.

## Estructura

```
tests/
├── __init__.py
├── conftest.py              # Fixtures y configuración compartida
├── test_preprocess.py       # Pruebas de preprocesamiento de imágenes
├── test_model_versioning.py # Pruebas del sistema de versionado
├── test_api_predict.py      # Pruebas del endpoint de predicción
├── test_api_retrain.py      # Pruebas de endpoints de reentrenamiento
└── test_models_loader.py    # Pruebas de carga de modelos
```

## Ejecutar las pruebas

### Todas las pruebas
```bash
pytest
```

### Con cobertura
```bash
pytest --cov=app --cov-report=html
```

### Pruebas específicas
```bash
pytest tests/test_preprocess.py
pytest tests/test_api_predict.py::test_predict_endpoint_success
```

### Por marcadores
```bash
pytest -m unit          # Solo pruebas unitarias
pytest -m api           # Solo pruebas de API
pytest -m "not slow"    # Excluir pruebas lentas
```

### Verbose
```bash
pytest -v
```

## Fixtures disponibles

- `temp_dir`: Directorio temporal para pruebas
- `mock_model`: Modelo TensorFlow mock
- `mock_models`: Tres modelos mock
- `sample_image_bytes`: Bytes de imagen de prueba
- `mock_image_file`: Archivo de imagen para FastAPI
- `app_with_mocked_models`: Instancia de app con modelos mockeados
- `client`: Cliente de prueba para la API
- `mock_backup_dir`: Directorio de backups temporal
- `mock_model_dir`: Directorio de modelos temporal
- `sample_model_file`: Archivo de modelo de prueba
- `mock_config`: Configuración mock con directorios temporales

## Escribir nuevas pruebas

1. Crear un archivo `test_*.py` en este directorio
2. Importar pytest y los fixtures necesarios
3. Usar los fixtures disponibles o crear nuevos en `conftest.py`
4. Seguir el patrón de nomenclatura: `test_*` para funciones de prueba

Ejemplo:
```python
def test_my_function(client):
    response = client.get("/my-endpoint")
    assert response.status_code == 200
```

## Notas

- Las pruebas que requieren modelos reales deben marcarse con `@pytest.mark.requires_model`
- Las pruebas lentas deben marcarse con `@pytest.mark.slow`
- Usar mocks para modelos de TensorFlow para evitar cargar modelos reales en pruebas

