# Explicación de Warnings en los Tests

Este documento explica los warnings que aparecen al ejecutar las pruebas.

## 1. Warning de Keras: `input_shape` en Dense

```
UserWarning: Do not pass an 'input_shape'/'input_dim' argument to a layer.
```

**Causa**: En los tests, creamos modelos de prueba usando:
```python
tf.keras.layers.Dense(10, activation='softmax', input_shape=(128, 128, 3))
```

**Explicación**: Keras recomienda usar `Input(shape=...)` como primera capa en lugar de pasar `input_shape` directamente a `Dense`.

**Impacto**: ⚠️ Bajo - Es solo un warning, no afecta la funcionalidad. Los modelos funcionan correctamente.

**Solución** (opcional):
```python
# En lugar de:
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='softmax', input_shape=(128, 128, 3))
])

# Usar:
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## 2. Warning de NumPy 2.0

```
DeprecationWarning: __array__ implementation doesn't accept a copy keyword
```

**Causa**: TensorFlow/Keras está usando una API de NumPy que cambió en NumPy 2.0.

**Explicación**: Es un problema de compatibilidad entre TensorFlow y NumPy 2.0. TensorFlow aún no está completamente actualizado para la nueva API.

**Impacto**: ⚠️ Muy bajo - Es un warning de deprecación, no afecta la funcionalidad actual.

**Solución**: Esperar a que TensorFlow actualice su código para ser compatible con NumPy 2.0, o usar NumPy < 2.0 (aunque esto puede no ser posible con Python 3.13).

## 3. Warning de datetime.utcnow()

```
DeprecationWarning: datetime.datetime.utcnow() is deprecated
```

**Causa**: En `app/utils/model_versioning.py` usamos `datetime.utcnow()` que está deprecado en Python 3.12+.

**Explicación**: Python recomienda usar `datetime.now(datetime.UTC)` en lugar de `datetime.utcnow()`.

**Impacto**: ⚠️ Medio - Funciona ahora pero será removido en futuras versiones de Python.

**Solución**: Actualizar el código para usar:
```python
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc)
```

## 4. Warning de formato HDF5

```
You are saving your model as an HDF5 file via model.save()
```

**Causa**: Guardamos modelos con extensión `.h5` (formato HDF5), que Keras considera legacy.

**Explicación**: Keras recomienda usar el formato nativo `.keras` en lugar de `.h5`.

**Impacto**: ⚠️ Bajo - El formato HDF5 sigue funcionando, pero `.keras` es el formato recomendado.

**Solución** (opcional): Cambiar extensiones de `.h5` a `.keras`:
```python
# En lugar de:
model.save('modelo.h5')

# Usar:
model.save('modelo.keras')
```

**Nota**: Esto requeriría actualizar todo el código que carga modelos y puede romper compatibilidad con modelos existentes.

## Resumen

| Warning | Severidad | Acción Requerida |
|---------|-----------|------------------|
| Keras input_shape | Baja | Opcional - Mejorar código de tests |
| NumPy 2.0 | Muy Baja | Ninguna - Esperar actualización de TensorFlow |
| datetime.utcnow() | Media | Recomendado - Actualizar código |
| Formato HDF5 | Baja | Opcional - Considerar migración a .keras |

## Cómo suprimir warnings en los tests

Si quieres suprimir estos warnings en los tests, puedes agregar esto a `pytest.ini`:

```ini
[pytest]
filterwarnings =
    ignore::UserWarning:keras.*
    ignore::DeprecationWarning:datetime
    ignore::DeprecationWarning:numpy
```

O en el código de tests:
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

