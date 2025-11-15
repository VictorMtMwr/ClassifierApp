"""
Pruebas unitarias para el módulo de preprocesamiento de imágenes
"""
import pytest
import numpy as np
from PIL import Image
import io
from app.preprocess import preprocess_image


def test_preprocess_image_valid_png():
    """Test que preprocess_image procesa correctamente una imagen PNG válida"""
    # Crear una imagen PNG de prueba
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    result = preprocess_image(img_bytes.getvalue())
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 128, 128, 3)  # Batch, height, width, channels


def test_preprocess_image_valid_jpg():
    """Test que preprocess_image procesa correctamente una imagen JPG válida"""
    # Crear una imagen JPG de prueba
    img = Image.new('RGB', (200, 200), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    result = preprocess_image(img_bytes.getvalue())
    
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 128, 128, 3)


def test_preprocess_image_resize():
    """Test que las imágenes se redimensionan correctamente a 128x128"""
    # Crear una imagen de tamaño diferente
    img = Image.new('RGB', (50, 50), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    result = preprocess_image(img_bytes.getvalue())
    
    assert result.shape[1] == 128  # height
    assert result.shape[2] == 128  # width


def test_preprocess_image_normalization():
    """Test que los valores están normalizados entre 0 y 1"""
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    result = preprocess_image(img_bytes.getvalue())
    
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_preprocess_image_invalid_bytes():
    """Test que preprocess_image maneja correctamente bytes inválidos"""
    invalid_bytes = b'not an image'
    
    with pytest.raises(Exception):  # Puede ser ValueError, IOError, etc.
        preprocess_image(invalid_bytes)

