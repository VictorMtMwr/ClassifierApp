from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ModelPrediction(BaseModel):
    """Schema para la predicción de un modelo individual"""
    class_: int = Field(..., alias="class", description="Índice de la clase predicha")
    class_name: str = Field(..., description="Nombre de la clase predicha")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidad de la predicción (0-1)")

    class Config:
        populate_by_name = True


class PredictResponse(BaseModel):
    """Schema para la respuesta del endpoint de predicción"""
    model1: ModelPrediction = Field(..., description="Predicción del modelo de especies")
    model2: ModelPrediction = Field(..., description="Predicción del modelo de hojas/formas")
    model3: ModelPrediction = Field(..., description="Predicción del modelo de plantas")


class RetrainResponse(BaseModel):
    """Schema para la respuesta del endpoint de reentrenamiento"""
    status: str = Field(..., description="Estado del entrenamiento")
    model: str = Field(..., description="Nombre del modelo reentrenado")
    classes_detected: List[str] = Field(..., description="Lista de clases detectadas en los datos")
    current_classes: List[str] = Field(..., description="Lista de clases actuales en el modelo")
    new_classes: List[str] = Field(..., description="Lista de nuevas clases detectadas")
    removed_classes: List[str] = Field(..., description="Lista de clases removidas")
    has_changes: bool = Field(..., description="Indica si hay cambios en las clases")
    message: str = Field(..., description="Mensaje descriptivo sobre el estado")


class CheckClassesResponse(BaseModel):
    """Schema para la respuesta del endpoint de verificación de clases"""
    model: str = Field(..., description="Nombre del modelo verificado")
    classes_detected: List[str] = Field(..., description="Lista de clases detectadas en los datos")
    current_classes: List[str] = Field(..., description="Lista de clases actuales en el modelo")
    new_classes: List[str] = Field(..., description="Lista de nuevas clases detectadas")
    removed_classes: List[str] = Field(..., description="Lista de clases removidas")
    has_changes: bool = Field(..., description="Indica si hay cambios en las clases")
    message: str = Field(..., description="Mensaje descriptivo sobre el estado")


class UpdateConfigResponse(BaseModel):
    """Schema para la respuesta del endpoint de actualización de configuración"""
    status: str = Field(..., description="Estado de la operación (success/info)")
    model: Optional[str] = Field(None, description="Nombre del modelo actualizado")
    new_classes_added: Optional[List[str]] = Field(None, description="Lista de nuevas clases agregadas")
    message: str = Field(..., description="Mensaje descriptivo sobre el resultado")


class GPUMemoryUsage(BaseModel):
    """Schema para información de uso de memoria de GPU"""
    current_gb: Optional[float] = Field(None, description="Memoria actual usada en GB")
    peak_gb: Optional[float] = Field(None, description="Memoria pico usada en GB")


class GPUDetails(BaseModel):
    """Schema para detalles de una GPU individual"""
    index: int = Field(..., description="Índice de la GPU")
    name: str = Field(..., description="Nombre de la GPU")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detalles adicionales de la GPU")
    memory_usage: Optional[GPUMemoryUsage] = Field(None, description="Información de uso de memoria")
    error: Optional[str] = Field(None, description="Mensaje de error si hubo problemas")


class TensorFlowGPUInfo(BaseModel):
    """Schema para información de GPU desde TensorFlow"""
    gpus_available: int = Field(..., description="Número de GPUs disponibles")
    gpus: List[GPUDetails] = Field(..., description="Lista de detalles de cada GPU")


class NvidiaSMIInfo(BaseModel):
    """Schema para información de GPU desde nvidia-smi"""
    index: str = Field(..., description="Índice de la GPU")
    name: str = Field(..., description="Nombre de la GPU")
    gpu_utilization_percent: str = Field(..., description="Porcentaje de utilización de GPU")
    memory_utilization_percent: str = Field(..., description="Porcentaje de utilización de memoria")
    memory_used_mb: str = Field(..., description="Memoria usada en MB")
    memory_total_mb: str = Field(..., description="Memoria total en MB")
    temperature_celsius: str = Field(..., description="Temperatura en grados Celsius")


class GPUStatusResponse(BaseModel):
    """Schema para la respuesta del endpoint de estado de GPU"""
    tensorflow_gpu_info: TensorFlowGPUInfo = Field(..., description="Información de GPU desde TensorFlow")
    nvidia_smi_info: Optional[List[NvidiaSMIInfo]] = Field(None, description="Información de GPU desde nvidia-smi")
    message: str = Field(..., description="Mensaje informativo sobre el monitoreo")


class ErrorResponse(BaseModel):
    """Schema para respuestas de error"""
    detail: str = Field(..., description="Mensaje de error detallado")

