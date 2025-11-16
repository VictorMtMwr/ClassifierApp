# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala solo dependencias necesarias para Pillow y TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt ./

# Instala TensorFlow y el resto de dependencias
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir tensorflow==2.19.0 \
    && pip install --no-cache-dir -r requirements.txt

# Copia el código fuente y pesos del modelo
COPY main.py ./
COPY app ./app
COPY models ./models
COPY backups ./backups

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
