# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para compilación
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Crear carpeta para logs y uploads
RUN mkdir -p logs uploads RAG vectorstore

# Script de entrada para decidir qué correr
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]