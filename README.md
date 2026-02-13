# ANIF - Asistente de Investigación Económica

Sistema RAG (Retrieval-Augmented Generation) de producción con IA para análisis de documentos económicos del equipo de investigación de ANIF (Asociación Nacional de Instituciones Financieras).

## Características Principales

- **Sistema RAG de Producción**: Procesamiento inteligente con rechazo automático cuando el contexto es insuficiente
- **Ingesta Incremental**: Sistema de hash MD5 que evita reprocesamiento de archivos sin cambios
- **API REST Completa**: Endpoints para chat, ingesta de documentos, y acceso a evidencias
- **Memoria Multi-turno**: Soporte para conversaciones con sessionId (hasta 2 turnos)
- **Observabilidad Completa**: Logs estructurados JSON con tiempos divididos (embedding vs inferencia)
- **Citas Precisas**: Cada respuesta incluye fragmentos específicos con doc_id y puntuación
- **Integración de Herramientas**: Clasificación de intenciones y escalamiento automático
- **Despliegue Dockerizado**: Un solo comando para levantar toda la infraestructura
- **Suite de Pruebas**: Más de 5 pruebas de integración con reportes automáticos

## Requisitos Previos

1. **Python 3.8+** o **Docker** (para despliegue containerizado)
2. **API Key de Groq** (gratuita)
   - Regístrate en [https://console.groq.com/](https://console.groq.com/)
   - Obtén tu API key gratuita

## Instalación y Despliegue

### Opción 1: Despliegue con Docker (Recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/JulianTorrest/Economia-Colombiana.git
cd "Economia Colombiana - ANIF"

# 2. Configurar variables de entorno
copy .env.example .env
# Editar .env y agregar: GROQ_API_KEY=tu_api_key_aqui

# 3. Levantar toda la infraestructura
docker-compose up --build
```

**Acceso:**

- Frontend (Streamlit): <http://localhost:8501>
- API REST: <http://localhost:8000>
- Documentación API: <http://localhost:8000/docs>

### Opción 2: Instalación Local

```bash
# 1. Clonar y configurar entorno
git clone https://github.com/JulianTorrest/Economia-Colombiana.git
cd "Economia Colombiana - ANIF"

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Configuración de API Key

```bash
# Crear archivo .env
echo GROQ_API_KEY=tu_api_key_aqui > .env
```

**Obtén tu API key gratuita en:** https://console.groq.com/

### Ejecución Local

```bash
# Opción A: Solo Frontend
streamlit run main.py

# Opción B: Solo API
python api.py

# Opción C: Ambos servicios
# Terminal 1:
python api.py
# Terminal 2:
streamlit run main.py
```

## API REST Endpoints

### POST /documentos
Ingesta incremental de documentos con verificación de hash:
```bash
curl -X POST "http://localhost:8000/documentos" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@documento.pdf"
```

### POST /chat
Chat con memoria multi-turno y citas:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"sessionId": "session123", "mensaje": "¿Qué dice sobre el PIB?"}'
```

### GET /chunks/{doc_id}/{chunk_id}
Acceso a evidencias específicas:
```bash
curl "http://localhost:8000/chunks/documento.pdf/chunk_1"
```

### GET /health
Estado del sistema:
```bash
curl "http://localhost:8000/health"
```

## Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documentos    │    │   Procesamiento  │    │   Vectorstore   │
│   RAG (PDFs,    │───▶│   LangChain +    │───▶│   FAISS +       │
│   Excel)        │    │   Embeddings     │    │   Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Streamlit     │    │   Groq LLM       │             │
│   Interface     │◀───│   (Llama 3.1)    │◀────────────┘
└─────────────────┘    └──────────────────┘
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documentos    │    │   Hash MD5 +     │    │   FAISS         │
│   (PDF/TXT)     │───▶│   Chunking +     │───▶│   Vectorstore   │
│                 │    │   Embeddings     │    │   Local         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             │
│   Streamlit UI  │    │   FastAPI +      │             │
│   (Frontend)    │◀───│   Groq LLM       │◀────────────┘
│   Port 8501     │    │   Port 8000      │
└─────────────────┘    └──────────────────┘
                              │
                    ┌──────────────────┐
                    │   Observabilidad │
                    │   Logs JSON +    │
                    │   Métricas       │
                    └──────────────────┘
```

## Estructura del Proyecto

```
Economia Colombiana - ANIF/
├── main.py                    # Frontend Streamlit
├── api.py                     # API REST con FastAPI
├── rag_core.py               # Sistema RAG principal
├── docker-compose.yml        # Orquestación de servicios
├── requirements.txt          # Dependencias Python
├── test_*.py                 # Suite de pruebas
├── run_tests.py              # Runner de pruebas
├── test_reports.py           # Generador de reportes
├── processed_documents.json  # Registro de hashes
└── RAG/                      # Documentos económicos
    ├── *.pdf                # Reportes técnicos
    └── *.txt                # Documentos de texto
```

## Pruebas y Calidad

### Ejecutar Suite de Pruebas

```bash
# Pruebas básicas
python run_tests.py --quick

# Pruebas completas con reportes
python run_tests.py --reports

# Pruebas específicas
python run_tests.py --api-only
python run_tests.py --rag-only
```

### Reportes Generados

- **Logs detallados**: `test_results/`
- **Reportes HTML**: `test_reports/`
- **Cobertura de código**: `test_reports/coverage_html/`
- **Formato JUnit**: Para integración CI/CD

## Despliegue en Streamlit Cloud

**SÍ, `main.py` es el archivo correcto para Streamlit Cloud.**

### Configuración para Streamlit Cloud

1. **Repositorio**: <https://github.com/JulianTorrest/Economia-Colombiana>
2. **Archivo principal**: `main.py`
3. **Secretos requeridos**:
   - `GROQ_API_KEY`: Tu API key de Groq

### Pasos de Despliegue

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu repositorio de GitHub
3. Selecciona `main.py` como archivo principal
4. Agrega `GROQ_API_KEY` en la sección de secretos
5. La aplicación se desplegará automáticamente

### Características del Despliegue

- **Auto-inicialización**: El sistema RAG se inicializa automáticamente
- **Gestión de memoria**: Optimizado para Streamlit Cloud
- **Carga diferida**: Importaciones lazy para startup rápido
- **Manejo de errores**: Fallbacks para conexiones API

##  Seguridad

- **API Keys**: Nunca hardcodees API keys en el código
- **Documentos**: Los documentos se procesan localmente
- **Datos**: No se envían datos sensibles a servicios externos (excepto queries a Groq)

##  Solución de Problemas

### Error: "No module named 'X'"
```bash
pip install -r requirements.txt
```

### Error: "API key not found"
- Verifica que ingresaste correctamente tu API key de Groq
- Asegúrate de que la API key sea válida

### Error: "No documents found"
- Verifica que la carpeta RAG contenga documentos
- Asegúrate de que los archivos sean PDF o Excel válidos

### Rendimiento lento
- Reduce el número de documentos para pruebas iniciales
- Considera usar chunks más pequeños (chunk_size=500)

##  Mejoras Futuras

- [ ] Soporte para más formatos de documentos
- [ ] Análisis de gráficos y tablas
- [ ] Exportación de respuestas a PDF
- [ ] Integración con bases de datos económicas
- [ ] Análisis de sentimientos en reportes
- [ ] Dashboard con métricas económicas




##  Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

---


*Sistema RAG especializado en análisis económico colombiano*
