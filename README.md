# 🏛️ ANIF - Asistente de Investigación Económica

Sistema RAG (Retrieval-Augmented Generation) con IA para análisis de documentos económicos del equipo de investigación de ANIF (Asociación Nacional de Instituciones Financieras).

## 🚀 Características

- **Sistema RAG Avanzado**: Procesamiento inteligente de documentos PDF y Excel
- **LLM Gratuito**: Integración con Groq (Llama 3.1 70B)
- **Interfaz Moderna**: Aplicación web con Streamlit
- **Análisis Especializado**: Enfocado en economía colombiana
- **Búsqueda Semántica**: Embeddings multilingües para mejor comprensión
- **Chat Interactivo**: Conversación natural con el asistente

## 📋 Requisitos Previos

1. **Python 3.8+**
2. **API Key de Groq** (gratuita)
   - Regístrate en [https://console.groq.com/](https://console.groq.com/)
   - Obtén tu API key gratuita

## 🛠️ Instalación

### 1. Clonar o descargar el proyecto
```bash
# Si tienes git instalado
git clone <url-del-repositorio>
cd "Economia Colombiana - ANIF"
```

### 2. Crear entorno virtual (recomendado)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
pip install -r requirements.txt
```

### 2. Configuración de API Key
Crea un archivo `.env` con tu API key de Groq:
```bash
# Copia el template
copy .env.template .env

# Edita .env y agrega tu API key
GROQ_API_KEY=gsk_tu_api_key_aqui
```

**Obtén tu API key gratuita en:** https://console.groq.com/

### 3. Inicialización del Sistema RAG (OBLIGATORIO)
**⚠️ IMPORTANTE: Este paso debe ejecutarse ANTES del despliegue**

```bash
python setup_rag.py
```

Este script:
- ✅ Valida la conexión con Groq
- ✅ Inicializa el sistema de embeddings
- ✅ Procesa todos los documentos RAG
- ✅ Crea la base de datos vectorial
- ✅ Genera archivo de estado del sistema

### 4. Despliegue
Una vez completada la inicialización:
```bash
streamlit run main.py
```

- "¿Qué dice el último reporte sobre el PIB tendencial?"
- "¿Cuál es el análisis del presupuesto general de la nación 2026?"
- "¿Qué impacto fiscal tiene el aumento del salario mínimo 2026?"
- "¿Cuáles son las elasticidades económicas más recientes?"

## 🏗️ Arquitectura del Sistema

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

## 📁 Estructura del Proyecto

```
Economia Colombiana - ANIF/
├── main.py                 # Aplicación principal de Streamlit
├── requirements.txt        # Dependencias de Python
├── README.md              # Este archivo
└── RAG/                   # Carpeta con documentos económicos
    ├── *.pdf             # Reportes y documentos técnicos
    └── *.xlsx            # Datos económicos en Excel
```

## 🔧 Configuración Avanzada

### Variables de Entorno (Opcional)
Crea un archivo `.env` para configuraciones:
```env
GROQ_API_KEY=tu_api_key_aqui
RAG_FOLDER_PATH=./RAG
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Personalización del Modelo
En `main.py`, puedes cambiar:
- **Modelo de Groq**: Línea 186 (`model="llama-3.1-70b-versatile"`)
- **Embeddings**: Línea 77 (modelo de sentence-transformers)
- **Parámetros de chunking**: Líneas 136-140

## 🚀 Despliegue en Streamlit Cloud

### 1. Preparar el repositorio
- Sube tu código a GitHub
- Asegúrate de incluir `requirements.txt`

### 2. Conectar con Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu repositorio de GitHub
3. Selecciona `main.py` como archivo principal

### 3. Configurar secretos
En Streamlit Cloud, agrega:
- `GROQ_API_KEY`: Tu API key de Groq

### 4. Desplegar
- La aplicación se desplegará automáticamente
- Comparte la URL con tu equipo

## 🔒 Seguridad

- **API Keys**: Nunca hardcodees API keys en el código
- **Documentos**: Los documentos se procesan localmente
- **Datos**: No se envían datos sensibles a servicios externos (excepto queries a Groq)

## 🐛 Solución de Problemas

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

## 📈 Mejoras Futuras

- [ ] Soporte para más formatos de documentos
- [ ] Análisis de gráficos y tablas
- [ ] Exportación de respuestas a PDF
- [ ] Integración con bases de datos económicas
- [ ] Análisis de sentimientos en reportes
- [ ] Dashboard con métricas económicas

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crea un Pull Request

## 📞 Soporte

Para soporte técnico o preguntas:
- Crea un issue en GitHub
- Contacta al equipo de desarrollo de ANIF

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

---

**Desarrollado para ANIF - Asociación Nacional de Instituciones Financieras**

*Sistema RAG especializado en análisis económico colombiano*
