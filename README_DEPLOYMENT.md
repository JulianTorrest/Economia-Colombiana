# 🚀 Guía de Deployment a Streamlit Cloud

## 📋 Pasos para Deployment

### 1. Preparación del Repositorio

#### Archivos Necesarios ✅
- `main.py` - Aplicación principal
- `requirements.txt` - Dependencias
- `.gitignore` - Archivos a excluir
- `.streamlit/config.toml` - Configuración de Streamlit
- `secrets.toml.example` - Ejemplo de secretos
- Carpeta `RAG/` - Documentos para el sistema RAG

#### Archivos Excluidos (por .gitignore)
- `.env` - Variables de entorno locales
- `*.faiss` - Base de datos vectorial (se regenera automáticamente)
- `*.pkl` - Archivos de cache
- `.streamlit/secrets.toml` - Secretos locales

### 2. Push al Repositorio GitHub

```bash
# Inicializar repositorio (si no está inicializado)
git init

# Agregar remote del repositorio
git remote add origin https://github.com/JulianTorrest/Economia-Colombiana.git

# Agregar todos los archivos
git add .

# Commit inicial
git commit -m "Initial deployment: ANIF RAG System with Hybrid Search"

# Push al repositorio
git push -u origin main
```

### 3. Configuración en Streamlit Cloud

#### A. Crear Nueva App
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu cuenta de GitHub
3. Selecciona el repositorio: `JulianTorrest/Economia-Colombiana`
4. Branch: `main`
5. Main file path: `main.py`

#### B. Configurar Secretos
En Streamlit Cloud > App Settings > Secrets, agrega:

```toml
GROQ_API_KEY = "tu_api_key_real_aqui"
```

### 4. Consideraciones Importantes

#### 🔄 Sistema RAG
- Los documentos en `RAG/` se subirán al repositorio
- El sistema se inicializará automáticamente en el primer uso
- La vectorización ocurre en la nube (puede tomar 1-2 minutos la primera vez)

#### 🔑 API Key de Groq
- **NUNCA** subas tu API key al repositorio
- Configúrala solo en Streamlit Cloud Secrets
- El sistema detectará automáticamente si está en local (.env) o en la nube (secrets)

#### 📊 Funcionalidades Disponibles
- **🤖 Agente**: Chat con sistema híbrido RAG + LLM
- **📊 Generación de Informes**: Informes automáticos con múltiples modos
- **🏛️ Herramientas ANIF**: 8 herramientas especializadas de análisis económico

#### 🔍 Modos de Búsqueda
- **🔄 Híbrido**: Combina documentos ANIF + conocimiento general
- **📚 Solo RAG**: Solo documentos internos
- **🌐 Solo Conocimiento General**: Solo LLM (como Google)

### 5. Troubleshooting

#### Error de Dependencias
Si hay errores de instalación:
1. Verifica `requirements.txt`
2. Asegúrate de que todas las versiones sean compatibles
3. Revisa los logs en Streamlit Cloud

#### Error de API Key
Si no funciona la conexión a Groq:
1. Verifica que la API key esté en Secrets
2. Confirma que la key sea válida y activa
3. Revisa que tenga el formato correcto (`gsk_...`)

#### Error de Documentos RAG
Si no carga documentos:
1. Verifica que la carpeta `RAG/` tenga documentos
2. Confirma que sean archivos PDF válidos
3. Espera a que complete la inicialización

### 6. URL de la Aplicación

Una vez deployada, tu aplicación estará disponible en:
`https://economia-colombiana-[hash].streamlit.app`

### 7. Actualizaciones

Para actualizar la aplicación:
```bash
git add .
git commit -m "Update: descripción de cambios"
git push origin main
```

Streamlit Cloud se actualizará automáticamente.

## 🎯 Resultado Final

Una aplicación web completa para análisis económico de ANIF con:
- Sistema híbrido RAG + LLM
- Generación automática de informes
- Herramientas especializadas de análisis
- Interfaz intuitiva y profesional
- Acceso desde cualquier dispositivo con internet
