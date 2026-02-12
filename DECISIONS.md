# Decisiones de Arquitectura y Diseño

## 1. Base de Datos Vectorial: FAISS (Local)
**Decisión:** Utilizar FAISS (Facebook AI Similarity Search) en modo local.
**Justificación:**
- **Latencia:** Al ejecutarse en memoria local, la recuperación es extremadamente rápida (<50ms).
- **Simplicidad:** Elimina la necesidad de gestionar un servicio externo (como Pinecone o Milvus) para el alcance de este MVP.
- **Costo:** Cero costo operativo.
- **Portabilidad:** El índice se guarda como un archivo `.faiss` que puede ser versionado o movido fácilmente en el contenedor Docker.

## 2. Estrategia de Fragmentación (Chunking)
**Configuración:** `RecursiveCharacterTextSplitter` con `chunk_size=1000` y `chunk_overlap=200`.
**Justificación:**
- Los documentos económicos de ANIF suelen tener párrafos densos. 1000 caracteres capturan suficiente contexto semántico para preguntas complejas.
- El solapamiento de 200 caracteres asegura que no se pierda contexto en los límites de los fragmentos, crucial para mantener la coherencia en frases cortadas.

## 3. Modelo LLM: Groq (Llama 3 70B)
**Decisión:** Usar Groq API con Llama 3.
**Justificación:**
- **Velocidad:** Groq ofrece la inferencia más rápida del mercado, esencial para una experiencia de chat fluida.
- **Capacidad de Razonamiento:** Llama 3 70B tiene un rendimiento comparable a GPT-4 en tareas de razonamiento y resumen, necesario para análisis económico.

## 4. Barreras de Seguridad (Guardrails)
**Implementación:**
1. **Umbral de Similitud:** Se descartan fragmentos con distancia L2 > 1.2 (en FAISS/HuggingFace embeddings).
2. **Rechazo Estricto:** Si el contexto recuperado es vacío o irrelevante, la API devuelve `rechazo: true` en lugar de alucinar una respuesta.
3. **Clasificación de Intención:** Se usa una herramienta interna para clasificar la consulta (Fiscal, Monetaria, etc.) y ajustar el System Prompt antes de generar la respuesta.

## 5. Ingesta Incremental (Hashing)
**Lógica:** Se calcula un hash MD5 del contenido binario de cada archivo PDF.
**Justificación:** Evita el reprocesamiento costoso (embedding) de archivos que no han cambiado, optimizando el tiempo de despliegue y recursos computacionales.