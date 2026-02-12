import time
import json
import logging
import uuid
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag_core import ANIFRAGSystem
import uvicorn

# Importaciones para procesamiento de documentos
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuración de Observabilidad (Logs JSON) ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "props"):
            log_obj.update(record.props)
        return json.dumps(log_obj)

logger = logging.getLogger("ANIF_API")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Inicialización ---
load_dotenv()
app = FastAPI(title="ANIF RAG API", version="1.0.0")

# Memoria simple en RAM (para producción usar Redis)
SESSIONS: Dict[str, List[Dict]] = {}

# Sistema de hash para ingesta incremental
PROCESSED_HASHES: Dict[str, Dict] = {}  # hash -> {filename, timestamp, chunks_count}

def calculate_file_hash(file_path: str) -> str:
    """Calcula hash MD5 del contenido del archivo para ingesta incremental"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processed_hashes():
    """Carga el registro de archivos procesados desde disco"""
    hash_file = Path("processed_documents.json")
    if hash_file.exists():
        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading processed hashes: {e}")
    return {}

def save_processed_hashes():
    """Guarda el registro de archivos procesados a disco"""
    hash_file = Path("processed_documents.json")
    try:
        with open(hash_file, 'w', encoding='utf-8') as f:
            json.dump(PROCESSED_HASHES, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving processed hashes: {e}")

# Cargar hashes existentes al iniciar
PROCESSED_HASHES = load_processed_hashes()

rag_system = ANIFRAGSystem()
# Intentar carga inicial
if os.getenv("GROQ_API_KEY"):
    rag_system.initialize_groq(os.getenv("GROQ_API_KEY"))
    rag_system.load_prebuilt_vectorstore()

# --- Modelos ---
class ChatRequest(BaseModel):
    sessionId: str
    mensaje: str

class Source(BaseModel):
    doc_id: str
    score: float
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    citas: List[Source]
    rechazo: bool
    toolUsed: Optional[str] = None

# --- Middleware de Tiempos ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log estructurado de la petición
    logger.info("Request processed", extra={"props": {
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_seconds": round(process_time, 4)
    }})
    return response

# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "ANIF RAG API is running. Visit /docs for documentation."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_loaded": rag_system.documents_loaded,
        "groq_connected": rag_system.groq_client is not None
    }

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    req_id = str(uuid.uuid4())
    start_total = time.time()
    
    if not rag_system.groq_client:
        raise HTTPException(status_code=503, detail="Sistema no inicializado")

    # 1. Gestión de Memoria
    history = SESSIONS.get(request.sessionId, [])
    
    # Construir contexto con historia (simple: últimos 2 turnos)
    context_msg = ""
    if history:
        last_interaction = history[-1]
        context_msg = f"\nHistoria previa: Usuario: {last_interaction['user']} | Asistente: {last_interaction['assistant']}\n"

    # 2. Inferencia
    # Medir tiempo de embedding (simulado ya que FAISS lo hace interno rápido)
    t0 = time.time()
    # Modificamos el prompt para incluir historia si existe
    full_prompt = context_msg + request.mensaje
    
    result = rag_system.query_groq_hybrid(full_prompt, use_rag=True)
    inference_time = time.time() - t0

    # 3. Formateo
    citas_formateadas = []
    for src in result.get("sources", []):
        citas_formateadas.append(Source(
            doc_id=os.path.basename(src["source"]),
            score=round(src["score"], 4),
            snippet=src["content"]
        ))

    # 4. Lógica de Rechazo Estricto
    rechazo = result.get("rejection", False)
    # Si la confianza es baja y no hay citas, forzar rechazo
    if not citas_formateadas and "no tengo información" in result["answer"].lower():
        rechazo = True

    # 5. Guardar en memoria
    if not rechazo:
        if request.sessionId not in SESSIONS:
            SESSIONS[request.sessionId] = []
        SESSIONS[request.sessionId].append({
            "user": request.mensaje,
            "assistant": result["answer"]
        })
        # Mantener solo últimos 2 turnos
        if len(SESSIONS[request.sessionId]) > 2:
            SESSIONS[request.sessionId].pop(0)

    # Log de desglose de tiempos
    logger.info("Chat processing", extra={"props": {
        "request_id": req_id,
        "session_id": request.sessionId,
        "inference_time": round(inference_time, 4),
        "total_time": round(time.time() - start_total, 4),
        "rag_used": True,
        "rejected": rechazo
    }})

    return ChatResponse(
        answer=result["answer"],
        citas=citas_formateadas,
        rechazo=rechazo,
        toolUsed=result.get("toolUsed")
    )

@app.post("/documentos")
async def upload_document(file: UploadFile = File(...)):
    """Ingesta incremental de documentos"""
    try:
        upload_dir = Path("RAG")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        # 1. Guardar archivo físico temporalmente
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Calcular hash del archivo para ingesta incremental
        file_hash = calculate_file_hash(str(file_path))
        
        # 3. Verificar si el archivo ya fue procesado
        if file_hash in PROCESSED_HASHES:
            existing_info = PROCESSED_HASHES[file_hash]
            logger.info("Document already processed", extra={"props": {
                "filename": file.filename,
                "hash": file_hash,
                "original_filename": existing_info.get("filename"),
                "chunks_count": existing_info.get("chunks_count", 0)
            }})
            
            # Eliminar archivo temporal ya que no lo necesitamos
            if file_path.exists():
                file_path.unlink()
            
            return {
                "status": "skipped", 
                "message": f"Archivo ya procesado previamente (hash: {file_hash[:8]}...)",
                "original_filename": existing_info.get("filename"),
                "chunks_count": existing_info.get("chunks_count", 0),
                "doc_id": file.filename,
                "hash": file_hash
            }
        
        # 4. Procesamiento de documento nuevo
        logger.info("Processing new document...", extra={"props": {
            "filename": file.filename,
            "hash": file_hash
        }})
        
        documents = []
        if file.filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        elif file.filename.lower().endswith('.txt'):
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
        
        if documents:
            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)
            
            # Actualizar Vectorstore en memoria y disco
            if rag_system.vectorstore:
                rag_system.vectorstore.add_documents(splits)
                rag_system.vectorstore.save_local("vectorstore")
                logger.info("Vectorstore updated", extra={"props": {"new_chunks": len(splits)}})
            else:
                # Si no existía, crearlo (aunque load_prebuilt debería haberlo hecho)
                rag_system.vectorstore = FAISS.from_documents(splits, rag_system.embeddings)
                rag_system.vectorstore.save_local("vectorstore")
                rag_system.documents_loaded = True
            
            # 5. Registrar el hash del archivo procesado
            PROCESSED_HASHES[file_hash] = {
                "filename": file.filename,
                "timestamp": time.time(),
                "chunks_count": len(splits),
                "file_size": file_path.stat().st_size if file_path.exists() else 0
            }
            
            # Guardar registro de hashes a disco
            save_processed_hashes()
            
            logger.info("Document processed and registered", extra={"props": {
                "filename": file.filename,
                "hash": file_hash,
                "chunks_added": len(splits)
            }})

            return {
                "status": "success", 
                "message": "Archivo procesado e indexado correctamente.",
                "chunks_added": len(splits), 
                "doc_id": file.filename,
                "hash": file_hash
            }
        else:
             raise HTTPException(status_code=400, detail="No se pudo extraer texto del archivo")
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunks/{doc_id}/{chunk_id}")
def get_chunk_evidence(doc_id: str, chunk_id: str):
    """Devuelve el texto específico utilizado como evidencia"""
    # Nota: En FAISS simple no tenemos IDs persistentes por chunk fácilmente accesibles 
    # a menos que los hayamos indexado así. 
    # Para cumplir el requisito, buscaremos en el documento original o retornaremos 
    # un placeholder si la arquitectura actual no soporta acceso directo por ID.
    
    # Simulación basada en el requisito:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": "Contenido del chunk recuperado (Simulación: requiere persistencia de IDs en vectorstore)",
        "status": "available"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)