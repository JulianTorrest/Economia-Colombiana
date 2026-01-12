#!/usr/bin/env python3
"""
Script de actualización incremental del sistema RAG para ANIF
Procesa solo documentos nuevos sin reentrenar los existentes
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Set
from datetime import datetime

def load_processed_files() -> Dict[str, str]:
    """Carga la lista de archivos ya procesados"""
    processed_file = "processed_documents.json"
    if os.path.exists(processed_file):
        with open(processed_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_processed_files(processed_files: Dict[str, str]):
    """Guarda la lista de archivos procesados"""
    with open("processed_documents.json", 'w', encoding='utf-8') as f:
        json.dump(processed_files, f, indent=2, ensure_ascii=False)

def get_file_hash(file_path: Path) -> str:
    """Calcula el hash MD5 de un archivo"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_new_documents(folder_path: str) -> List[Path]:
    """Encuentra documentos nuevos o modificados"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ ERROR: La carpeta {folder_path} no existe")
        return []
    
    processed_files = load_processed_files()
    new_files = []
    
    print(f"🔍 Escaneando carpeta {folder_path} en busca de archivos nuevos...")
    
    for file_path in folder.glob("*"):
        if file_path.suffix.lower() in ['.pdf', '.txt']:
            file_key = str(file_path.relative_to(folder))
            current_hash = get_file_hash(file_path)
            
            # Verificar si es nuevo o modificado
            if file_key not in processed_files or processed_files[file_key] != current_hash:
                new_files.append(file_path)
                print(f"📄 Nuevo/Modificado: {file_path.name}")
    
    if not new_files:
        print("✅ No se encontraron archivos nuevos para procesar")
    else:
        print(f"📊 {len(new_files)} archivos nuevos encontrados")
    
    return new_files

def load_new_documents(file_paths: List[Path]) -> List[Document]:
    """Carga solo los documentos nuevos"""
    documents = []
    
    print("🔄 Procesando documentos nuevos...")
    
    for i, file_path in enumerate(file_paths):
        try:
            print(f"📄 Procesando ({i+1}/{len(file_paths)}): {file_path.name}")
            
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                
        except Exception as e:
            print(f"⚠️ Error procesando {file_path.name}: {str(e)}")
            continue
    
    print(f"✅ {len(documents)} documentos nuevos cargados")
    return documents

def update_vectorstore(new_documents: List[Document], embeddings):
    """Actualiza el vectorstore existente con documentos nuevos"""
    try:
        print("🔄 Actualizando base de datos vectorial...")
        
        if not new_documents:
            print("ℹ️ No hay documentos nuevos para agregar")
            return True
        
        # Cargar vectorstore existente
        if not os.path.exists("vectorstore"):
            print("❌ ERROR: Vectorstore existente no encontrado")
            return False
        
        vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        print("✅ Vectorstore existente cargado")
        
        # Dividir nuevos documentos en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        new_texts = text_splitter.split_documents(new_documents)
        print(f"📝 {len(new_texts)} nuevos fragmentos de texto creados")
        
        if new_texts:
            # Agregar nuevos documentos al vectorstore existente
            vectorstore.add_documents(new_texts)
            
            # Guardar vectorstore actualizado
            vectorstore.save_local("vectorstore")
            print(f"✅ Vectorstore actualizado con {len(new_texts)} nuevos fragmentos")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR al actualizar vectorstore: {str(e)}")
        return False

def update_processed_files_registry(file_paths: List[Path], folder_path: str):
    """Actualiza el registro de archivos procesados"""
    processed_files = load_processed_files()
    folder = Path(folder_path)
    
    for file_path in file_paths:
        file_key = str(file_path.relative_to(folder))
        file_hash = get_file_hash(file_path)
        processed_files[file_key] = file_hash
    
    save_processed_files(processed_files)
    print(f"✅ Registro actualizado con {len(file_paths)} archivos")

def update_rag_flag(total_new_docs: int):
    """Actualiza el archivo de estado del sistema"""
    try:
        # Leer estado actual
        current_count = 0
        if os.path.exists("rag_ready.flag"):
            with open("rag_ready.flag", 'r') as f:
                for line in f:
                    if line.startswith("DOCUMENTS_COUNT="):
                        current_count = int(line.split("=")[1].strip())
                        break
        
        # Actualizar con nuevos documentos
        new_total = current_count + total_new_docs
        
        with open("rag_ready.flag", "w") as f:
            f.write("RAG_SYSTEM_READY=True\n")
            f.write(f"DOCUMENTS_COUNT={new_total}\n")
            f.write(f"VECTORSTORE_READY=True\n")
            f.write(f"LAST_UPDATE={datetime.now().isoformat()}\n")
        
        print(f"✅ Estado actualizado: {current_count} → {new_total} documentos")
        
    except Exception as e:
        print(f"⚠️ Error actualizando estado: {str(e)}")

def main():
    """Función principal de actualización incremental"""
    print("🔄 ANIF RAG System - Actualización Incremental")
    print("=" * 50)
    
    # 1. Verificar que el sistema base existe
    if not os.path.exists("rag_ready.flag"):
        print("❌ ERROR: Sistema RAG no inicializado. Ejecuta 'python setup_rag.py' primero.")
        sys.exit(1)
    
    if not os.path.exists("vectorstore"):
        print("❌ ERROR: Vectorstore base no encontrado. Ejecuta 'python setup_rag.py' primero.")
        sys.exit(1)
    
    # 2. Cargar configuración
    print("\n1️⃣ Cargando configuración...")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY no encontrada en .env")
        sys.exit(1)
    
    # 3. Inicializar embeddings
    print("\n2️⃣ Inicializando sistema de embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("✅ Sistema de embeddings inicializado")
    except Exception as e:
        print(f"❌ ERROR al inicializar embeddings: {str(e)}")
        sys.exit(1)
    
    # 4. Encontrar documentos nuevos
    print("\n3️⃣ Buscando documentos nuevos...")
    rag_folder = r"C:\Users\betol\OneDrive\Documentos\Economia Colombiana - ANIF\RAG"
    new_files = find_new_documents(rag_folder)
    
    if not new_files:
        print("\n🎉 ¡Sistema actualizado! No hay documentos nuevos para procesar.")
        return
    
    # 5. Cargar documentos nuevos
    print(f"\n4️⃣ Cargando {len(new_files)} documentos nuevos...")
    new_documents = load_new_documents(new_files)
    
    if not new_documents:
        print("⚠️ No se pudieron cargar documentos nuevos")
        return
    
    # 6. Actualizar vectorstore
    print(f"\n5️⃣ Actualizando base de datos vectorial...")
    if not update_vectorstore(new_documents, embeddings):
        sys.exit(1)
    
    # 7. Actualizar registro de archivos procesados
    print(f"\n6️⃣ Actualizando registro...")
    update_processed_files_registry(new_files, rag_folder)
    
    # 8. Actualizar estado del sistema
    print(f"\n7️⃣ Actualizando estado del sistema...")
    update_rag_flag(len(new_documents))
    
    print("\n" + "=" * 50)
    print("🎉 ¡Actualización incremental completada exitosamente!")
    print(f"📊 {len(new_documents)} nuevos documentos agregados al sistema")
    print("✅ El sistema está listo para usar con los nuevos documentos")
    print("=" * 50)

if __name__ == "__main__":
    main()
