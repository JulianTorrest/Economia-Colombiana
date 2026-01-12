#!/usr/bin/env python3
"""
Script de inicialización del sistema RAG para ANIF
Este script debe ejecutarse ANTES del despliegue para asegurar que todo esté operativo
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import pickle

def load_environment():
    """Carga las variables de entorno"""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY no encontrada en .env")
        return None
    return api_key

def validate_groq_connection(api_key: str) -> bool:
    """Valida la conexión con Groq"""
    try:
        if not api_key.startswith('gsk_'):
            print("❌ ERROR: Formato de API key inválido")
            return False
            
        client = Groq(api_key=api_key)
        
        # Test de conexión real
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            model="llama-3.1-8b-instant",
            max_tokens=10
        )
        
        print("✅ Conexión con Groq validada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ ERROR al conectar con Groq: {str(e)}")
        return False

def initialize_embeddings():
    """Inicializa el sistema de embeddings"""
    try:
        print("🔄 Inicializando sistema de embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("✅ Sistema de embeddings inicializado")
        return embeddings
    except Exception as e:
        print(f"❌ ERROR al inicializar embeddings: {str(e)}")
        return None

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Carga documentos desde una carpeta"""
    documents = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ ERROR: La carpeta {folder_path} no existe")
        return documents
    
    print(f"🔄 Cargando documentos desde {folder_path}...")
    
    files = list(folder.glob("*"))
    total_files = len(files)
    
    for i, file_path in enumerate(files):
        try:
            print(f"📄 Procesando ({i+1}/{total_files}): {file_path.name}")
            
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
    
    print(f"✅ {len(documents)} documentos cargados exitosamente")
    return documents

def create_vectorstore(documents: List[Document], embeddings):
    """Crea y guarda el vectorstore"""
    try:
        print("🔄 Creando vectorstore...")
        
        if not documents:
            print("❌ ERROR: No hay documentos para crear el vectorstore")
            return None
        
        # Dividir documentos en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = text_splitter.split_documents(documents)
        print(f"📝 {len(texts)} fragmentos de texto creados")
        
        if not texts:
            print("❌ ERROR: No se generaron fragmentos de texto")
            return None
        
        # Crear vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Guardar vectorstore
        vectorstore_path = "vectorstore"
        vectorstore.save_local(vectorstore_path)
        print(f"✅ Vectorstore guardado en {vectorstore_path}")
        
        return vectorstore
        
    except Exception as e:
        print(f"❌ ERROR al crear vectorstore: {str(e)}")
        return None

def main():
    """Función principal de inicialización"""
    print("🏛️ ANIF RAG System - Inicialización Pre-Despliegue")
    print("=" * 50)
    
    # 1. Cargar variables de entorno
    print("\n1️⃣ Validando configuración...")
    api_key = load_environment()
    if not api_key:
        sys.exit(1)
    
    # 2. Validar conexión con Groq
    print("\n2️⃣ Validando conexión con Groq...")
    if not validate_groq_connection(api_key):
        sys.exit(1)
    
    # 3. Inicializar embeddings
    print("\n3️⃣ Inicializando sistema de embeddings...")
    embeddings = initialize_embeddings()
    if not embeddings:
        sys.exit(1)
    
    # 4. Cargar documentos
    print("\n4️⃣ Cargando documentos RAG...")
    rag_folder = r"C:\Users\betol\OneDrive\Documentos\Economia Colombiana - ANIF\RAG"
    documents = load_documents_from_folder(rag_folder)
    if not documents:
        print("⚠️ ADVERTENCIA: No se cargaron documentos")
    
    # 5. Crear vectorstore
    print("\n5️⃣ Creando base de datos vectorial...")
    vectorstore = create_vectorstore(documents, embeddings)
    if not vectorstore:
        sys.exit(1)
    
    # 6. Crear archivo de estado
    print("\n6️⃣ Creando archivo de estado...")
    with open("rag_ready.flag", "w") as f:
        f.write("RAG_SYSTEM_READY=True\n")
        f.write(f"DOCUMENTS_COUNT={len(documents)}\n")
        f.write(f"VECTORSTORE_READY=True\n")
    
    print("\n" + "=" * 50)
    print("🎉 ¡Sistema RAG inicializado exitosamente!")
    print("✅ El sistema está listo para despliegue")
    print("=" * 50)

if __name__ == "__main__":
    main()
