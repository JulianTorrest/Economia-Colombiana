#!/usr/bin/env python3
"""
Script para crear el registro inicial de documentos procesados
Ejecutar UNA SOLA VEZ después de la inicialización completa del sistema
"""

import os
import json
import hashlib
from pathlib import Path

def get_file_hash(file_path: Path) -> str:
    """Calcula el hash MD5 de un archivo"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def create_initial_registry():
    """Crea el registro inicial de todos los archivos actualmente en RAG"""
    rag_folder = Path(r"C:\Users\betol\OneDrive\Documentos\Economia Colombiana - ANIF\RAG")
    
    if not rag_folder.exists():
        print(f"ERROR: La carpeta RAG no existe: {rag_folder}")
        return
    
    processed_files = {}
    
    print("Creando registro inicial de documentos procesados...")
    
    for file_path in rag_folder.glob("*"):
        if file_path.suffix.lower() in ['.pdf', '.txt']:
            try:
                file_key = str(file_path.relative_to(rag_folder))
                file_hash = get_file_hash(file_path)
                processed_files[file_key] = file_hash
                print(f"Registrado: {file_path.name}")
            except Exception as e:
                print(f"Error procesando {file_path.name}: {str(e)}")
    
    # Guardar registro
    with open("processed_documents.json", 'w', encoding='utf-8') as f:
        json.dump(processed_files, f, indent=2, ensure_ascii=False)
    
    print(f"Registro inicial creado con {len(processed_files)} archivos")
    print("Archivo guardado: processed_documents.json")

if __name__ == "__main__":
    create_initial_registry()
