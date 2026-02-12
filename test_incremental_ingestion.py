#!/usr/bin/env python3
"""
Test para verificar la funcionalidad de ingesta incremental con hash
"""

import requests
import tempfile
import os
from pathlib import Path

def test_incremental_ingestion():
    """Prueba la ingesta incremental con hash"""
    
    # URL base de la API
    base_url = "http://localhost:8000"
    
    # Crear archivo de prueba
    test_content = """
    Este es un documento de prueba para verificar la ingesta incremental.
    
    Contenido económico de ejemplo:
    - PIB de Colombia: 314 mil millones USD
    - Inflación objetivo: 3%
    - Tasa de cambio: Variable
    """
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file_path = f.name
    
    try:
        print("🧪 Prueba de Ingesta Incremental")
        print("=" * 50)
        
        # 1. Primera subida - debe procesar
        print("\n1️⃣ Primera subida del archivo...")
        with open(temp_file_path, 'rb') as f:
            files = {'file': ('test_document.txt', f, 'text/plain')}
            response1 = requests.post(f"{base_url}/documentos", files=files)
        
        print(f"Status: {response1.status_code}")
        result1 = response1.json()
        print(f"Respuesta: {result1}")
        
        if result1.get("status") == "success":
            print("✅ Primera subida exitosa")
            file_hash = result1.get("hash")
            chunks_added = result1.get("chunks_added")
            print(f"Hash: {file_hash}")
            print(f"Chunks agregados: {chunks_added}")
        else:
            print("❌ Error en primera subida")
            return False
        
        # 2. Segunda subida del mismo archivo - debe saltar
        print("\n2️⃣ Segunda subida del mismo archivo...")
        with open(temp_file_path, 'rb') as f:
            files = {'file': ('test_document_copy.txt', f, 'text/plain')}
            response2 = requests.post(f"{base_url}/documentos", files=files)
        
        print(f"Status: {response2.status_code}")
        result2 = response2.json()
        print(f"Respuesta: {result2}")
        
        if result2.get("status") == "skipped":
            print("✅ Segunda subida saltada correctamente (archivo duplicado)")
            print(f"Hash detectado: {result2.get('hash')}")
            print(f"Archivo original: {result2.get('original_filename')}")
        else:
            print("❌ Error: debería haber saltado el archivo duplicado")
            return False
        
        # 3. Modificar archivo y subir - debe procesar
        print("\n3️⃣ Subida de archivo modificado...")
        modified_content = test_content + "\n\nContenido adicional modificado."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(modified_content)
            modified_file_path = f.name
        
        try:
            with open(modified_file_path, 'rb') as f:
                files = {'file': ('test_document_modified.txt', f, 'text/plain')}
                response3 = requests.post(f"{base_url}/documentos", files=files)
            
            print(f"Status: {response3.status_code}")
            result3 = response3.json()
            print(f"Respuesta: {result3}")
            
            if result3.get("status") == "success":
                print("✅ Archivo modificado procesado correctamente")
                new_hash = result3.get("hash")
                print(f"Nuevo hash: {new_hash}")
                print(f"Hash diferente al original: {new_hash != file_hash}")
            else:
                print("❌ Error procesando archivo modificado")
                return False
        
        finally:
            # Limpiar archivo modificado
            if os.path.exists(modified_file_path):
                os.unlink(modified_file_path)
        
        print("\n" + "=" * 50)
        print("🎉 ¡Prueba de ingesta incremental EXITOSA!")
        print("✅ Archivos idénticos se saltan correctamente")
        print("✅ Archivos modificados se procesan correctamente")
        print("✅ Sistema de hash funcionando perfectamente")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a la API")
        print("💡 Asegúrate de que la API esté ejecutándose en http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    success = test_incremental_ingestion()
    exit(0 if success else 1)
