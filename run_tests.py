#!/usr/bin/env python3
"""
Script para ejecutar todas las pruebas del sistema ANIF
Incluye diferentes categorías de pruebas con opciones de configuración
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path

def run_command(cmd, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - EXITOSO")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} - FALLÓ")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error ejecutando {description}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ejecutar pruebas del sistema ANIF")
    parser.add_argument("--quick", action="store_true", help="Solo pruebas rápidas (sin Docker/performance)")
    parser.add_argument("--api-only", action="store_true", help="Solo pruebas de API")
    parser.add_argument("--rag-only", action="store_true", help="Solo pruebas del sistema RAG")
    parser.add_argument("--streamlit-only", action="store_true", help="Solo pruebas de Streamlit")
    parser.add_argument("--docker-only", action="store_true", help="Solo pruebas de Docker")
    parser.add_argument("--integration", action="store_true", help="Incluir pruebas de integración")
    parser.add_argument("--performance", action="store_true", help="Incluir pruebas de rendimiento")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output verbose")
    
    args = parser.parse_args()
    
    print("🏛️ ANIF - Suite de Pruebas Automatizadas")
    print("=" * 50)
    
    # Verificar que pytest esté instalado
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("❌ pytest no está instalado. Instalando...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
    
    # Configurar opciones base de pytest
    pytest_opts = [sys.executable, "-m", "pytest"]
    if args.verbose:
        pytest_opts.append("-v")
    
    # Configurar marcadores para excluir según argumentos
    exclude_markers = []
    if args.quick:
        exclude_markers.extend(["docker", "performance", "slow"])
    
    if exclude_markers:
        pytest_opts.extend(["-m", f"not ({' or '.join(exclude_markers)})"])
    
    success_count = 0
    total_tests = 0
    
    # Ejecutar pruebas según argumentos
    if args.api_only:
        total_tests += 1
        if run_command(" ".join(pytest_opts + ["test_api.py"]), "Pruebas de API"):
            success_count += 1
    
    elif args.rag_only:
        total_tests += 1
        if run_command(" ".join(pytest_opts + ["test_rag_system.py"]), "Pruebas del Sistema RAG"):
            success_count += 1
    
    elif args.streamlit_only:
        total_tests += 1
        if run_command(" ".join(pytest_opts + ["test_streamlit_app.py"]), "Pruebas de Streamlit"):
            success_count += 1
    
    elif args.docker_only:
        total_tests += 1
        if run_command(" ".join(pytest_opts + ["test_docker_deployment.py"]), "Pruebas de Docker"):
            success_count += 1
    
    else:
        # Ejecutar todas las pruebas
        test_files = [
            ("test_api.py", "Pruebas de API"),
            ("test_rag_system.py", "Pruebas del Sistema RAG"),
            ("test_streamlit_app.py", "Pruebas de Streamlit"),
        ]
        
        # Agregar pruebas de Docker solo si no es quick
        if not args.quick:
            test_files.append(("test_docker_deployment.py", "Pruebas de Docker"))
        
        for test_file, description in test_files:
            if Path(test_file).exists():
                total_tests += 1
                if run_command(" ".join(pytest_opts + [test_file]), description):
                    success_count += 1
    
    # Ejecutar pruebas de integración si se solicita
    if args.integration:
        total_tests += 1
        integration_opts = pytest_opts + ["-m", "integration"]
        if run_command(" ".join(integration_opts), "Pruebas de Integración"):
            success_count += 1
    
    # Ejecutar pruebas de rendimiento si se solicita
    if args.performance:
        total_tests += 1
        performance_opts = pytest_opts + ["-m", "performance"]
        if run_command(" ".join(performance_opts), "Pruebas de Rendimiento"):
            success_count += 1
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    print(f"✅ Exitosas: {success_count}/{total_tests}")
    print(f"❌ Fallidas: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ¡Todas las pruebas pasaron exitosamente!")
        return 0
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa los logs arriba.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
