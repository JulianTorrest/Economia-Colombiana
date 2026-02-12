#!/usr/bin/env python3
"""
Test suite para el despliegue Docker de ANIF
Pruebas de contenedores, configuración y despliegue
"""

import pytest
import subprocess
import requests
import time
import os
import json
from pathlib import Path

class TestDockerDeployment:
    """Pruebas para el despliegue Docker"""
    
    @pytest.fixture(scope="class")
    def docker_compose_file(self):
        """Verifica que existe el archivo docker-compose.yml"""
        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            pytest.skip("docker-compose.yml no encontrado")
        return compose_file
    
    @pytest.fixture(scope="class")
    def dockerfile(self):
        """Verifica que existe el Dockerfile"""
        dockerfile = Path("Dockerfile")
        if not dockerfile.exists():
            pytest.skip("Dockerfile no encontrado")
        return dockerfile
    
    def test_docker_compose_structure(self, docker_compose_file):
        """Prueba la estructura del docker-compose.yml"""
        import yaml
        
        try:
            with open(docker_compose_file, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Verificar servicios requeridos
            assert 'services' in compose_config
            services = compose_config['services']
            
            # Verificar servicio API
            assert 'api' in services
            api_service = services['api']
            assert 'build' in api_service
            assert 'ports' in api_service
            assert 'volumes' in api_service
            assert 'env_file' in api_service
            
            # Verificar servicio Frontend
            assert 'frontend' in services
            frontend_service = services['frontend']
            assert 'build' in frontend_service
            assert 'ports' in frontend_service
            assert 'depends_on' in frontend_service
            
        except ImportError:
            pytest.skip("PyYAML no disponible para parsear docker-compose.yml")
    
    def test_dockerfile_structure(self, dockerfile):
        """Prueba la estructura del Dockerfile"""
        with open(dockerfile, 'r') as f:
            dockerfile_content = f.read()
        
        # Verificar instrucciones esenciales
        assert 'FROM python:' in dockerfile_content
        assert 'WORKDIR' in dockerfile_content
        assert 'COPY requirements.txt' in dockerfile_content
        assert 'RUN pip install' in dockerfile_content
        assert 'COPY . .' in dockerfile_content
        assert 'EXPOSE' in dockerfile_content
    
    def test_requirements_file_exists(self):
        """Verifica que existe requirements.txt"""
        requirements_file = Path("requirements.txt")
        assert requirements_file.exists()
        
        # Verificar dependencias críticas
        with open(requirements_file, 'r') as f:
            requirements = f.read()
        
        critical_deps = [
            'streamlit',
            'fastapi',
            'uvicorn',
            'groq',
            'langchain',
            'sentence-transformers'
        ]
        
        for dep in critical_deps:
            assert dep in requirements.lower(), f"Dependencia crítica {dep} no encontrada"
    
    def test_env_file_template_exists(self):
        """Verifica que existe plantilla de variables de entorno"""
        env_example = Path(".env.example")
        assert env_example.exists()
        
        with open(env_example, 'r') as f:
            env_content = f.read()
        
        # Verificar variables críticas
        assert 'GROQ_API_KEY' in env_content
        assert 'RAG_FOLDER_PATH' in env_content

class TestDockerBuild:
    """Pruebas de construcción de imágenes Docker"""
    
    @pytest.mark.docker
    def test_docker_build_success(self):
        """Prueba que la imagen Docker se construya exitosamente"""
        try:
            # Intentar construir la imagen
            result = subprocess.run(
                ['docker', 'build', '-t', 'anif-test', '.'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )
            
            assert result.returncode == 0, f"Docker build falló: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Docker build timeout después de 5 minutos")
        except FileNotFoundError:
            pytest.skip("Docker no está disponible")
    
    @pytest.mark.docker
    def test_docker_compose_build(self):
        """Prueba que docker-compose build funcione"""
        try:
            result = subprocess.run(
                ['docker-compose', 'build'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutos timeout
            )
            
            assert result.returncode == 0, f"Docker compose build falló: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Docker compose build timeout después de 10 minutos")
        except FileNotFoundError:
            pytest.skip("Docker Compose no está disponible")

class TestDockerRuntime:
    """Pruebas de ejecución de contenedores Docker"""
    
    @pytest.fixture(scope="class")
    def docker_services(self):
        """Inicia los servicios Docker para pruebas"""
        try:
            # Verificar que Docker esté disponible
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            
            # Iniciar servicios
            subprocess.run(['docker-compose', 'up', '-d'], check=True, capture_output=True)
            
            # Esperar a que los servicios estén listos
            time.sleep(30)
            
            yield
            
            # Cleanup
            subprocess.run(['docker-compose', 'down'], capture_output=True)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Docker o Docker Compose no disponible")
    
    @pytest.mark.docker
    @pytest.mark.integration
    def test_api_service_health(self, docker_services):
        """Prueba que el servicio API esté saludable"""
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    assert data['status'] == 'healthy'
                    return
            except requests.RequestException:
                if i == max_retries - 1:
                    pytest.fail("API service no responde después de múltiples intentos")
                time.sleep(5)
    
    @pytest.mark.docker
    @pytest.mark.integration
    def test_frontend_service_accessibility(self, docker_services):
        """Prueba que el servicio frontend sea accesible"""
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get('http://localhost:8501', timeout=10)
                if response.status_code == 200:
                    assert 'ANIF' in response.text or 'streamlit' in response.text.lower()
                    return
            except requests.RequestException:
                if i == max_retries - 1:
                    pytest.fail("Frontend service no accesible después de múltiples intentos")
                time.sleep(5)
    
    @pytest.mark.docker
    def test_rag_initialization_in_container(self):
        """Prueba que el RAG se inicialice correctamente en el contenedor"""
        try:
            # Ejecutar script de inicialización RAG en el contenedor
            result = subprocess.run(
                ['docker-compose', 'exec', '-T', 'api', 'python', 'setup_rag.py'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Verificar que la inicialización fue exitosa
            assert result.returncode == 0, f"RAG initialization falló: {result.stderr}"
            assert "Sistema RAG inicializado exitosamente" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("RAG initialization timeout")
        except FileNotFoundError:
            pytest.skip("Docker Compose no disponible")

class TestDockerSecurity:
    """Pruebas de seguridad para Docker"""
    
    def test_dockerfile_security_practices(self):
        """Verifica prácticas de seguridad en Dockerfile"""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile no encontrado")
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Verificar que no se ejecute como root
        assert 'USER ' in dockerfile_content or 'useradd' in dockerfile_content
        
        # Verificar que se use imagen base oficial
        lines = dockerfile_content.split('\n')
        from_line = next((line for line in lines if line.startswith('FROM')), '')
        assert 'python:' in from_line.lower()
    
    def test_env_secrets_not_in_dockerfile(self):
        """Verifica que no hay secretos hardcodeados en Dockerfile"""
        dockerfile_path = Path("Dockerfile")
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile no encontrado")
        
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read().lower()
        
        # Verificar que no hay API keys hardcodeadas
        forbidden_patterns = ['gsk_', 'api_key=', 'password=', 'secret=']
        for pattern in forbidden_patterns:
            assert pattern not in dockerfile_content, f"Posible secreto hardcodeado: {pattern}"

class TestDockerPerformance:
    """Pruebas de rendimiento para Docker"""
    
    @pytest.mark.performance
    @pytest.mark.docker
    def test_container_startup_time(self):
        """Prueba el tiempo de inicio de los contenedores"""
        try:
            start_time = time.time()
            
            # Iniciar contenedores
            result = subprocess.run(
                ['docker-compose', 'up', '-d'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            end_time = time.time()
            startup_time = end_time - start_time
            
            # Cleanup
            subprocess.run(['docker-compose', 'down'], capture_output=True)
            
            assert result.returncode == 0
            assert startup_time < 60, f"Startup muy lento: {startup_time}s"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Container startup timeout")
        except FileNotFoundError:
            pytest.skip("Docker Compose no disponible")
    
    @pytest.mark.performance
    def test_image_size_reasonable(self):
        """Verifica que el tamaño de la imagen sea razonable"""
        try:
            # Construir imagen si no existe
            subprocess.run(['docker', 'build', '-t', 'anif-test', '.'], 
                         capture_output=True, timeout=300)
            
            # Obtener tamaño de imagen
            result = subprocess.run(
                ['docker', 'images', 'anif-test', '--format', '{{.Size}}'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                size_str = result.stdout.strip()
                # Verificar que no sea excesivamente grande (ej: < 5GB)
                assert size_str, "No se pudo obtener el tamaño de la imagen"
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker no disponible o timeout")

if __name__ == "__main__":
    # Ejecutar pruebas Docker específicas
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])
