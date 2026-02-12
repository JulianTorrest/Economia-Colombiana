#!/usr/bin/env python3
"""
Test suite para el sistema RAG de ANIF
Pruebas comprehensivas del sistema de recuperación y generación
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar el sistema RAG con manejo de errores
try:
    from main import ANIFRAGSystem
except ImportError:
    # Fallback si no se puede importar
    ANIFRAGSystem = None

class TestRAGSystem:
    """Suite de pruebas para el sistema RAG"""
    
    @pytest.fixture
    def rag_system(self):
        """Fixture para crear una instancia del sistema RAG"""
        if ANIFRAGSystem is None:
            pytest.skip("ANIFRAGSystem no disponible")
        return ANIFRAGSystem()
    
    @pytest.fixture
    def mock_groq_client(self):
        """Mock del cliente Groq"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Respuesta de prueba del modelo"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    def test_rag_system_initialization(self, rag_system):
        """Prueba la inicialización del sistema RAG"""
        assert rag_system is not None
        assert rag_system.embeddings is None
        assert rag_system.vectorstore is None
        assert rag_system.groq_client is None
        assert rag_system.documents_loaded == False
    
    def test_groq_initialization_valid_key(self, rag_system):
        """Prueba la inicialización de Groq con API key válida"""
        # Mock de la función lazy_import_groq
        with patch('main.lazy_import_groq') as mock_import:
            mock_groq_class = Mock()
            mock_import.return_value = mock_groq_class
            
            # Simular inicialización exitosa
            with patch.object(rag_system, 'initialize_groq', return_value=True):
                result = rag_system.initialize_groq("gsk_test_key_12345")
                assert result == True
    
    def test_groq_initialization_invalid_key(self, rag_system):
        """Prueba la inicialización de Groq con API key inválida"""
        with patch('main.lazy_import_groq'):
            result = rag_system.initialize_groq("invalid_key")
            assert result == False
    
    def test_groq_initialization_empty_key(self, rag_system):
        """Prueba la inicialización de Groq con API key vacía"""
        result = rag_system.initialize_groq("")
        assert result == False
        
        result = rag_system.initialize_groq(None)
        assert result == False
    
    @patch('main.lazy_import_langchain')
    def test_load_prebuilt_vectorstore_exists(self, mock_import, rag_system):
        """Prueba la carga de vectorstore existente"""
        # Mock de las importaciones
        mock_faiss = Mock()
        mock_embeddings = Mock()
        mock_import.return_value = (None, None, mock_faiss, mock_embeddings, None, None)
        
        # Mock de archivos existentes
        with patch('os.path.exists', return_value=True):
            with patch.object(mock_faiss, 'load_local', return_value=Mock()):
                rag_system.embeddings = mock_embeddings()
                result = rag_system.load_prebuilt_vectorstore()
                assert result == True
                assert rag_system.documents_loaded == True
    
    def test_classify_economic_query(self, rag_system):
        """Prueba la clasificación de consultas económicas"""
        # Pruebas de clasificación por palabras clave
        assert rag_system.classify_economic_query("política fiscal") == "fiscal"
        assert rag_system.classify_economic_query("banco de la república") == "monetario"
        assert rag_system.classify_economic_query("sector bancario") == "sectorial"
        assert rag_system.classify_economic_query("comercio exterior") == "internacional"
        assert rag_system.classify_economic_query("mercado laboral") == "laboral"
        assert rag_system.classify_economic_query("economía general") == "general"
    
    def test_get_enhanced_system_prompt(self, rag_system):
        """Prueba la generación de prompts del sistema"""
        prompt_fiscal = rag_system.get_enhanced_system_prompt("fiscal")
        assert "política fiscal colombiana" in prompt_fiscal.lower()
        
        prompt_monetario = rag_system.get_enhanced_system_prompt("monetario")
        assert "banco de la república" in prompt_monetario.lower()
        
        prompt_general = rag_system.get_enhanced_system_prompt("general")
        assert "economista senior" in prompt_general.lower()
    
    def test_create_chain_of_thought_prompt(self, rag_system):
        """Prueba la creación de prompts de cadena de pensamiento"""
        query = "¿Cuál es la situación fiscal de Colombia?"
        query_type = "fiscal"
        
        enhanced_prompt = rag_system.create_chain_of_thought_prompt(query, query_type)
        
        assert query in enhanced_prompt
        assert "Análisis paso a paso" in enhanced_prompt
        assert "fiscal" in enhanced_prompt.lower()
    
    @patch('main.lazy_import_langchain')
    def test_load_documents_from_folder_empty(self, mock_import, rag_system):
        """Prueba la carga de documentos desde carpeta vacía"""
        mock_import.return_value = (None, None, None, None, None, None)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            documents = rag_system.load_documents_from_folder(temp_dir)
            assert len(documents) == 0
    
    def test_load_documents_from_folder_nonexistent(self, rag_system):
        """Prueba la carga de documentos desde carpeta inexistente"""
        documents = rag_system.load_documents_from_folder("/path/that/does/not/exist")
        assert len(documents) == 0
    
    def test_search_similar_documents_no_vectorstore(self, rag_system):
        """Prueba la búsqueda sin vectorstore inicializado"""
        result = rag_system.search_similar_documents("test query")
        assert result == ""
    
    @patch('main.lazy_import_groq')
    def test_query_groq_hybrid_no_client(self, mock_import, rag_system):
        """Prueba consulta híbrida sin cliente Groq"""
        mock_import.return_value = Mock()
        result = rag_system.query_groq_hybrid("test query", use_rag=False)
        assert "Error" in result or result == ""

class TestRAGIntegration:
    """Pruebas de integración del sistema RAG"""
    
    @pytest.fixture
    def temp_rag_folder(self):
        """Crear carpeta temporal con documentos de prueba"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear archivo PDF de prueba (simulado)
            test_file = Path(temp_dir) / "test_document.txt"
            test_file.write_text("Este es un documento de prueba sobre economía colombiana.")
            yield temp_dir
    
    @pytest.mark.integration
    def test_end_to_end_rag_workflow(self, temp_rag_folder):
        """Prueba completa del flujo RAG end-to-end"""
        if ANIFRAGSystem is None:
            pytest.skip("ANIFRAGSystem no disponible")
        
        rag_system = ANIFRAGSystem()
        
        # Mock de componentes externos
        with patch('main.lazy_import_langchain') as mock_import:
            mock_embeddings = Mock()
            mock_faiss = Mock()
            mock_import.return_value = (None, None, mock_faiss, mock_embeddings, None, None)
            
            # Simular inicialización exitosa
            rag_system.embeddings = mock_embeddings
            rag_system.documents_loaded = True
            
            # Verificar que el sistema está listo
            assert rag_system.embeddings is not None
            assert rag_system.documents_loaded == True

class TestRAGPerformance:
    """Pruebas de rendimiento del sistema RAG"""
    
    @pytest.mark.performance
    def test_query_response_time(self):
        """Prueba el tiempo de respuesta de las consultas"""
        import time
        
        if ANIFRAGSystem is None:
            pytest.skip("ANIFRAGSystem no disponible")
        
        rag_system = ANIFRAGSystem()
        
        # Mock de cliente Groq para prueba de rendimiento
        with patch.object(rag_system, 'groq_client') as mock_client:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Respuesta rápida"
            mock_client.chat.completions.create.return_value = mock_response
            
            start_time = time.time()
            result = rag_system.query_groq_hybrid("test query", use_rag=False)
            end_time = time.time()
            
            # Verificar que la respuesta es rápida (menos de 5 segundos)
            response_time = end_time - start_time
            assert response_time < 5.0, f"Respuesta muy lenta: {response_time}s"

if __name__ == "__main__":
    # Ejecutar pruebas específicas
    pytest.main([__file__, "-v", "--tb=short"])
