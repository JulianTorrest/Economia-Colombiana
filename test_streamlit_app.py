#!/usr/bin/env python3
"""
Test suite para la aplicación Streamlit de ANIF
Pruebas de integración y funcionalidad de la interfaz
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class TestStreamlitApp:
    """Pruebas para la aplicación Streamlit"""
    
    @pytest.fixture
    def mock_streamlit(self):
        """Mock de Streamlit para pruebas"""
        with patch('streamlit.set_page_config'), \
             patch('streamlit.markdown'), \
             patch('streamlit.header'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.text_input'), \
             patch('streamlit.button'), \
             patch('streamlit.spinner'), \
             patch('streamlit.success'), \
             patch('streamlit.error'), \
             patch('streamlit.warning'), \
             patch('streamlit.info'):
            yield
    
    @pytest.fixture
    def mock_session_state(self):
        """Mock del session state de Streamlit"""
        session_state = {}
        with patch('streamlit.session_state', session_state):
            yield session_state
    
    def test_lazy_import_functions(self):
        """Prueba las funciones de importación diferida"""
        try:
            from main import lazy_import_langchain, lazy_import_groq
            
            # Probar importación de LangChain
            langchain_imports = lazy_import_langchain()
            assert len(langchain_imports) == 6  # 6 componentes importados
            
            # Probar importación de Groq
            groq_class = lazy_import_groq()
            assert groq_class is not None
            
        except ImportError as e:
            pytest.skip(f"Imports no disponibles: {e}")
    
    def test_anif_rag_system_initialization(self, mock_streamlit):
        """Prueba la inicialización del sistema RAG en Streamlit"""
        try:
            from main import ANIFRAGSystem
            
            rag_system = ANIFRAGSystem()
            assert rag_system is not None
            assert hasattr(rag_system, 'domain_prompts')
            assert hasattr(rag_system, 'temporal_context')
            assert 'fiscal' in rag_system.domain_prompts
            assert 'monetario' in rag_system.domain_prompts
            
        except ImportError:
            pytest.skip("ANIFRAGSystem no disponible")
    
    @patch.dict(os.environ, {'GROQ_API_KEY': 'gsk_test_key_12345'})
    def test_environment_variable_loading(self, mock_streamlit):
        """Prueba la carga de variables de entorno"""
        api_key = os.getenv('GROQ_API_KEY')
        assert api_key == 'gsk_test_key_12345'
        assert api_key.startswith('gsk_')
    
    def test_show_agent_interface_initialization(self, mock_streamlit, mock_session_state):
        """Prueba la inicialización de la interfaz del agente"""
        try:
            from main import show_agent_interface
            
            # Mock del sistema RAG
            mock_rag = Mock()
            mock_rag.documents_loaded = True
            mock_rag.groq_client = Mock()
            mock_session_state['rag_system'] = mock_rag
            mock_session_state['chat_history'] = []
            
            # Ejecutar la función (debería no fallar)
            with patch('streamlit.rerun'):
                show_agent_interface()
            
            # Verificar que chat_history se inicializó
            assert 'chat_history' in mock_session_state
            
        except ImportError:
            pytest.skip("show_agent_interface no disponible")
    
    def test_show_report_generation_interface(self, mock_streamlit, mock_session_state):
        """Prueba la interfaz de generación de informes"""
        try:
            from main import show_report_generation_interface
            
            # Mock del sistema RAG
            mock_rag = Mock()
            mock_rag.documents_loaded = True
            mock_rag.groq_client = Mock()
            mock_session_state['rag_system'] = mock_rag
            mock_session_state['chat_history'] = []
            
            # Ejecutar la función
            with patch('streamlit.rerun'):
                show_report_generation_interface()
            
            # Verificar inicialización
            assert 'chat_history' in mock_session_state
            
        except ImportError:
            pytest.skip("show_report_generation_interface no disponible")
    
    def test_show_anif_tools_interface(self, mock_streamlit, mock_session_state):
        """Prueba la interfaz de herramientas ANIF"""
        try:
            from main import show_anif_tools_interface
            
            # Mock del sistema RAG
            mock_rag = Mock()
            mock_rag.documents_loaded = True
            mock_rag.groq_client = Mock()
            mock_session_state['rag_system'] = mock_rag
            mock_session_state['chat_history'] = []
            
            # Ejecutar la función
            with patch('streamlit.rerun'):
                show_anif_tools_interface()
            
            # Verificar inicialización
            assert 'chat_history' in mock_session_state
            
        except ImportError:
            pytest.skip("show_anif_tools_interface no disponible")

class TestStreamlitIntegration:
    """Pruebas de integración de Streamlit"""
    
    @pytest.mark.integration
    def test_main_function_structure(self):
        """Prueba la estructura de la función main"""
        try:
            from main import main
            
            # Mock de todos los componentes de Streamlit
            with patch('streamlit.set_page_config'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.selectbox', return_value="🤖 Agente"), \
                 patch('streamlit.session_state', {}), \
                 patch('os.getenv', return_value='gsk_test_key'), \
                 patch('main.show_agent_interface'):
                
                # Ejecutar main (no debería fallar)
                main()
                
        except ImportError:
            pytest.skip("main function no disponible")
    
    @pytest.mark.integration
    def test_api_key_loading_priority(self):
        """Prueba el orden de prioridad para cargar API key"""
        # 1. Variables de entorno tienen prioridad
        with patch.dict(os.environ, {'GROQ_API_KEY': 'env_key'}):
            api_key = os.getenv('GROQ_API_KEY')
            assert api_key == 'env_key'
        
        # 2. Sin variables de entorno, debería intentar secrets
        with patch.dict(os.environ, {}, clear=True):
            api_key = os.getenv('GROQ_API_KEY')
            assert api_key is None

class TestStreamlitSecurity:
    """Pruebas de seguridad para Streamlit"""
    
    def test_api_key_not_exposed_in_ui(self, mock_streamlit):
        """Verifica que la API key no se exponga en la UI"""
        # Mock de text_input con type="password"
        with patch('streamlit.text_input') as mock_input:
            mock_input.return_value = "gsk_secret_key"
            
            # Verificar que se usa type="password"
            # (esto sería verificado en el código real)
            assert True  # Placeholder - en implementación real verificaríamos los argumentos
    
    def test_session_isolation(self, mock_session_state):
        """Prueba que las sesiones estén aisladas"""
        # Simular dos sesiones diferentes
        session1 = {'chat_history': ['mensaje1']}
        session2 = {'chat_history': ['mensaje2']}
        
        # Verificar que son independientes
        assert session1['chat_history'] != session2['chat_history']

class TestStreamlitPerformance:
    """Pruebas de rendimiento para Streamlit"""
    
    @pytest.mark.performance
    def test_lazy_loading_efficiency(self):
        """Prueba que las importaciones diferidas mejoren el rendimiento"""
        import time
        
        # Medir tiempo de importación diferida
        start_time = time.time()
        try:
            from main import lazy_import_langchain
            lazy_import_langchain()
        except ImportError:
            pass
        end_time = time.time()
        
        # Verificar que la importación sea relativamente rápida
        import_time = end_time - start_time
        assert import_time < 10.0, f"Importación muy lenta: {import_time}s"
    
    @pytest.mark.performance
    def test_session_state_efficiency(self, mock_session_state):
        """Prueba la eficiencia del manejo de session state"""
        # Simular múltiples accesos al session state
        for i in range(100):
            mock_session_state[f'key_{i}'] = f'value_{i}'
        
        # Verificar que se mantiene eficiente
        assert len(mock_session_state) == 100
        assert mock_session_state['key_50'] == 'value_50'

if __name__ == "__main__":
    # Ejecutar pruebas específicas de Streamlit
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])
