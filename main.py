# -*- coding: utf-8 -*-
"""
ANIF - Asistente de Investigación Económica
Sistema RAG con IA para análisis de documentos económicos
"""

import streamlit as st
import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
import plotly.express as px
from rag_core import ANIFRAGSystem

# Importaciones diferidas - solo cuando se necesiten
# Esto evita demoras en el startup de Streamlit Cloud

def lazy_import_langchain():
    """Importa LangChain solo cuando se necesita"""
    try:
        from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        return PyPDFLoader, UnstructuredExcelLoader, FAISS, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter, Document
    except ImportError:
        try:
            from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader
            from langchain.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain.schema import Document
            return PyPDFLoader, UnstructuredExcelLoader, FAISS, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter, Document
        except ImportError:
            from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain.schema import Document
            return PyPDFLoader, UnstructuredExcelLoader, FAISS, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter, Document

def lazy_import_groq():
    """Importa Groq solo cuando se necesita"""
    from groq import Groq
    return Groq

# Configuración de la página
st.set_page_config(
    page_title="ANIF - Asistente de Investigación Económica",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left-color: #4ecdc4;
    }
</style>
""", unsafe_allow_html=True)

def show_agent_interface():
    """Interfaz principal del agente"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("Chat con el Asistente")
    
    # Inicialización automática del RAG cuando se accede al agente
    if not st.session_state.rag_system.documents_loaded:
        with st.spinner("Inicializando sistema RAG automáticamente..."):
            # En main.py envolvemos la llamada lógica con spinner
            success = st.session_state.rag_system.load_prebuilt_vectorstore() 
            if success:
                st.success("Sistema RAG inicializado correctamente")
                st.rerun()
            else:
                # Mostrar información específica del error para debugging
                if not os.path.exists("vectorstore"):
                    st.info(" Vectorstore no encontrado - esto es normal en Streamlit Cloud")
                if not os.path.exists("RAG"):
                    st.info(" Carpeta RAG no encontrada - documentos no disponibles")
                st.warning("Continuando solo con conocimiento general del modelo Groq")
                # No return - continuar con funcionalidad limitada
    
    # Mostrar historial de chat
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong> Usuario:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong> Asistente:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Input para nueva pregunta
    user_question = st.text_input(
        "Haz tu pregunta sobre economía colombiana:",
        placeholder="Ej: ¿Cuál es la perspectiva fiscal para 2026 según los documentos?",
        key="user_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        send_button = st.button(" Enviar", type="primary")
    
    with col2:
        st.markdown("**🔍 Modo de Búsqueda:**")
        search_mode = st.radio(
            "Selecciona el modo:",
            ["🔄 Híbrido (RAG + Conocimiento General)", " Solo RAG", " Solo Conocimiento General"],
            index=0,
            key="search_mode"
        )
    
    # Procesar pregunta manual
    if send_button and user_question:
        if not st.session_state.rag_system.groq_client:
            st.error(" Por favor, configura tu API key de Groq primero")
            return
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner(" Generando respuesta..."):
            try:
                if search_mode == " Híbrido (RAG + Conocimiento General)":
                    result = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=True)
                elif search_mode == " Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(user_question) if st.session_state.rag_system.documents_loaded else ""
                    result = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=bool(context))
                else:  # Solo Conocimiento General
                    result = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=False)
                
                # Manejo robusto de la respuesta (dict o str)
                if isinstance(result, dict):
                    response = result["answer"]
                    sources = result.get("sources", [])
                else:
                    response = result
                    sources = []
                    
            except Exception as e:
                response = f"Error al consultar Groq: {str(e)}"
                sources = []
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources  # Guardamos fuentes en el historial
        })
        
        st.rerun()
    
    # Ejemplos de preguntas
    if not st.session_state.chat_history:
        st.markdown("---")
        st.header("💡 Preguntas de ejemplo")
        
        example_questions = [
            "¿Cuáles son las perspectivas fiscales para Colombia en 2026?",
            "¿Qué dice el último reporte sobre el PIB tendencial?",
            "¿Cuál es el análisis del presupuesto general de la nación 2026?",
            "¿Qué impacto fiscal tiene el aumento del salario mínimo 2026?",
            "¿Cuáles son las elasticidades económicas más recientes?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(f"❓ {question}", key=f"example_{i}"):
                    if not st.session_state.rag_system.groq_client:
                        st.error(" Por favor, configura tu API key de Groq primero")
                        return
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    
                    with st.spinner("🤖 Generando respuesta..."):
                        try:
                            result = st.session_state.rag_system.query_groq_hybrid(question, use_rag=True)
                            response = result["answer"] if isinstance(result, dict) else result
                        except Exception as e:
                            response = f"Error al consultar Groq: {str(e)}"
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()

def show_report_generation_interface():
    """Interfaz para generación automática de informes"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.header(" Generación Automática de Informes")
    
    # Inicialización automática del RAG cuando se accede a informes
    if not st.session_state.rag_system.documents_loaded:
        with st.spinner("Inicializando sistema RAG automáticamente..."):
            success = st.session_state.rag_system.load_prebuilt_vectorstore()
            if success:
                st.success("Sistema RAG inicializado correctamente")
                st.rerun()
            else:
                # Mostrar información específica del error para debugging
                if not os.path.exists("vectorstore"):
                    st.info(" Vectorstore no encontrado - esto es normal en Streamlit Cloud")
                if not os.path.exists("RAG"):
                    st.info(" Carpeta RAG no encontrada - documentos no disponibles")
                st.warning("Continuando solo con conocimiento general del modelo Groq")
    
    # Mostrar historial de chat
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 Usuario:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong> Asistente:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Panel de Evidencia (Requisito Técnico)
            if "sources" in message and message["sources"]:
                with st.expander("📚 Ver Evidencia / Fuentes"):
                    for idx, source in enumerate(message["sources"]):
                        st.markdown(f"**Fuente {idx+1}:** `{source.get('source', 'Desconocido')}` (Score: {source.get('score', 0):.2f})")
                        st.caption(f"...{source.get('content', '')}...")
                        st.divider()
    
    # Input para nueva pregunta
    user_question = st.text_input(
        "Solicita un informe económico específico:",
        placeholder="Ej: Genera un informe sobre las perspectivas fiscales de Colombia para 2026",
        key="report_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        send_button = st.button("📤 Generar", type="primary")
    
    with col2:
        st.markdown("**🔍 Modo de Búsqueda:**")
        search_mode = st.radio(
            "Selecciona el modo:",
            [" Híbrido (RAG + Conocimiento General)", " Solo RAG", " Solo Conocimiento General"],
            index=0,
            key="report_search_mode"
        )
    
    # Procesar solicitud de informe
    if send_button and user_question:
        if not st.session_state.rag_system.groq_client:
            st.error(" Por favor, configura tu API key de Groq primero")
            return
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner(" Generando informe..."):
            try:
                # Agregar contexto específico para informes
                enhanced_question = f"""Como experto analista económico de ANIF, genera un informe profesional y detallado sobre: {user_question}

                El informe debe incluir:
                1. Resumen ejecutivo
                2. Análisis detallado con datos específicos
                3. Tendencias y proyecciones
                4. Recomendaciones de política
                5. Conclusiones y próximos pasos
                
                Usa un formato profesional con títulos, subtítulos y estructura clara."""
                
                if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
                    result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
                    response = result["answer"] if isinstance(result, dict) else result
                elif search_mode == "📚 Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(enhanced_question) if st.session_state.rag_system.documents_loaded else ""
                    result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=bool(context))
                    response = result["answer"] if isinstance(result, dict) else result
                else:  # Solo Conocimiento General
                    result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=False)
                    response = result["answer"] if isinstance(result, dict) else result
            except Exception as e:
                response = f"Error al generar informe: {str(e)}"
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        st.rerun()
    
    # Ejemplos de informes
    if not st.session_state.chat_history:
        st.markdown("---")
        st.header(" Tipos de informes disponibles")
        
        report_examples = [
            "Informe de perspectivas fiscales Colombia 2026",
            "Análisis sectorial del sistema financiero colombiano",
            "Reporte de impacto económico del salario mínimo 2026",
            "Evaluación de la política monetaria del Banco de la República",
            "Informe de competitividad económica regional"
        ]
        
        cols = st.columns(2)
        for i, report in enumerate(report_examples):
            with cols[i % 2]:
                if st.button(f"📊 {report}", key=f"report_example_{i}"):
                    if not st.session_state.rag_system.groq_client:
                        st.error(" Por favor, configura tu API key de Groq primero")
                        return
                    
                    enhanced_question = f"""Como experto analista económico de ANIF, genera un informe profesional y detallado sobre: {report}

                    El informe debe incluir:
                    1. Resumen ejecutivo
                    2. Análisis detallado con datos específicos
                    3. Tendencias y proyecciones
                    4. Recomendaciones de política
                    5. Conclusiones y próximos pasos
                    
                    Usa un formato profesional con títulos, subtítulos y estructura clara."""
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": report
                    })
                    
                    with st.spinner("📊 Generando informe..."):
                        try:
                            result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
                            response = result["answer"] if isinstance(result, dict) else result
                        except Exception as e:
                            response = f"Error al generar informe: {str(e)}"
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()

def show_anif_tools_interface():
    """Interfaz para herramientas especializadas de ANIF"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.header(" Herramientas Especializadas ANIF")
    
    # Inicialización automática del RAG cuando se accede a herramientas ANIF
    if not st.session_state.rag_system.documents_loaded:
        with st.spinner("Inicializando sistema RAG automáticamente..."):
            success = st.session_state.rag_system.load_prebuilt_vectorstore()
            if success:
                st.success("Sistema RAG inicializado correctamente")
                st.rerun()
            else:
                # Mostrar información específica del error para debugging
                if not os.path.exists("vectorstore"):
                    st.info(" Vectorstore no encontrado - esto es normal en Streamlit Cloud")
                if not os.path.exists("RAG"):
                    st.info("Carpeta RAG no encontrada - documentos no disponibles")
                st.warning("Continuando solo con conocimiento general del modelo Groq")
    
    # Mostrar historial de chat
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong> Usuario:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong> Asistente:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Input para nueva consulta especializada
    user_question = st.text_input(
        "Consulta especializada ANIF:",
        placeholder="Ej: Análisis de elasticidades económicas según metodología ANIF",
        key="anif_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        send_button = st.button(" Analizar", type="primary")
    
    with col2:
        st.markdown("**🔍 Modo de Búsqueda:**")
        search_mode = st.radio(
            "Selecciona el modo:",
            ["🔄 Híbrido (RAG + Conocimiento General)", " Solo RAG", " Solo Conocimiento General"],
            index=0,
            key="anif_search_mode"
        )
    
    # Procesar consulta especializada
    if send_button and user_question:
        if not st.session_state.rag_system.groq_client:
            st.error(" Por favor, configura tu API key de Groq primero")
            return
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner("🏛️ Procesando análisis especializado..."):
            try:
                # Agregar contexto específico para herramientas ANIF
                enhanced_question = f"""Como investigador senior de ANIF (Asociación Nacional de Instituciones Financieras), proporciona un análisis técnico especializado sobre: {user_question}

                El análisis debe incluir:
                1. Marco metodológico ANIF aplicable
                2. Datos y estadísticas específicas del sector financiero colombiano
                3. Análisis comparativo con estándares internacionales
                4. Implicaciones para el sistema financiero y la economía
                5. Recomendaciones técnicas especializadas
                
                Usa terminología técnica apropiada y referencias a estudios ANIF cuando sea relevante."""
                
                if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
                    result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
                    response = result["answer"] if isinstance(result, dict) else result
                elif search_mode == "📚 Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(enhanced_question) if st.session_state.rag_system.documents_loaded else ""
                    result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=bool(context))
                    response = result["answer"] if isinstance(result, dict) else result
                else:  # Solo Conocimiento General
                    result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=False)
                    response = result["answer"] if isinstance(result, dict) else result
            except Exception as e:
                response = f"Error al procesar análisis: {str(e)}"
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        st.rerun()
    
    # Herramientas especializadas disponibles
    if not st.session_state.chat_history:
        st.markdown("---")
        st.header("🔧 Herramientas Especializadas Disponibles")
        
        anif_tools = [
            "Análisis de elasticidades económicas sectoriales",
            "Evaluación de riesgo sistémico del sector financiero",
            "Cálculo de indicadores de profundización financiera",
            "Análisis de transmisión de política monetaria",
            "Evaluación de impacto regulatorio en el sector financiero"
        ]
        
        cols = st.columns(2)
        for i, tool in enumerate(anif_tools):
            with cols[i % 2]:
                if st.button(f"🔧 {tool}", key=f"anif_tool_{i}"):
                    if not st.session_state.rag_system.groq_client:
                        st.error(" Por favor, configura tu API key de Groq primero")
                        return
                    
                    enhanced_question = f"""Como investigador senior de ANIF (Asociación Nacional de Instituciones Financieras), proporciona un análisis técnico especializado sobre: {tool}

                    El análisis debe incluir:
                    1. Marco metodológico ANIF aplicable
                    2. Datos y estadísticas específicas del sector financiero colombiano
                    3. Análisis comparativo con estándares internacionales
                    4. Implicaciones para el sistema financiero y la economía
                    5. Recomendaciones técnicas especializadas
                    
                    Usa terminología técnica apropiada y referencias a estudios ANIF cuando sea relevante."""
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": tool
                    })
                    
                    with st.spinner(" Procesando análisis especializado..."):
                        try:
                            result = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
                            response = result["answer"] if isinstance(result, dict) else result
                        except Exception as e:
                            response = f"Error al procesar análisis: {str(e)}"
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ ANIF - Asistente de Investigación Económica</h1>
        <p>Sistema RAG con IA para análisis de documentos económicos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar el sistema RAG primero
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = ANIFRAGSystem()
    
    # Inicializar Groq automáticamente desde múltiples fuentes
    if not st.session_state.rag_system.groq_client:
        groq_api_key = None
        
        # 1. Intentar desde variables de entorno (Docker/producción)
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        # 2. Si no está en env, intentar desde secretos de Streamlit Cloud
        if not groq_api_key:
            try:
                groq_api_key = st.secrets.get("GROQ_API_KEY", None)
            except:
                pass
        
        # 3. Si encontramos la API key, inicializar automáticamente
        if groq_api_key:
            if st.session_state.rag_system.initialize_groq(groq_api_key):
                st.sidebar.success(" API Key cargada automáticamente")
            else:
                st.sidebar.error(" Error con API Key automática")
        else:
            # 4. Solo mostrar input manual si no hay API key en ningún lado
            with st.sidebar:
                st.header("⚙️ Configuración")
                st.warning(" API key no encontrada en variables de entorno ni secretos")
                st.info(" Configura GROQ_API_KEY como variable de entorno o secreto")
                
                groq_api_key = st.text_input(
                    " Groq API Key (Manual)",
                    type="password",
                    help="Solo necesario si no está configurada como variable de entorno"
                )
                
                if groq_api_key:
                    if st.session_state.rag_system.initialize_groq(groq_api_key):
                        st.success(" Groq conectado")
                    else:
                        st.error(" Error conectando Groq")
    
    # Menú de navegación
    menu_options = [" Agente", " Generación de Informes", " Herramientas ANIF"]
    selected_menu = st.selectbox("Selecciona una funcionalidad:", menu_options, key="main_menu")

    # Inicialización lazy del RAG - solo cuando se necesite
    # No inicializar automáticamente para evitar timeouts en Streamlit Cloud
    
    # Inicializar chat_history globalmente
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Mostrar contenido según el menú seleccionado
    if selected_menu == "🤖 Agente":
        show_agent_interface()
    elif selected_menu == "📊 Generación de Informes":
        show_report_generation_interface()
    elif selected_menu == "🏛️ Herramientas ANIF":
        show_anif_tools_interface()

if __name__ == "__main__":
    main()

