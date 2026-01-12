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

# Importaciones de LangChain
try:
    from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader
        from langchain.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.schema import Document
    except ImportError:
        from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.schema import Document

# Importaciones para Groq
from groq import Groq

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

class ANIFRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.groq_client = None
        self.documents_loaded = False
        
        # Configuración avanzada para conocimiento general
        self.domain_prompts = {
            "fiscal": "Como experto senior en política fiscal colombiana con conocimiento actualizado de reformas tributarias, regla fiscal, y sostenibilidad de la deuda pública",
            "monetario": "Como analista especializado del Banco de la República con conocimiento profundo de política monetaria, metas de inflación, y transmisión de política",
            "sectorial": "Como especialista en análisis sectorial de la economía colombiana con expertise en banca, industria, servicios, y sector externo",
            "internacional": "Como experto en economía internacional con enfoque en Colombia, incluyendo comercio exterior, flujos de capital, y comparaciones regionales",
            "laboral": "Como especialista en mercado laboral colombiano con conocimiento de empleo, salarios, productividad, y políticas de empleo",
            "general": "Como economista senior especializado en Colombia con visión integral de la economía nacional"
        }
        
        self.temporal_context = """
        Contexto económico actual de Colombia (2024-2026):
        - Economía post-pandemia en proceso de normalización
        - Banco de la República en ciclo de política monetaria restrictiva
        - Inflación convergiendo gradualmente hacia la meta del 3%
        - Reformas estructurales en implementación (tributaria, pensional, salud)
        - Volatilidad en precios de commodities (petróleo, carbón, café)
        - Fortalecimiento del peso colombiano vs USD
        - Elecciones presidenciales 2026 generando expectativas
        - Retos fiscales por envejecimiento poblacional
        - Transición energética en marcha
        - Digitalización acelerada del sistema financiero
        """
    
    def initialize_groq(self, api_key: str):
        """Inicializa el cliente de Groq"""
        if not api_key or api_key.strip() == "":
            st.error("❌ API key de Groq vacía o no proporcionada.")
            return False
            
        try:
            if not api_key.startswith('gsk_'):
                st.error("❌ Formato de API key inválido. Debe comenzar con 'gsk_'")
                return False
                
            self.groq_client = Groq(api_key=api_key.strip())
            st.success("✅ Cliente Groq inicializado correctamente")
            return True
            
        except Exception as e:
            st.error(f"❌ Error al inicializar Groq: {str(e)}")
            return False
    
    def load_prebuilt_vectorstore(self):
        """Carga el vectorstore pre-construido o lo crea automáticamente"""
        try:
            # Inicializar embeddings solo cuando sea necesario
            if not self.embeddings:
                with st.spinner("🧠 Inicializando modelo de embeddings..."):
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
            
            # Intentar cargar vectorstore existente primero
            if os.path.exists("vectorstore") and os.path.exists("rag_ready.flag"):
                with st.spinner("📚 Cargando base de conocimiento existente..."):
                    self.vectorstore = FAISS.load_local("vectorstore", self.embeddings, allow_dangerous_deserialization=True)
                    self.documents_loaded = True
                    return True
            else:
                # Solo crear nuevo vectorstore si no existe
                return self.initialize_rag_automatically()
                
        except Exception as e:
            st.error(f"❌ Error cargando sistema RAG: {str(e)}")
            # En caso de error, intentar crear nuevo vectorstore
            return self.initialize_rag_automatically()
    
    def initialize_rag_automatically(self):
        """Inicializa el RAG automáticamente cargando documentos desde la carpeta RAG"""
        try:
            st.info("🚀 Inicializando sistema RAG automáticamente...")
            
            rag_folder = "RAG"
            if not os.path.exists(rag_folder):
                st.error(f"❌ Carpeta {rag_folder} no encontrada")
                return False
            
            with st.spinner("📄 Cargando documentos..."):
                documents = self.load_documents_from_folder(rag_folder)
            
            if not documents:
                st.warning("⚠️ No se encontraron documentos válidos en la carpeta RAG")
                return False
            
            with st.spinner("✂️ Procesando documentos..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                splits = text_splitter.split_documents(documents)
            
            with st.spinner("🧠 Creando base de conocimiento..."):
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
                self.documents_loaded = True
            
            st.success(f"✅ Sistema RAG inicializado con {len(documents)} documentos y {len(splits)} chunks")
            return True
            
        except Exception as e:
            st.error(f"❌ Error en inicialización automática: {str(e)}")
            return False
    
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """Carga documentos desde una carpeta"""
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            st.error(f"La carpeta {folder_path} no existe")
            return documents
        
        files = list(folder.glob("*"))
        total_files = len(files)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_path in enumerate(files):
            try:
                status_text.text(f"Procesando: {file_path.name}")
                
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    loader = UnstructuredExcelLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                
                progress_bar.progress((i + 1) / total_files)
                
            except Exception as e:
                st.warning(f"Error procesando {file_path.name}: {str(e)}")
        
        status_text.text(f"✅ Procesados {len(documents)} documentos")
        progress_bar.empty()
        
        return documents
    
    def classify_economic_query(self, prompt: str) -> str:
        """Clasifica el tipo de consulta económica para aplicar prompts especializados"""
        prompt_lower = prompt.lower()
        
        fiscal_keywords = ["fiscal", "tributario", "impuesto", "déficit", "deuda", "presupuesto", "gasto público", "ingresos públicos"]
        monetary_keywords = ["monetario", "inflación", "tasa de interés", "banco república", "política monetaria", "banrep"]
        sectorial_keywords = ["bancario", "financiero", "industrial", "servicios", "agropecuario", "minero", "construcción"]
        international_keywords = ["exportaciones", "importaciones", "balanza", "tipo de cambio", "comercio exterior", "fdi"]
        laboral_keywords = ["empleo", "desempleo", "salario", "productividad", "mercado laboral"]
        
        if any(keyword in prompt_lower for keyword in fiscal_keywords):
            return "fiscal"
        elif any(keyword in prompt_lower for keyword in monetary_keywords):
            return "monetario"
        elif any(keyword in prompt_lower for keyword in sectorial_keywords):
            return "sectorial"
        elif any(keyword in prompt_lower for keyword in international_keywords):
            return "internacional"
        elif any(keyword in prompt_lower for keyword in laboral_keywords):
            return "laboral"
        else:
            return "general"
    
    def get_enhanced_system_prompt(self, query_type: str) -> str:
        """Genera prompts del sistema especializados por dominio"""
        base_context = f"""
        {self.temporal_context}
        
        {self.domain_prompts.get(query_type, self.domain_prompts["general"])}.
        
        Instrucciones avanzadas:
        - Proporciona análisis técnico preciso con datos específicos cuando sea posible
        - Incluye cifras, porcentajes, y comparaciones históricas relevantes
        - Contextualiza dentro del panorama económico actual de Colombia (2024-2026)
        - Compara con países de referencia cuando sea pertinente (Chile, Perú, México)
        - Identifica tendencias, riesgos, y oportunidades
        - Sugiere implicaciones para política económica
        - Cita fuentes implícitas de tu conocimiento (ej: "según datos del DANE", "conforme a reportes del Banco de la República")
        - Responde en español con terminología técnica apropiada
        - Estructura cada respuesta con títulos y subtítulos claros
        - Conecta explícitamente cada punto con el anterior y siguiente
        - Usa frases de transición que muestren relaciones causales
        - Proporciona ejemplos concretos y cifras específicas
        - Concluye cada sección con implicaciones para la siguiente
        - Evita listas de puntos inconexos; construye un argumento fluido
        """
        
        return base_context
    
    def create_chain_of_thought_prompt(self, prompt: str, query_type: str) -> str:
        """Crea prompts con razonamiento profundo y estructurado"""
        return f"""
        Realiza un análisis económico profundo y estructurado de la siguiente consulta:

        **PREGUNTA:** {prompt}

        **MARCO ANALÍTICO OBLIGATORIO:**

        ## 1. DIAGNÓSTICO INICIAL
        - Identifica el problema/tema central y sus dimensiones
        - Establece el alcance temporal y sectorial del análisis
        - Define las variables económicas clave involucradas

        ## 2. ANÁLISIS CAUSAL PROFUNDO
        - Examina las causas fundamentales (no solo síntomas)
        - Identifica las cadenas de causalidad económica
        - Analiza factores estructurales vs coyunturales
        - Evalúa interacciones entre variables macroeconómicas

        ## 3. CONTEXTUALIZACIÓN INTEGRAL
        - Situación actual de Colombia (2024-2026) con datos específicos
        - Comparación con ciclos económicos anteriores (últimos 10 años)
        - Benchmarking con países similares (Chile, Perú, México, Brasil)
        - Impacto de factores externos (commodities, Fed, geopolítica)

        ## 4. ANÁLISIS SECTORIAL Y DISTRIBUTIVO
        - Efectos diferenciados por sectores económicos
        - Impactos en diferentes grupos socioeconómicos
        - Implicaciones regionales dentro de Colombia
        - Conexiones con cadenas de valor globales

        ## 5. PROYECCIÓN Y ESCENARIOS
        - Tendencias esperadas a corto plazo (6-12 meses)
        - Escenarios alternativos (optimista, base, pesimista)
        - Factores de riesgo y oportunidades emergentes
        - Puntos de inflexión críticos a monitorear

        ## 6. SÍNTESIS ESTRATÉGICA
        - Conclusiones integradas que conecten todos los elementos
        - Recomendaciones de política económica específicas y viables
        - Métricas clave para seguimiento y evaluación
        - Implicaciones para diferentes stakeholders

        **INSTRUCCIONES CRÍTICAS:**
        - Cada sección debe conectar lógicamente con las demás
        - Usa datos cuantitativos específicos cuando sea posible
        - Cita fuentes implícitas (DANE, Banrep, ANIF, FMI, etc.)
        - Mantén rigor técnico pero claridad expositiva
        - Evita generalidades; sé específico y concreto
        - Construye un argumento coherente de principio a fin
        """
    
    def enhanced_general_knowledge_query(self, prompt: str) -> str:
        """Modo de conocimiento general potencializado con todas las mejoras"""
        if not self.groq_client:
            return "Error: Cliente de Groq no inicializado. Por favor, configura tu API key de Groq en la barra lateral."
        
        try:
            query_type = self.classify_economic_query(prompt)
            enhanced_prompt = self.create_chain_of_thought_prompt(prompt, query_type)
            system_prompt = self.get_enhanced_system_prompt(query_type)
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.15,
                top_p=0.85,
                max_tokens=6000,
                frequency_penalty=0.2,
                presence_penalty=0.15
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                return "Error: API key de Groq inválida o faltante. Por favor, verifica tu API key en la barra lateral."
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                return "Error: No se puede conectar a Groq. Verifica tu conexión a internet y que la API key sea válida."
            elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
                return "Error: Límite de uso de Groq alcanzado. Espera un momento antes de intentar nuevamente."
            else:
                return f"Error al consultar Groq: {error_msg}"
    
    def query_groq_hybrid(self, prompt: str, use_rag: bool = True) -> str:
        """Consulta híbrida que combina RAG con conocimiento externo del LLM"""
        if not self.groq_client:
            return "Error: Cliente de Groq no inicializado. Por favor, configura tu API key de Groq en la barra lateral."
        
        try:
            context = ""
            rag_confidence = 0
            
            if use_rag and self.documents_loaded:
                context = self.search_similar_documents(prompt, k=5)
                rag_confidence = min(len(context) / 2000, 1.0) if context.strip() else 0
            
            if not use_rag or rag_confidence < 0.2:
                return self.enhanced_general_knowledge_query(prompt)
            
            query_type = self.classify_economic_query(prompt)
            system_prompt = self.get_enhanced_system_prompt(query_type)
            
            if rag_confidence > 0.3 and context:
                user_prompt = f"""Tengo información específica de documentos ANIF y también mi conocimiento general actualizado.

                Información de documentos ANIF:
                {context}
                
                Pregunta: {prompt}
                
                Por favor, proporciona una respuesta completa que combine:
                1. La información específica de los documentos ANIF (si es relevante)
                2. Tu conocimiento general actualizado sobre el tema
                3. Análisis que conecte ambas fuentes
                
                Indica claramente qué información proviene de cada fuente."""
            else:
                user_prompt = f"""Información limitada de documentos ANIF:
                {context if context else "No hay información específica disponible en los documentos."}
                
                Pregunta: {prompt}
                
                Proporciona un análisis completo principalmente basado en tu conocimiento general actualizado sobre economía colombiana, complementando con cualquier información relevante de los documentos."""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                top_p=0.9,
                max_tokens=4000,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                return "Error: API key de Groq inválida o faltante. Por favor, verifica tu API key en la barra lateral."
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                return "Error: No se puede conectar a Groq. Verifica tu conexión a internet y que la API key sea válida."
            elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
                return "Error: Límite de uso de Groq alcanzado. Espera un momento antes de intentar nuevamente."
            else:
                return f"Error al consultar Groq: {error_msg}"
    
    def search_similar_documents(self, query: str, k: int = 3) -> str:
        """Busca documentos similares y retorna el contexto"""
        if not self.vectorstore:
            return ""
        
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            st.error(f"Error en búsqueda: {str(e)}")
            return ""

def show_agent_interface():
    """Interfaz principal del agente"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("💬 Chat con el Asistente")
    
    # Inicialización automática del RAG cuando se accede al agente
    if not st.session_state.rag_system.documents_loaded:
        with st.spinner("🚀 Inicializando sistema RAG automáticamente..."):
            success = st.session_state.rag_system.load_prebuilt_vectorstore()
            if success:
                st.success("✅ Sistema RAG inicializado correctamente")
                st.rerun()
            else:
                st.error("❌ Error al inicializar el sistema RAG")
                st.warning("⚠️ Continuando solo con conocimiento general")
                # No return - continuar con funcionalidad limitada
    
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
                <strong>🤖 Asistente:</strong><br>
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
        send_button = st.button("📤 Enviar", type="primary")
    
    with col2:
        st.markdown("**🔍 Modo de Búsqueda:**")
        search_mode = st.radio(
            "Selecciona el modo:",
            ["🔄 Híbrido (RAG + Conocimiento General)", "📚 Solo RAG", "🌐 Solo Conocimiento General"],
            index=0,
            key="search_mode"
        )
    
    # Procesar pregunta manual
    if send_button and user_question:
        if not st.session_state.rag_system.groq_client:
            st.error("⚠️ Por favor, configura tu API key de Groq primero")
            return
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner("🤖 Generando respuesta..."):
            try:
                if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
                    response = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=True)
                elif search_mode == "📚 Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(user_question) if st.session_state.rag_system.documents_loaded else ""
                    response = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=bool(context))
                else:  # Solo Conocimiento General
                    response = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=False)
            except Exception as e:
                response = f"Error al consultar Groq: {str(e)}"
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
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
                        st.error("⚠️ Por favor, configura tu API key de Groq primero")
                        return
                    
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    
                    with st.spinner("🤖 Generando respuesta..."):
                        try:
                            response = st.session_state.rag_system.query_groq_hybrid(question, use_rag=True)
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
    
    st.header("📊 Generación Automática de Informes")
    
    # Inicialización automática del RAG cuando se accede a informes
    if not st.session_state.rag_system.documents_loaded:
        with st.spinner("🚀 Inicializando sistema RAG automáticamente..."):
            success = st.session_state.rag_system.load_prebuilt_vectorstore()
            if success:
                st.success("✅ Sistema RAG inicializado correctamente")
                st.rerun()
            else:
                st.error("❌ Error al inicializar el sistema RAG")
                st.warning("⚠️ Continuando solo con conocimiento general")
    
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
                <strong>🤖 Asistente:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
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
            ["🔄 Híbrido (RAG + Conocimiento General)", "📚 Solo RAG", "🌐 Solo Conocimiento General"],
            index=0,
            key="report_search_mode"
        )
    
    # Procesar solicitud de informe
    if send_button and user_question:
        if not st.session_state.rag_system.groq_client:
            st.error("⚠️ Por favor, configura tu API key de Groq primero")
            return
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        with st.spinner("📊 Generando informe..."):
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
                    response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
                elif search_mode == "📚 Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(enhanced_question) if st.session_state.rag_system.documents_loaded else ""
                    response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=bool(context))
                else:  # Solo Conocimiento General
                    response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=False)
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
        st.header("📋 Tipos de informes disponibles")
        
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
                        st.error("⚠️ Por favor, configura tu API key de Groq primero")
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
                            response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
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
    
    st.header("🏛️ Herramientas Especializadas ANIF")
    
    # Inicialización automática del RAG cuando se accede a herramientas ANIF
    if not st.session_state.rag_system.documents_loaded:
        with st.spinner("🚀 Inicializando sistema RAG automáticamente..."):
            success = st.session_state.rag_system.load_prebuilt_vectorstore()
            if success:
                st.success("✅ Sistema RAG inicializado correctamente")
                st.rerun()
            else:
                st.error("❌ Error al inicializar el sistema RAG")
                st.warning("⚠️ Continuando solo con conocimiento general")
    
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
                <strong>🤖 Asistente:</strong><br>
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
        send_button = st.button("🔍 Analizar", type="primary")
    
    with col2:
        st.markdown("**🔍 Modo de Búsqueda:**")
        search_mode = st.radio(
            "Selecciona el modo:",
            ["🔄 Híbrido (RAG + Conocimiento General)", "📚 Solo RAG", "🌐 Solo Conocimiento General"],
            index=0,
            key="anif_search_mode"
        )
    
    # Procesar consulta especializada
    if send_button and user_question:
        if not st.session_state.rag_system.groq_client:
            st.error("⚠️ Por favor, configura tu API key de Groq primero")
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
                    response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
                elif search_mode == "📚 Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(enhanced_question) if st.session_state.rag_system.documents_loaded else ""
                    response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=bool(context))
                else:  # Solo Conocimiento General
                    response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=False)
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
                        st.error("⚠️ Por favor, configura tu API key de Groq primero")
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
                    
                    with st.spinner("🏛️ Procesando análisis especializado..."):
                        try:
                            response = st.session_state.rag_system.query_groq_hybrid(enhanced_question, use_rag=True)
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
    
    # Inicializar Groq automáticamente usando secretos de Streamlit Cloud
    try:
        # Intentar obtener la API key desde los secretos de Streamlit Cloud
        groq_api_key = st.secrets.get("GROQ_API_KEY", None)
        
        if groq_api_key:
            # Inicializar Groq automáticamente si la key está disponible
            if not st.session_state.rag_system.groq_client:
                st.session_state.rag_system.initialize_groq(groq_api_key)
        else:
            # Fallback: mostrar input manual solo si no hay secreto configurado
            with st.sidebar:
                st.header("⚙️ Configuración")
                st.warning("⚠️ API key no encontrada en secretos de Streamlit Cloud")
                
                groq_api_key = st.text_input(
                    "🔑 Groq API Key (Fallback)",
                    type="password",
                    help="Configura GROQ_API_KEY en los secretos de Streamlit Cloud"
                )
                
                if groq_api_key:
                    if st.session_state.rag_system.initialize_groq(groq_api_key):
                        st.success("✅ Groq conectado")
                    else:
                        st.error("❌ Error conectando Groq")
                        
    except Exception as e:
        # En desarrollo local, mostrar input manual
        with st.sidebar:
            st.header("⚙️ Configuración")
            st.info("🏠 Modo desarrollo local")
            
            groq_api_key = st.text_input(
                "🔑 Groq API Key",
                type="password",
                help="Obtén tu API key gratuita en https://console.groq.com"
            )
            
            if groq_api_key:
                if st.session_state.rag_system.initialize_groq(groq_api_key):
                    st.success("✅ Groq conectado")
                else:
                    st.error("❌ Error conectando Groq")
    
    # Menú de navegación
    menu_options = ["🤖 Agente", "📊 Generación de Informes", "🏛️ Herramientas ANIF"]
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
