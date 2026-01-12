import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import time
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importaciones para RAG
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
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
    except ImportError:
        from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e7bb8 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2e7bb8;
        background-color: #f8f9fa;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #1976d2;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left-color: #388e3c;
    }
    .sidebar-content {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ANIFRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.groq_client = None
        self.documents_loaded = False
        self.system_ready = False
        
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
        
    def initialize_embeddings(self):
        """Inicializa los embeddings usando HuggingFace"""
        if self.embeddings is None:
            with st.spinner("Inicializando sistema de embeddings..."):
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
        return self.embeddings
    
    def initialize_groq(self, api_key: str):
        """Inicializa el cliente de Groq"""
        if not api_key or api_key.strip() == "":
            st.error("❌ API key de Groq vacía o no proporcionada.")
            return False
            
        try:
            # Validar formato del API key
            if not api_key.startswith('gsk_'):
                st.error("❌ Formato de API key inválido. Debe comenzar con 'gsk_'")
                return False
                
            self.groq_client = Groq(api_key=api_key.strip())
            
            # Test básico sin hacer llamada a la API (para evitar errores de red)
            st.success("✅ Cliente Groq inicializado correctamente")
            return True
            
        except Exception as e:
            error_msg = str(e)
            st.error(f"❌ Error detallado al inicializar Groq: {error_msg}")
            return False
    
    def load_documents_from_folder(self, folder_path: str) -> List[Document]:
        """Carga documentos desde una carpeta"""
        documents = []
        folder = Path(folder_path)
        
        if not folder.exists():
            st.error(f"La carpeta {folder_path} no existe")
            return documents
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        files = list(folder.glob("*"))
        total_files = len(files)
        
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
    
    def load_prebuilt_vectorstore(self):
        """Carga el vectorstore pre-construido o lo crea automáticamente"""
        try:
            # Inicializar embeddings
            if not self.embeddings:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            
            # Intentar cargar vectorstore pre-construido
            if os.path.exists("vectorstore") and os.path.exists("rag_ready.flag"):
                self.vectorstore = FAISS.load_local("vectorstore", self.embeddings, allow_dangerous_deserialization=True)
                self.documents_loaded = True
                st.success("✅ Sistema RAG cargado exitosamente")
                return True
            else:
                # Si no existe, crear automáticamente (especialmente para Streamlit Cloud)
                st.info("🔄 Vectorstore no encontrado. Inicializando RAG automáticamente...")
                return self.initialize_rag_automatically()
                
        except Exception as e:
            st.error(f"❌ Error cargando sistema RAG: {str(e)}")
            # Intentar inicialización automática como fallback
            st.info("🔄 Intentando inicialización automática...")
            return self.initialize_rag_automatically()
    
    def initialize_rag_automatically(self):
        """Inicializa el RAG automáticamente cargando documentos desde la carpeta RAG"""
        try:
            st.info("🚀 Inicializando sistema RAG automáticamente...")
            
            # Verificar si existe la carpeta RAG
            rag_folder = "RAG"
            if not os.path.exists(rag_folder):
                st.error(f"❌ Carpeta {rag_folder} no encontrada")
                return False
            
            # Cargar documentos
            with st.spinner("📄 Cargando documentos..."):
                documents = self.load_documents_from_folder(rag_folder)
            
            if not documents:
                st.warning("⚠️ No se encontraron documentos válidos en la carpeta RAG")
                return False
            
            # Dividir documentos en chunks
            with st.spinner("✂️ Procesando documentos..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                splits = text_splitter.split_documents(documents)
            
            # Crear vectorstore
            with st.spinner("🧠 Creando base de conocimiento..."):
                self.vectorstore = FAISS.from_documents(splits, self.embeddings)
                self.documents_loaded = True
            
            st.success(f"✅ Sistema RAG inicializado con {len(documents)} documentos y {len(splits)} chunks")
            return True
            
        except Exception as e:
            st.error(f"❌ Error en inicialización automática: {str(e)}")
            return False
    
    def classify_economic_query(self, prompt: str) -> str:
        """Clasifica el tipo de consulta económica para aplicar prompts especializados"""
        prompt_lower = prompt.lower()
        
        # Palabras clave por dominio
        fiscal_keywords = ["fiscal", "tributario", "impuesto", "déficit", "deuda", "presupuesto", "gasto público", "ingresos públicos"]
        monetary_keywords = ["monetario", "inflación", "tasa de interés", "banco república", "política monetaria", "banrep"]
        sectorial_keywords = ["bancario", "financiero", "industrial", "servicios", "agropecuario", "minero", "construcción"]
        international_keywords = ["exportaciones", "importaciones", "balanza", "tipo de cambio", "comercio exterior", "fdi"]
        laboral_keywords = ["empleo", "desempleo", "salario", "productividad", "mercado laboral"]
        
        # Clasificación por coincidencias
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
        """
        
        return base_context
    
    def create_chain_of_thought_prompt(self, prompt: str, query_type: str) -> str:
        """Crea prompts con razonamiento paso a paso"""
        return f"""
        Analiza esta consulta económica paso a paso:
        
        1. **Contexto de la pregunta**: ¿Qué información específica se solicita?
        2. **Datos relevantes**: ¿Qué indicadores, cifras o tendencias son pertinentes?
        3. **Análisis técnico**: ¿Cuáles son las variables clave y sus interrelaciones?
        4. **Contexto temporal**: ¿Cómo se relaciona con la situación económica actual (2024-2026)?
        5. **Comparación**: ¿Cómo se compara con períodos anteriores o países similares?
        6. **Síntesis**: Respuesta fundamentada con conclusiones y recomendaciones
        
        Pregunta: {prompt}
        
        Proporciona un análisis completo siguiendo esta estructura.
        """
    
    def enhanced_general_knowledge_query(self, prompt: str) -> str:
        """Modo de conocimiento general potencializado con todas las mejoras"""
        if not self.groq_client:
            return "Error: Cliente de Groq no inicializado. Por favor, configura tu API key de Groq en la barra lateral."
        
        try:
            # 1. Clasificar tipo de consulta
            query_type = self.classify_economic_query(prompt)
            
            # 2. Crear prompt especializado con chain-of-thought
            enhanced_prompt = self.create_chain_of_thought_prompt(prompt, query_type)
            
            # 3. Obtener prompt del sistema especializado
            system_prompt = self.get_enhanced_system_prompt(query_type)
            
            # 4. Usar parámetros optimizados según el tipo de consulta
            if query_type in ["fiscal", "monetario"]:
                # Consultas técnicas requieren mayor precisión
                temperature = 0.1
                top_p = 0.8
            else:
                # Consultas generales pueden ser más creativas
                temperature = 0.2
                top_p = 0.9
            
            # 5. Llamada optimizada a Groq
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                model="llama-3.1-70b-versatile",  # Modelo más potente
                temperature=temperature,
                top_p=top_p,
                max_tokens=4000,  # Respuestas más completas
                frequency_penalty=0.1,  # Evita repeticiones
                presence_penalty=0.1   # Fomenta diversidad
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            # Fallback al modelo estándar si el avanzado falla
            try:
                return self.query_groq_hybrid_fallback(prompt)
            except:
                if "api_key" in error_msg.lower():
                    return "Error: API key de Groq inválida o faltante. Por favor, verifica tu API key en la barra lateral."
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    return "Error: No se puede conectar a Groq. Verifica tu conexión a internet y que la API key sea válida."
                elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    return "Error: Límite de uso de Groq alcanzado. Espera un momento antes de intentar nuevamente."
                else:
                    return f"Error al consultar Groq: {error_msg}"
    
    def query_groq_hybrid_fallback(self, prompt: str) -> str:
        """Método de respaldo con el modelo estándar"""
        query_type = self.classify_economic_query(prompt)
        system_prompt = self.get_enhanced_system_prompt(query_type)
        
        response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
    
    def query_groq_hybrid(self, prompt: str, use_rag: bool = True) -> str:
        """Consulta híbrida que combina RAG con conocimiento externo del LLM"""
        if not self.groq_client:
            return "Error: Cliente de Groq no inicializado. Por favor, configura tu API key de Groq en la barra lateral."
        
        try:
            context = ""
            rag_confidence = 0
            
            # Intentar búsqueda RAG primero si está habilitada
            if use_rag and self.documents_loaded:
                context = self.search_similar_documents(prompt, k=5)
                # Evaluar confianza del RAG basado en la longitud y relevancia del contexto
                rag_confidence = min(len(context) / 2000, 1.0) if context.strip() else 0
            
            # Si no hay RAG o es modo solo conocimiento general, usar el método potencializado
            if not use_rag or rag_confidence < 0.2:
                return self.enhanced_general_knowledge_query(prompt)
            
            # Construir el prompt del sistema para modo híbrido
            query_type = self.classify_economic_query(prompt)
            system_prompt = self.get_enhanced_system_prompt(query_type)
            
            # Construir el prompt del usuario según la confianza del RAG
            if rag_confidence > 0.3 and context:
                # RAG tiene información relevante
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
                # RAG tiene poca información relevante
                user_prompt = f"""Información limitada de documentos ANIF:
                {context if context else "No hay información específica disponible en los documentos."}
                
                Pregunta: {prompt}
                
                Proporciona un análisis completo principalmente basado en tu conocimiento general actualizado sobre economía colombiana, complementando con cualquier información relevante de los documentos."""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.1-70b-versatile",
                temperature=0.2,
                top_p=0.9,
                max_tokens=4000,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            # Fallback al modelo estándar
            try:
                return self.query_groq_hybrid_fallback(prompt)
            except:
                if "api_key" in error_msg.lower():
                    return "Error: API key de Groq inválida o faltante. Por favor, verifica tu API key en la barra lateral."
                elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                    return "Error: No se puede conectar a Groq. Verifica tu conexión a internet y que la API key sea válida."
                elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    return "Error: Límite de uso de Groq alcanzado. Espera un momento antes de intentar nuevamente."
                else:
                    return f"Error al consultar Groq: {error_msg}"
    
    def query_groq(self, prompt: str, context: str = "") -> str:
        """Método legacy para compatibilidad - redirige al método híbrido"""
        return self.query_groq_hybrid(prompt, use_rag=bool(context))
    
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

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ ANIF - Asistente de Investigación Económica</h1>
        <p>Sistema RAG con IA para análisis de documentos económicos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Menú de navegación
    menu_options = ["🤖 Agente", "📊 Generación de Informes", "🏛️ Herramientas ANIF"]
    selected_menu = st.selectbox("Selecciona una funcionalidad:", menu_options, key="main_menu")
    
    # Inicializar el sistema RAG automáticamente
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = ANIFRAGSystem()
        
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.header("⚙️ Configuración")
        
        # API Key de Groq
        env_api_key = os.getenv("GROQ_API_KEY")
        
        if env_api_key:
            st.info("🔑 API Key cargada desde Streamlit Secrets")
            groq_api_key = env_api_key
        else:
            groq_api_key = st.text_input(
                "🔑 Groq API Key", 
                type="password",
                help="Obtén tu API key gratuita en https://console.groq.com/"
            )
        
        if groq_api_key:
            if st.session_state.rag_system.initialize_groq(groq_api_key):
                st.success("✅ Groq conectado")
        
        st.markdown("---")
        
        # Sistema RAG - Solo mostrar estado
        st.header("📚 Sistema RAG")
        
        # Estado del sistema
        if st.session_state.rag_system.documents_loaded:
            st.success("✅ Sistema RAG operativo")
        else:
            st.info("🔄 Sistema RAG inicializándose automáticamente...")
        
        if st.session_state.rag_system.groq_client:
            st.success("✅ Groq conectado")
        else:
            st.warning("⚠️ Groq no conectado")
        
        st.markdown("---")
        
        # Información del sistema
        st.header("ℹ️ Información")
        st.info("""
        **Documentos disponibles:**
        - Reportes económicos ANIF
        - Documentos técnicos
        - Análisis fiscales
        - Seguimientos económicos
        - Datos históricos
        """)
        
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Área principal de chat
    st.header("💬 Chat con el Asistente")
    
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
        # Opciones del sistema híbrido
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
        
        # Agregar pregunta al historial
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })
        
        # Determinar modo de búsqueda y generar respuesta
        with st.spinner("🤖 Generando respuesta..."):
            try:
                if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
                    response = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=True)
                elif search_mode == "📚 Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(user_question) if st.session_state.rag_system.documents_loaded else ""
                    response = st.session_state.rag_system.query_groq(user_question, context)
                else:  # Solo Conocimiento General
                    response = st.session_state.rag_system.query_groq_hybrid(user_question, use_rag=False)
            except Exception as e:
                response = f"Error al consultar Groq: {str(e)}"
        
        # Agregar respuesta al historial
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Recargar la página para mostrar la nueva conversación
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
                    # Procesar pregunta de ejemplo directamente
                    if not st.session_state.rag_system.groq_client:
                        st.error("⚠️ Por favor, configura tu API key de Groq primero")
                        return
                    
                    # Agregar pregunta al historial
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    
                    # Generar respuesta usando sistema híbrido por defecto
                    with st.spinner("🤖 Generando respuesta..."):
                        try:
                            response = st.session_state.rag_system.query_groq_hybrid(question, use_rag=True)
                        except Exception as e:
                            response = f"Error al consultar Groq: {str(e)}"
                    
                    # Agregar respuesta al historial
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()

def show_report_generation_interface():
    """Interfaz para generación automática de informes"""
    st.header("📊 Generación Automática de Informes")
    
    # Verificar que el sistema RAG esté cargado
    if not st.session_state.rag_system.documents_loaded:
        st.error("❌ Sistema RAG no cargado. El sistema debe estar operativo para generar informes.")
        return
    
    if not st.session_state.rag_system.groq_client:
        st.error("❌ Groq no conectado. Configura tu API key primero.")
        return
    
    # Sidebar para configuración de informes
    with st.sidebar:
        st.header("⚙️ Configuración de Informes")
        
        # Tipo de informe
        report_types = [
            "📈 Informe de Coyuntura Económica",
            "💰 Análisis Fiscal",
            "🏦 Reporte Sectorial Bancario",
            "📊 Resumen Ejecutivo",
            "🔍 Análisis Comparativo",
            "📋 Informe Personalizado"
        ]
        
        selected_report = st.selectbox("Tipo de Informe:", report_types)
        
        # Período de análisis
        st.subheader("📅 Período de Análisis")
        period_options = [
            "Último mes",
            "Último trimestre", 
            "Último semestre",
            "Último año",
            "Personalizado"
        ]
        selected_period = st.selectbox("Período:", period_options)
        
        # Opciones adicionales
        st.subheader("🎯 Opciones")
        include_charts = st.checkbox("Incluir gráficos", value=True)
        include_recommendations = st.checkbox("Incluir recomendaciones", value=True)
        detailed_analysis = st.checkbox("Análisis detallado", value=False)
        
        # Formato de salida
        output_format = st.selectbox("Formato de salida:", ["📄 Markdown", "📋 PDF", "📊 PowerPoint"])
        
        # Modo de búsqueda
        st.subheader("🔍 Modo de Búsqueda")
        report_search_mode = st.radio(
            "Selecciona el modo:",
            ["🔄 Híbrido (RAG + Conocimiento General)", "📚 Solo RAG", "🌐 Solo Conocimiento General"],
            index=0,
            key="report_search_mode"
        )
    
    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Generación de Informe")
        
        if selected_report == "📋 Informe Personalizado":
            custom_topic = st.text_area(
                "Describe el tema específico del informe:",
                placeholder="Ej: Impacto del aumento del salario mínimo en la inflación y el empleo durante 2025-2026"
            )
        
        # Botón para generar informe
        if st.button("🚀 Generar Informe", type="primary", use_container_width=True):
            generate_report(selected_report, selected_period, include_charts, 
                          include_recommendations, detailed_analysis, 
                          custom_topic if selected_report == "📋 Informe Personalizado" else None,
                          report_search_mode)
    
    with col2:
        st.subheader("ℹ️ Información")
        st.info("""
        **Tipos de Informe:**
        
        📈 **Coyuntura Económica**: Análisis de indicadores macroeconómicos actuales
        
        💰 **Análisis Fiscal**: Evaluación del presupuesto y política fiscal
        
        🏦 **Sectorial Bancario**: Estado del sistema financiero
        
        📊 **Resumen Ejecutivo**: Síntesis de múltiples documentos
        
        🔍 **Comparativo**: Análisis entre períodos o sectores
        
        📋 **Personalizado**: Informe sobre tema específico
        """)

def generate_report(report_type, period, include_charts, include_recommendations, detailed_analysis, custom_topic=None, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Genera un informe automático basado en los parámetros seleccionados"""
    
    with st.spinner("🔄 Generando informe... Esto puede tomar unos minutos."):
        
        # Definir consultas según el tipo de informe
        queries = get_report_queries(report_type, period, custom_topic)
        
        # Recopilar información
        report_sections = {}
        
        for section_name, query in queries.items():
            try:
                # Generar contenido según el modo de búsqueda seleccionado
                if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
                    section_content = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
                elif search_mode == "📚 Solo RAG":
                    context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
                    section_content = st.session_state.rag_system.query_groq(query, context)
                else:  # Solo Conocimiento General
                    section_content = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
                
                report_sections[section_name] = section_content
                
            except Exception as e:
                st.error(f"Error generando sección {section_name}: {str(e)}")
                report_sections[section_name] = f"Error al generar esta sección: {str(e)}"
        
        # Mostrar el informe generado
        display_generated_report(report_type, period, report_sections, include_recommendations)

def get_report_queries(report_type, period, custom_topic=None):
    """Define las consultas para cada tipo de informe"""
    
    base_queries = {
        "📈 Informe de Coyuntura Económica": {
            "Resumen Ejecutivo": f"Proporciona un resumen ejecutivo de la situación económica actual de Colombia en el {period.lower()}",
            "Indicadores Macroeconómicos": f"¿Cuáles son los principales indicadores macroeconómicos de Colombia en el {period.lower()}? Incluye PIB, inflación, desempleo",
            "Política Monetaria": f"¿Cuál ha sido la política monetaria del Banco de la República en el {period.lower()}?",
            "Perspectivas": f"¿Cuáles son las perspectivas económicas para Colombia según los documentos más recientes?"
        },
        "💰 Análisis Fiscal": {
            "Situación Fiscal Actual": f"¿Cuál es la situación fiscal actual de Colombia en el {period.lower()}?",
            "Ingresos y Gastos": f"Analiza los ingresos y gastos del gobierno colombiano en el {period.lower()}",
            "Déficit y Deuda": f"¿Cuál es el estado del déficit fiscal y la deuda pública en el {period.lower()}?",
            "Reformas Fiscales": f"¿Qué reformas fiscales se han implementado o propuesto en el {period.lower()}?"
        },
        "🏦 Reporte Sectorial Bancario": {
            "Estado del Sistema Financiero": f"¿Cuál es el estado actual del sistema financiero colombiano en el {period.lower()}?",
            "Indicadores Bancarios": f"¿Cuáles son los principales indicadores del sector bancario en el {period.lower()}?",
            "Riesgos y Oportunidades": f"¿Cuáles son los principales riesgos y oportunidades del sector bancario en el {period.lower()}?",
            "Regulación Financiera": f"¿Qué cambios regulatorios han afectado al sector financiero en el {period.lower()}?"
        },
        "📊 Resumen Ejecutivo": {
            "Puntos Clave": f"¿Cuáles son los puntos más importantes de la economía colombiana en el {period.lower()}?",
            "Tendencias Principales": f"¿Cuáles son las principales tendencias económicas identificadas en el {period.lower()}?",
            "Recomendaciones": f"¿Qué recomendaciones de política económica se sugieren para Colombia?"
        },
        "🔍 Análisis Comparativo": {
            "Comparación Temporal": f"Compara la situación económica actual con períodos anteriores",
            "Análisis de Cambios": f"¿Qué cambios significativos se han observado en la economía colombiana?",
            "Evaluación de Políticas": f"Evalúa la efectividad de las políticas económicas implementadas"
        }
    }
    
    if custom_topic:
        return {
            "Análisis Principal": f"Realiza un análisis completo sobre: {custom_topic}",
            "Contexto y Antecedentes": f"Proporciona el contexto y antecedentes relevantes sobre: {custom_topic}",
            "Impactos y Consecuencias": f"¿Cuáles son los principales impactos y consecuencias de: {custom_topic}?",
            "Recomendaciones": f"¿Qué recomendaciones se pueden hacer respecto a: {custom_topic}?"
        }
    
    return base_queries.get(report_type, base_queries["📊 Resumen Ejecutivo"])

def display_generated_report(report_type, period, sections, include_recommendations):
    """Muestra el informe generado en la interfaz"""
    
    st.success("✅ Informe generado exitosamente")
    
    # Título del informe
    st.markdown(f"# {report_type}")
    st.markdown(f"**Período de Análisis:** {period}")
    st.markdown(f"**Fecha de Generación:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.markdown("---")
    
    # Mostrar cada sección
    for section_name, content in sections.items():
        st.subheader(f"## {section_name}")
        st.markdown(content)
        return
    
    if not st.session_state.rag_system.groq_client:
        st.error("❌ Groq no conectado. Configura tu API key primero.")
        return
    
    # Sidebar para selección de herramientas
    with st.sidebar:
        st.header("🛠️ Herramientas Disponibles")
        
        anif_tools = [
            "💰 Monitor Fiscal",
            "🏦 Análisis Política Monetaria", 
            "📋 Tracker de Reformas",
            "🌍 Benchmarking Internacional",
            "📊 Dashboard Indicadores",
            "⚖️ Análisis Regulatorio",
            "📈 Proyecciones Económicas",
            "🔍 Análisis Sectorial"
        ]
        
        selected_tool = st.selectbox("Selecciona una herramienta:", anif_tools)
        
        # Modo de búsqueda para herramientas ANIF
        st.markdown("---")
        st.subheader("🔍 Modo de Búsqueda")
        anif_search_mode = st.radio(
            "Selecciona el modo:",
            ["🔄 Híbrido (RAG + Conocimiento General)", "📚 Solo RAG", "🌐 Solo Conocimiento General"],
            index=0,
            key="anif_search_mode"
        )
    
    # Mostrar la herramienta seleccionada
    if selected_tool == "💰 Monitor Fiscal":
        show_fiscal_monitor(anif_search_mode)
    elif selected_tool == "🏦 Análisis Política Monetaria":
        show_monetary_policy_analysis(anif_search_mode)
    elif selected_tool == "📋 Tracker de Reformas":
        show_reform_tracker(anif_search_mode)
    elif selected_tool == "🌍 Benchmarking Internacional":
        show_international_benchmarking(anif_search_mode)
    elif selected_tool == "📊 Dashboard Indicadores":
        show_indicators_dashboard(anif_search_mode)
    elif selected_tool == "⚖️ Análisis Regulatorio":
        show_regulatory_analysis(anif_search_mode)
    elif selected_tool == "📈 Proyecciones Económicas":
        show_economic_projections(anif_search_mode)
    elif selected_tool == "🔍 Análisis Sectorial":
        show_sectoral_analysis(anif_search_mode)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 Seguimiento Automático del Estado Fiscal
        
        Esta herramienta analiza automáticamente los documentos para extraer y monitorear:
        - Estado actual del déficit/superávit fiscal
        - Evolución de ingresos y gastos públicos
        - Cumplimiento de metas fiscales
        - Proyecciones y alertas
        """)
        
        # Opciones de análisis
        fiscal_analysis_type = st.selectbox(
            "Tipo de análisis:",
            ["Situación Actual", "Tendencias Históricas", "Proyecciones", "Alertas y Riesgos"]
        )
        
        if st.button("🔍 Ejecutar Análisis Fiscal", type="primary"):
            execute_fiscal_analysis(fiscal_analysis_type, search_mode)
    
    with col2:
        st.info("""
        **Indicadores Clave:**
        - Déficit/Superávit Fiscal
        - Deuda Pública/PIB
        - Ingresos Tributarios
        - Gasto Público
        - Regla Fiscal
        """)

def show_monetary_policy_analysis(search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Análisis de política monetaria"""
    st.subheader("🏦 Análisis de Política Monetaria")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Evaluación de Decisiones del Banco de la República
        
        Analiza automáticamente:
        - Decisiones de tasas de interés
        - Comunicados y minutas del JDBR
        - Impacto en mercados financieros
        - Coherencia con objetivos de inflación
        """)
        
        monetary_focus = st.selectbox(
            "Enfoque del análisis:",
            ["Última Decisión", "Tendencia de Tasas", "Comunicación BanRep", "Efectividad de Política"]
        )
        
        if st.button("📈 Analizar Política Monetaria", type="primary"):
            execute_monetary_analysis(monetary_focus, search_mode)
    
    with col2:
        st.info("""
        **Elementos Analizados:**
        - Tasa de Política Monetaria
        - Meta de Inflación
        - Expectativas de Mercado
        - Comunicación Oficial
        - Transmisión de Política
        """)

def show_reform_tracker(search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Seguimiento de reformas"""
    st.subheader("📋 Tracker de Reformas")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### ⚖️ Seguimiento de Reformas Estructurales
        
        Monitorea el progreso de:
        - Reforma tributaria
        - Reforma pensional
        - Reforma a la salud
        - Reformas laborales
        - Otras reformas estructurales
        """)
        
        reform_type = st.selectbox(
            "Tipo de reforma:",
            ["Todas las Reformas", "Reforma Tributaria", "Reforma Pensional", "Reforma Salud", "Reformas Laborales"]
        )
        
        if st.button("📊 Generar Reporte de Reformas", type="primary"):
            execute_reform_tracking(reform_type, search_mode)
    
    with col2:
        st.info("""
        **Estado de Seguimiento:**
        - Propuestas Presentadas
        - Trámite Legislativo
        - Modificaciones
        - Impacto Esperado
        - Cronograma
        """)

def show_international_benchmarking(search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Benchmarking internacional"""
    st.subheader("🌍 Benchmarking Internacional")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🔄 Comparación con Países de Referencia
        
        Compara indicadores de Colombia con:
        - Países de la región (Chile, Perú, México)
        - Países emergentes similares
        - Mejores prácticas internacionales
        """)
        
        benchmark_focus = st.selectbox(
            "Área de comparación:",
            ["Indicadores Fiscales", "Política Monetaria", "Competitividad", "Desarrollo Financiero"]
        )
        
        if st.button("🌐 Ejecutar Benchmarking", type="primary"):
            execute_benchmarking(benchmark_focus, search_mode)
    
    with col2:
        st.info("""
        **Países de Referencia:**
        - Chile
        - Perú
        - México
        - Brasil
        - Países OCDE
        """)

def show_indicators_dashboard(search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Dashboard de indicadores"""
    st.subheader("📊 Dashboard de Indicadores Económicos")
    
    st.markdown("""
    ### 📈 Panel de Control de Indicadores Clave
    
    Extrae y visualiza automáticamente los principales indicadores económicos de los documentos.
    """)
    
    # Crear métricas simuladas (en implementación real extraería de documentos)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PIB (Crecimiento)", "3.2%", "0.5%")
    
    with col2:
        st.metric("Inflación", "5.8%", "-0.3%")
    
    with col3:
        st.metric("Desempleo", "9.1%", "-0.2%")
    
    with col4:
        st.metric("Tasa de Política", "10.75%", "0.0%")
    
    if st.button("🔄 Actualizar Indicadores"):
        st.info("Funcionalidad de extracción automática de indicadores en desarrollo")

def show_regulatory_analysis(search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Análisis regulatorio"""
    st.subheader("⚖️ Análisis Regulatorio")
    
    st.markdown("""
    ### 📜 Evaluación de Cambios Normativos
    
    Analiza el impacto de nuevas regulaciones en el sector financiero y la economía.
    """)
    
    regulatory_area = st.selectbox(
        "Área regulatoria:",
        ["Regulación Financiera", "Normativa Fiscal", "Regulación Cambiaria", "Supervisión Bancaria"]
    )
    
    if st.button("⚖️ Analizar Impacto Regulatorio"):
        execute_regulatory_analysis(regulatory_area, search_mode)

def show_economic_projections(search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Proyecciones económicas"""
    st.subheader("📈 Proyecciones Económicas")
    
    st.markdown("""
    ### 🔮 Consolidación de Proyecciones
    
    Recopila y compara proyecciones económicas de diferentes fuentes.
    """)
    
    projection_variable = st.selectbox(
        "Variable a proyectar:",
        ["PIB", "Inflación", "Tasa de Cambio", "Desempleo", "Déficit Fiscal"]
    )
    
    if st.button("📊 Generar Consolidado de Proyecciones"):
        execute_projections_analysis(projection_variable, search_mode)

def show_sectoral_analysis(search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Análisis sectorial"""
    st.subheader("🔍 Análisis Sectorial")
    
    st.markdown("""
    ### 🏭 Evaluación por Sectores Económicos
    
    Analiza el desempeño y perspectivas de sectores específicos.
    """)
    
    sector = st.selectbox(
        "Sector a analizar:",
        ["Sector Financiero", "Sector Real", "Sector Externo", "Sector Público", "Sector Energético"]
    )
    
    if st.button("🔍 Ejecutar Análisis Sectorial"):
        execute_sectoral_analysis(sector, search_mode)

# Funciones de ejecución para cada herramienta
def execute_fiscal_analysis(analysis_type, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Ejecuta análisis fiscal específico"""
    with st.spinner("🔄 Analizando información fiscal..."):
        query = f"Analiza la situación fiscal actual de Colombia enfocándote en {analysis_type.lower()}. Incluye datos específicos sobre déficit, deuda pública, ingresos y gastos."
        
        # Usar el modo de búsqueda seleccionado
        if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
        elif search_mode == "📋 Solo RAG":
            context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
            response = st.session_state.rag_system.query_groq(query, context)
        else:  # Solo Conocimiento General
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
        
        st.success("✅ Análisis fiscal completado")
        st.markdown("### 📊 Resultados del Análisis Fiscal")
        st.markdown(response)

def execute_monetary_analysis(focus, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Ejecuta análisis de política monetaria"""
    with st.spinner("🔄 Analizando política monetaria..."):
        query = f"Analiza la política monetaria del Banco de la República enfocándote en {focus.lower()}. Incluye decisiones recientes, comunicación oficial y impacto esperado."
        
        # Usar el modo de búsqueda seleccionado
        if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
        elif search_mode == "📋 Solo RAG":
            context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
            response = st.session_state.rag_system.query_groq(query, context)
        else:  # Solo Conocimiento General
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
        
        st.success("✅ Análisis de política monetaria completado")
        st.markdown("### 🏦 Resultados del Análisis Monetario")
        st.markdown(response)

def execute_reform_tracking(reform_type, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Ejecuta seguimiento de reformas"""
    with st.spinner("🔄 Rastreando información sobre reformas..."):
        query = f"Proporciona un seguimiento detallado sobre {reform_type.lower()} en Colombia. Incluye estado actual, avances, obstáculos y cronograma esperado."
        
        # Usar el modo de búsqueda seleccionado
        if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
        elif search_mode == "📋 Solo RAG":
            context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
            response = st.session_state.rag_system.query_groq(query, context)
        else:  # Solo Conocimiento General
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
        
        st.success("✅ Seguimiento de reformas completado")
        st.markdown("### 📋 Estado de las Reformas")
        st.markdown(response)

def execute_benchmarking(focus, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Ejecuta benchmarking internacional"""
    with st.spinner("🔄 Realizando comparación internacional..."):
        query = f"Compara los indicadores de {focus.lower()} de Colombia con países de referencia como Chile, Perú y México. Identifica brechas y mejores prácticas."
        
        # Usar el modo de búsqueda seleccionado
        if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
        elif search_mode == "📋 Solo RAG":
            context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
            response = st.session_state.rag_system.query_groq(query, context)
        else:  # Solo Conocimiento General
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
        
        st.success("✅ Benchmarking internacional completado")
        st.markdown("### 🌍 Comparación Internacional")
        st.markdown(response)

def execute_regulatory_analysis(area, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Ejecuta análisis regulatorio"""
    with st.spinner("🔄 Analizando impacto regulatorio..."):
        query = f"Analiza los cambios regulatorios recientes en {area.lower()} y su impacto en el sector financiero y la economía colombiana."
        
        # Usar el modo de búsqueda seleccionado
        if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
        elif search_mode == "📋 Solo RAG":
            context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
            response = st.session_state.rag_system.query_groq(query, context)
        else:  # Solo Conocimiento General
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
        
        st.success("✅ Análisis regulatorio completado")
        st.markdown("### ⚖️ Impacto Regulatorio")
        st.markdown(response)

def execute_projections_analysis(variable, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Ejecuta análisis de proyecciones"""
    with st.spinner("🔄 Consolidando proyecciones..."):
        query = f"Recopila y compara las proyecciones más recientes para {variable.lower()} en Colombia de diferentes fuentes oficiales y privadas."
        
        # Usar el modo de búsqueda seleccionado
        if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
        elif search_mode == "📋 Solo RAG":
            context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
            response = st.session_state.rag_system.query_groq(query, context)
        else:  # Solo Conocimiento General
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
        
        st.success("✅ Consolidado de proyecciones completado")
        st.markdown("### 📈 Proyecciones Consolidadas")
        st.markdown(response)

def execute_sectoral_analysis(sector, search_mode="🔄 Híbrido (RAG + Conocimiento General)"):
    """Ejecuta análisis sectorial"""
    with st.spinner("🔄 Analizando sector específico..."):
        query = f"Analiza el desempeño, retos y perspectivas del {sector.lower()} en Colombia según la información más reciente disponible."
        
        # Usar el modo de búsqueda seleccionado
        if search_mode == "🔄 Híbrido (RAG + Conocimiento General)":
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=True)
        elif search_mode == "📋 Solo RAG":
            context = st.session_state.rag_system.search_similar_documents(query, k=5) if st.session_state.rag_system.documents_loaded else ""
            response = st.session_state.rag_system.query_groq(query, context)
        else:  # Solo Conocimiento General
            response = st.session_state.rag_system.query_groq_hybrid(query, use_rag=False)
        
        st.success("✅ Análisis sectorial completado")
        st.markdown("### 🔍 Análisis del Sector")
        st.markdown(response)

if __name__ == "__main__":
    main()