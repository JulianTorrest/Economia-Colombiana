# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Importaciones diferidas para optimizar carga
def lazy_import_langchain():
    try:
        from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        return PyPDFLoader, UnstructuredExcelLoader, FAISS, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter, Document
    except ImportError:
        # Fallbacks para versiones anteriores
        from langchain.document_loaders import PyPDFLoader, UnstructuredExcelLoader
        from langchain.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        return PyPDFLoader, UnstructuredExcelLoader, FAISS, HuggingFaceEmbeddings, RecursiveCharacterTextSplitter, Document

def lazy_import_groq():
    from groq import Groq
    return Groq

class ANIFRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.groq_client = None
        self.documents_loaded = False
        
        self.domain_prompts = {
            "fiscal": "Como experto senior en política fiscal colombiana...",
            "monetario": "Como analista especializado del Banco de la República...",
            "sectorial": "Como especialista en análisis sectorial...",
            "internacional": "Como experto en economía internacional...",
            "laboral": "Como especialista en mercado laboral colombiano...",
            "general": "Como economista senior especializado en Colombia..."
        }
        
        self.temporal_context = """
        Contexto económico actual de Colombia (2024-2026):
        - Economía post-pandemia en proceso de normalización
        - Banco de la República en ciclo de política monetaria restrictiva
        - Inflación convergiendo gradualmente hacia la meta del 3%
        """
    
    def initialize_groq(self, api_key: str) -> bool:
        if not api_key or not api_key.startswith('gsk_'):
            return False
        try:
            Groq = lazy_import_groq()
            self.groq_client = Groq(api_key=api_key.strip())
            return True
        except Exception as e:
            print(f"Error initializing Groq: {e}")
            return False
    
    def load_prebuilt_vectorstore(self) -> bool:
        try:
            _, _, FAISS, HuggingFaceEmbeddings, _, _ = lazy_import_langchain()
            
            if not self.embeddings:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
            
            if os.path.exists("vectorstore") and os.path.exists("rag_ready.flag"):
                self.vectorstore = FAISS.load_local("vectorstore", self.embeddings, allow_dangerous_deserialization=True)
                self.documents_loaded = True
                return True
            return False
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            return False

    def classify_economic_query(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(k in prompt_lower for k in ["fiscal", "tributario", "impuesto", "déficit", "deuda"]): return "fiscal"
        if any(k in prompt_lower for k in ["monetario", "inflación", "tasa", "banrep"]): return "monetario"
        if any(k in prompt_lower for k in ["bancario", "financiero", "industrial", "servicios"]): return "sectorial"
        if any(k in prompt_lower for k in ["exportaciones", "importaciones", "dólar", "cambio"]): return "internacional"
        if any(k in prompt_lower for k in ["empleo", "desempleo", "salario"]): return "laboral"
        return "general"

    def get_enhanced_system_prompt(self, query_type: str) -> str:
        return f"""
        {self.temporal_context}
        {self.domain_prompts.get(query_type, self.domain_prompts["general"])}.
        Instrucciones: Responde en español técnico, cita fuentes implícitas, usa datos específicos.
        """

    def search_similar_documents_with_scores(self, query: str, k: int = 3) -> List[Tuple[Any, float]]:
        """Retorna documentos y sus puntajes de similitud (distancia L2, menor es mejor)"""
        if not self.vectorstore:
            return []
        try:
            # FAISS retorna distancia L2. Score conversion puede variar.
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            return docs_and_scores
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def search_similar_documents(self, query: str, k: int = 3) -> str:
        """Legacy wrapper para mantener compatibilidad"""
        results = self.search_similar_documents_with_scores(query, k)
        return "\n\n".join([doc.page_content for doc, score in results])

    def query_groq_hybrid(self, prompt: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Retorna un diccionario con la respuesta y metadatos para cumplir con los requisitos de API.
        Estructura: { 'answer': str, 'context_used': bool, 'sources': list, 'rejection': bool }
        """
        if not self.groq_client:
            return {"answer": "Error: Groq no inicializado", "rejection": True}
        
        context_text = ""
        sources = []
        rag_confidence = 0.0
        
        if use_rag and self.documents_loaded:
            results = self.search_similar_documents_with_scores(prompt, k=4)
            # Filtrar por umbral de calidad (ej. distancia < 1.0 para L2, o score > 0.7 para coseno)
            # Asumimos L2 de FAISS donde 0 es idéntico. Un umbral razonable depende de los embeddings.
            valid_results = [r for r in results if r[1] < 1.2] 
            
            if valid_results:
                context_text = "\n\n".join([doc.page_content for doc, score in valid_results])
                sources = [{"content": doc.page_content[:200], "source": doc.metadata.get("source", "unknown"), "score": float(score)} for doc, score in valid_results]
                rag_confidence = 1.0 # Simplificado para el ejemplo
            
        # Lógica de rechazo estricto para la API (si se solicita solo RAG)
        if use_rag and not context_text:
             # Si es modo estricto, podríamos rechazar aquí. 
             # Para modo híbrido, continuamos con conocimiento general.
             pass

        query_type = self.classify_economic_query(prompt)
        system_prompt = self.get_enhanced_system_prompt(query_type)
        
        if context_text:
            user_prompt = f"""Información de documentos ANIF:\n{context_text}\n\nPregunta: {prompt}\n\nResponde combinando los documentos y tu conocimiento."""
        else:
            user_prompt = f"""Pregunta: {prompt}\n\nResponde basándote en tu conocimiento experto de la economía colombiana."""

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=2000
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "sources": sources,
                "rejection": False,
                "toolUsed": "intent_classifier"
            }
            
        except Exception as e:
            return {"answer": f"Error en Groq: {str(e)}", "rejection": True}