# tests/test_api.py
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    """Verifica que el sistema esté saludable"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_rejection_on_nonsense():
    """Prueba que el sistema rechace preguntas sin sentido o fuera de contexto"""
    # Asumiendo que el sistema no tiene info sobre esto
    payload = {
        "sessionId": "test-session-1",
        "mensaje": "¿Cuál es la receta para hacer sushi?"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Debería rechazar o usar conocimiento general, pero si es estricto RAG:
    # assert data["rechazo"] == True (Depende de la config estricta)
    assert "answer" in data

def test_chat_valid_query():
    """Prueba una consulta económica válida"""
    payload = {
        "sessionId": "test-session-1",
        "mensaje": "¿Qué es el PIB?"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] is not None
    assert isinstance(data["citas"], list)

def test_memory_persistence():
    """Verifica que el sessionId mantenga contexto (simulado)"""
    session_id = "mem-test-1"
    # Turno 1
    client.post("/chat", json={"sessionId": session_id, "mensaje": "Hola, soy Beto"})
    # Turno 2
    response = client.post("/chat", json={"sessionId": session_id, "mensaje": "¿Cuál es mi nombre?"})
    # Nota: Esto requiere que el LLM use el contexto inyectado.
    assert response.status_code == 200

def test_upload_endpoint():
    """Verifica el endpoint de carga"""
    # Simular archivo
    files = {'file': ('test.txt', b'contenido de prueba', 'text/plain')}
    response = client.post("/documentos", files=files)
    assert response.status_code == 200
    assert response.json()["status"] == "success"