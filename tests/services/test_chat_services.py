from fastapi.responses import StreamingResponse
from app.services.chat_services import chat, get_chat_name

import pytest

def test_chat_success():
    # Test the chat function with a sample question and context
    question = "Quali sono i vostri prodotti?"
    context = "prodotti dell'azienda"
    
    response = chat(question, context)
    
    assert isinstance(response, StreamingResponse), "La risposta non è di tipo StreamingResponse"
    assert response.status_code == 200, "La risposta non ha restituito un codice di stato 200"

def test_get_chat_name_success():
    # Test the get_chat_name function with a sample context
    context = "Parla dei prodotti e dell'azienda"
    response = get_chat_name(context)
    assert isinstance(response, str), "La risposta non è di tipo stringa"
    assert len(response) > 0, "La risposta non contiene dati validi"