import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from app.routes.llm import router
from app.services.llm_response_service import LLMResponseService

app = FastAPI()
app.include_router(router, prefix="/llm")

transport = ASGITransport(app=app)

@pytest.mark.asyncio
async def test_create_chat_response_success(monkeypatch):
    # Mock LLMResponseService's generate_llm_response to return a list
    def fake_generate_llm_response(self, question):
        return ["fake context", "more fake content"]

    # Mock the method
    monkeypatch.setattr(LLMResponseService, "generate_llm_response", fake_generate_llm_response)

    # Using AsyncClient to test FastAPI route
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/llm/", json={"question": "Ciao, come stai?"})

    # Add assertions as needed
    assert response.status_code == 200
    assert response.json() == ["fake context", "more fake content"]

@pytest.mark.asyncio
async def test_create_chat_response_no_question():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/llm/", json={"question": ""})
        assert response.status_code == 400
        assert response.json() == {"detail": "Nessuna domanda fornita"}


@pytest.mark.asyncio
async def test_generate_chat_name_success(monkeypatch):
    # Fake function to simulate generating a chat name
    def fake_get_chat_name(context: str, *args, **kwargs) -> str:
        return "Chat di prova"

    # Use monkeypatch to replace the original function with the fake one
    monkeypatch.setattr("app.services.llm_response_service.LLMResponseService.generate_llm_chat_name", fake_get_chat_name)

    # Using AsyncClient to test the FastAPI route
    async with AsyncClient(transport=transport,base_url="http://test") as ac:
        response = await ac.post("/llm/chat_name", json={"context": "Questo è un esempio di contesto."})
        # Verify the status code
        assert response.status_code == 200
        
        # Verify that the response is the expected chat name
        assert response.json() == "Chat di prova"

@pytest.mark.asyncio
async def test_generate_chat_name_no_context():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/llm/chat_name", json={"context": ""})
        assert response.status_code == 400
        assert response.json() == {"detail": "Nessun contesto fornito"}

@pytest.mark.asyncio
async def test_ping():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/llm/ping")
        assert response.status_code == 200