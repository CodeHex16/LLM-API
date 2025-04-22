import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI

from app.routes.chroma import router

app = FastAPI()
app.include_router(router, prefix="/chat")

transport = ASGITransport(app=app)

@pytest.mark.asyncio
async def test_create_chat_response_success(monkeypatch):
    def fake_embedding(question):
        return "fake context"

    def fake_chat(question, context):
        print(f"Fake chat called with question: {question} and context: {context}")
        return {"answer": "Fake answer"}

    monkeypatch.setattr("app.routes.chroma.chat", fake_chat)
    monkeypatch.setattr("app.routes.chroma.embedding", fake_embedding)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/chat/", json={"question": "Ciao, come stai?"})
        assert response.status_code == 200
        assert response.text == '{"answer":"Fake answer"}'


@pytest.mark.asyncio
async def test_create_chat_response_no_question():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/chat/", json={"question": ""})
        assert response.status_code == 400
        assert response.json() == {"detail": "Nessuna domanda fornita"}


@pytest.mark.asyncio
async def test_generate_chat_name_success(monkeypatch):
    def fake_get_chat_name(context):
        return "Chat di prova"

    monkeypatch.setattr("app.routes.chroma.get_chat_name", fake_get_chat_name)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/chat/chat_name", json={"context": "Questo è un esempio di contesto."})
        assert response.status_code == 200
        assert response.text == '"Chat di prova"'


@pytest.mark.asyncio
async def test_generate_chat_name_no_context():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/chat/chat_name", json={"context": ""})
        assert response.status_code == 400
        assert response.json() == {"detail": "Nessun contesto fornito"}
