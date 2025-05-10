import pytest

from app.main import app
from httpx import ASGITransport, AsyncClient

transport = ASGITransport(app=app)

@pytest.mark.asyncio
async def test_main():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/ping")
        assert response.status_code == 200
        assert response.json().get("status") == "ok"