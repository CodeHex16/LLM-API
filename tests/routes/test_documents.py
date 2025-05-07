from typing import BinaryIO
import pytest
from httpx import AsyncClient, ASGITransport

from fastapi import FastAPI, HTTPException

from app.routes.documents import router
from unittest.mock import MagicMock, patch 
from io import BytesIO
from fastapi import UploadFile

from app.routes.documents import upload_file
import os

app = FastAPI()
app.include_router(router)

transport = ASGITransport(app=app)

@pytest.mark.asyncio
async def test_upload_file_no_filename():
    files = [UploadFile(
        filename="",
        file=BytesIO(b"file content here")
    )]
    with pytest.raises(HTTPException) as exc_info:
        await upload_file(files, "toktok")
    assert exc_info.value.status_code == 400
    assert "No filename provided" in exc_info.value.detail


@pytest.mark.asyncio
async def test_upload_file_no_file():
    with pytest.raises(HTTPException) as exc_info:
        await upload_file(None, "test_token")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No files uploaded"

@pytest.mark.asyncio
async def test_upload_file_with_error_ext(monkeypatch):
    mock_add_document = MagicMock()
    fileManager = MagicMock()
    fileManager.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda _: fileManager)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create the mock files
        files = [
            ("files", ("test.exe", BytesIO(b"test content"), "text/plain")),  # Valid file
        ]
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files,  # Correct usage of file as a tuple
        )
        print("Response:", response.json())
        assert response.status_code == 400
        assert "Only txt/pdf files are allowed" in response.json()["detail"], "Error message should indicate invalid file type"

@pytest.mark.asyncio
async def test_upload_file_with_error_mme(monkeypatch):
  
    fileManager = MagicMock()
    fileManager.add_document = MagicMock()
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda _: fileManager)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create the mock files
        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/test")),  # Valid file
        ]
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files,  # Correct usage of file as a tuple
        )
        print("Response:", response.json())
        assert response.status_code == 400
        assert "Invalid content type" in response.json()["detail"], "Error message should indicate invalid content type"

@pytest.mark.asyncio
async def test_upload_file_successful(monkeypatch):
    async def mock_add_document(_,__):
        pass
    fileManager = MagicMock()
    fileManager.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda _: fileManager)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create the mock files
        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/plain")),  # Valid file
        ]
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files,  # Correct usage of file as a tuple
        )
        print("Response:", response.json())
        assert "Successfully" in response.json()["message"], "Error message should indicate successful upload"
        assert " 1 " in response.json()["message"], "Should upload 1 file."

@pytest.mark.asyncio
async def test_upload_file_add_document_http_exception(monkeypatch):
    async def mock_add_document(_,__):
        raise HTTPException(status_code=400, detail="EERROR")
    fileManager = MagicMock()
    fileManager.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda _: fileManager)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create the mock files
        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/plain")),  # Valid file
        ]
        
        response = await ac.post(
                "/documents/upload_file?token=test_token",
                files=files,  # Correct usage of file as a tuple
            )
        assert "EERROR" in response.json()["detail"], "EERROR should be in the error message"
     
@pytest.mark.asyncio
async def test_upload_file_add_document_exception(monkeypatch):
    async def mock_add_document(_,__):
        raise Exception("EERROR")
    fileManager = MagicMock()
    fileManager.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda _: fileManager)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create the mock files
        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/plain")),  # Valid file
        ]
     
        response = await ac.post(
                "/documents/upload_file?token=test_token",
                files=files,  # Correct usage of file as a tuple
            )
        assert "EERROR" in response.json()["detail"], "EERROR should be in the error message"
     
@pytest.mark.asyncio
async def test_upload_files_successful(monkeypatch):
    async def mock_add_document(_,__):
        pass
    fileManager = MagicMock()
    fileManager.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda _: fileManager)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create the mock files
        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/plain")),  # Valid file
            ("files", ("test2.txt", BytesIO(b"test content"), "text/plain")),  # Valid file
        ]
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files,  # Correct usage of file as a tuple
        )
        print("Response:", response.json())
        assert "Successfully" in response.json()["message"], "Error message should indicate successful upload"
        assert " 2 " in response.json()["message"], "Should upload 1 file."

@pytest.mark.asyncio
async def test_upload_file_with_one_error(monkeypatch):
    async def mock_add_document(_,__):
        pass
    fileManager = MagicMock()
    fileManager.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda _: fileManager)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Create the mock files
        files = [
            ("files", ("test.txt", BytesIO(b"test content"), "text/plain")),  # Valid file
            ("files", ("test.exe", BytesIO(b"test content"), "text/plain")),  # Valid file
        ]
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files,  # Correct usage of file as a tuple
        )
        print("Response:", response.json())
        assert "Processed 1 files" in response.json()["message"], "Error message should indicate successful upload"
        



@pytest.mark.asyncio
async def test_delete_file_success(monkeypatch):
    async def mock_delete_document(*args, **kwargs):
        pass

    file_manager = MagicMock()
    file_manager.get_full_path.return_value = "/data/documents/test.txt"
    file_manager.delete_document = mock_delete_document

    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda _: file_manager)

    payload = {
        "id": "doc123",
        "title": "test.txt",
        "token": "test_token",
        "current_password": "password123"
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.request("DELETE", "/documents/delete_file", json=payload)
        assert response.status_code == 200
        assert response.json()["message"] == "File deleted successfully"

@pytest.mark.asyncio
async def test_delete_file_file_manager_not_found(monkeypatch):
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda _: None)

    payload = {
        "id": "doc123",
        "title": "invalid.xyz",
        "token": "test_token",
        "current_password": "password123"
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.request("DELETE", "/documents/delete_file", json=payload)
        assert response.status_code == 400
        assert response.json()["detail"] == "File manager not found"

@pytest.mark.asyncio
async def test_delete_file_http_exception_404(monkeypatch):
    async def mock_delete_document(*args, **kwargs):
        raise HTTPException(status_code=404, detail="Document not found")

    file_manager = MagicMock()
    file_manager.get_full_path.return_value = "/data/documents/test.txt"
    file_manager.delete_document = mock_delete_document

    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda _: file_manager)

    payload = {
        "id": "doc123",
        "title": "test.txt",
        "token": "test_token",
        "current_password": "password123"
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.request("DELETE", "/documents/delete_file", json=payload)
        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found"

@pytest.mark.asyncio
async def test_delete_file_http_exception_500(monkeypatch):
    async def mock_delete_document(*args, **kwargs):
        raise HTTPException(status_code=500, detail="Error in deleting file")

    file_manager = MagicMock()
    file_manager.get_full_path.return_value = "/data/documents/test.txt"
    file_manager.delete_document = mock_delete_document

    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda _: file_manager)

    payload = {
        "id": "doc123",
        "title": "test.txt",
        "token": "test_token",
        "current_password": "password123"
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.request("DELETE", "/documents/delete_file", json=payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "Error in deleting file"

@pytest.mark.asyncio
async def test_delete_file_http_exception_default(monkeypatch):
    async def mock_delete_document(*args, **kwargs):
        raise HTTPException(status_code=501, detail="Error in deleting file")

    file_manager = MagicMock()
    file_manager.get_full_path.return_value = "/data/documents/test.txt"
    file_manager.delete_document = mock_delete_document

    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda _: file_manager)

    payload = {
        "id": "doc123",
        "title": "test.txt",
        "token": "test_token",
        "current_password": "password123"
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.request("DELETE", "/documents/delete_file", json=payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "Error in deleting file"


@pytest.mark.asyncio
async def test_delete_file_general_exception(monkeypatch):
    async def mock_delete_document(*args, **kwargs):
        raise Exception("Unexpected error")

    file_manager = MagicMock()
    file_manager.get_full_path.return_value = "/data/documents/test.txt"
    file_manager.delete_document = mock_delete_document

    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda _: file_manager)

    payload = {
        "id": "doc123",
        "title": "test.txt",
        "token": "test_token",
        "current_password": "password123"
    }

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.request("DELETE", "/documents/delete_file", json=payload)
        assert response.status_code == 500
        assert "Error in deleting file" in response.json()["detail"]

@pytest.mark.asyncio
async def test_get_documents(monkeypatch):
    monkeypatch.setattr(os, "listdir", lambda _: ["doc1.txt", "doc2.pdf"])
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/documents/get_documents")

        print("Response:", response.json())

        assert response.status_code == 200
        assert response.json() == ["doc1.txt", "doc2.pdf"], "Should return the list of documents"