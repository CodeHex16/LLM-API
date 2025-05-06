import pytest
from httpx import AsyncClient, ASGITransport

from fastapi import FastAPI, HTTPException

from app.routes.documents import router
from unittest.mock import MagicMock, patch 
from io import BytesIO
from fastapi import UploadFile

from app.routes.documents import upload_file

app = FastAPI()
app.include_router(router)

transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_upload_file_no_file():
    with pytest.raises(HTTPException) as exc_info:
        await upload_file(None, "test_token")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No file uploaded"

@pytest.mark.asyncio
async def test_upload_file_no_file_name():
    file = UploadFile(filename="", file=BytesIO(b"test content"))
    with pytest.raises(HTTPException) as exc_info:
        await upload_file(file, "test_token")
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No filename provided"


@pytest.mark.asyncio
async def test_upload_file_error_filename_ext():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Mock an empty file as UploadFile using BytesIO
        empty_file = BytesIO(b"")
        # Prepare the file for sending (as a tuple with filename)
        files_to_upload = {"file": ("jj.exe", empty_file)}  # Correct format with filename "hi"

        # Sending the file with no filename (empty file)
        response = await ac.post(
            "/documents/upload_file?token=test_token",  # Token passed as query parameter
            files=files_to_upload,  # Correct usage of file as a tuple
        )
        print("Response:", response.json())
        assert response.status_code == 400
        assert response.json() == {'detail': 'Only txt/pdf files are allowed'}

@pytest.mark.asyncio
async def test_upload_file_error_filename_content_type():
    pass

@pytest.mark.asyncio
async def test_upload_file_filename_ok(monkeypatch):

    fm = MagicMock()
    # Mock the method that deletes the document and ensure it accepts both parameters
    async def mock_delete_document(file_path, token):
        # Simulate a successful document deletion (you can also log or return a value if needed)
        return True  # Or perform the necessary action you want to simulate
    # Set the mock method to return our custom method
    fm.add_document = mock_delete_document

    # Use monkeypatch to replace the real function with the mocked one
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda x: fm)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Mock an empty file as UploadFile using BytesIO
        empty_file = BytesIO(b"")
        # Prepare the file for sending (as a tuple with filename)
        files_to_upload = {"file": ("jj.txt", empty_file)}  # Correct format with filename "hi"

        # Sending the file with no filename (empty file)
        response = await ac.post(
            "/documents/upload_file?token=test_token",  # Token passed as query parameter
            files=files_to_upload,  # Correct usage of file as a tuple
        )
        print("Response:", response.json())
        assert response.json() == {"message": "File uploaded successfully"}

   
@pytest.mark.asyncio
async def test_delete_file_no_file_path(monkeypatch):
    
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda x: None)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
      response = await ac.delete("/documents/delete_file?file_path=/testfile&token=test_token")
      assert response.status_code == 400

@pytest.mark.asyncio
async def test_delete_file_error(monkeypatch):
    # Create a MagicMock instance for the file manager
    fm = MagicMock()

    # Mock the method that deletes the document and make it raise an HTTPException
    async def mock_delete_document(file_path, token):
        raise HTTPException(status_code=500)

    # Set the mock method to return our custom method
    fm.delete_document = mock_delete_document

    # Use monkeypatch to replace the real function with the mocked one
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda x: fm)

    # Make the HTTP DELETE request to the endpoint
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete("/documents/delete_file?file_path=test.txt&token=pp")

    print("Response:", response.json())
    # Check if the status code is 400, which we expect from the exception
    assert response.status_code == 500
    assert response.json() == {"detail": "Error in deleting file"}

    
@pytest.mark.asyncio
async def test_delete_file_ok(monkeypatch):
    # Create a MagicMock instance for the file manager
    fm = MagicMock()

    # Mock the method that deletes the document and ensure it accepts both parameters
    async def mock_delete_document(file_path, token):
        # Simulate a successful document deletion (you can also log or return a value if needed)
        return True  # Or perform the necessary action you want to simulate

    # Set the mock method to return our custom method
    fm.delete_document = mock_delete_document

    # Use monkeypatch to replace the real function with the mocked one
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda x: fm)

    # Make the HTTP DELETE request to the endpoint
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete("/documents/delete_file?file_path=test.txt&token=pp")
    # Assert that the response is as expected
    assert response.status_code == 200
    assert response.json() == {"message": "File deleted successfully"}

@pytest.mark.asyncio
async def test_upload_file_missing_file():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/documents/upload_file?token=test_token", files={})
    assert response.status_code == 422
    print("Response:", response.json())
    assert response.json()["detail"][0]["input"] == None

@pytest.mark.asyncio
async def test_upload_file_missing_filename():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files={"file": ("", b"some content", "text/plain")},  # filename is ""
        )
    assert response.status_code == 422
    print("Response:", response.json())
    assert response.json()["detail"][0]["input"] == "some content"

@pytest.mark.asyncio
async def test_upload_file_invalid_extension():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files={"file": ("file.exe", b"some content", "application/octet-stream")},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only txt/pdf files are allowed"

@pytest.mark.asyncio
async def test_upload_file_invalid_content_type():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files={"file": ("file.pdf", b"some content", "application/octet-stream")},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Only txt/pdf content type is allowed"

@pytest.mark.asyncio
async def test_exception_handling_500(monkeypatch):
    # Mock the file manager to raise an HTTPException
    fm = MagicMock()
    async def mock_add_document(file, token):
        raise HTTPException(status_code=500, detail="Error in uploading and processing file")
    
    fm.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda x: fm)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        empty_file = BytesIO(b"")
        files_to_upload = {"file": ("jj.txt", empty_file)}
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files_to_upload,
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error in uploading and processing file"}

@pytest.mark.asyncio
async def test_exception_handling_400(monkeypatch):
    # Mock the file manager to raise an HTTPException
    fm = MagicMock()
    async def mock_add_document(file, token):
        raise HTTPException(status_code=400, detail="Document already exists")
    
    fm.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda x: fm)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        empty_file = BytesIO(b"")
        files_to_upload = {"file": ("jj.txt", empty_file)}
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files_to_upload,
        )
    assert response.status_code == 400
    assert response.json() == {"detail": "Document already exists"}

@pytest.mark.asyncio
async def test_exception_handling_501(monkeypatch):
    # Mock the file manager to raise an HTTPException
    fm = MagicMock()
    async def mock_add_document(file, token):
        raise HTTPException(status_code=501, detail="Document already exists")
    
    fm.add_document = mock_add_document
    monkeypatch.setattr("app.routes.documents.get_file_manager", lambda x: fm)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        empty_file = BytesIO(b"")
        files_to_upload = {"file": ("jj.txt", empty_file)}
        response = await ac.post(
            "/documents/upload_file?token=test_token",
            files=files_to_upload,
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "Error in uploading and processing file"}


@pytest.mark.asyncio
async def test_delete_exception_handling_404(monkeypatch):
    # Mock the file manager to raise an HTTPException
    fm = MagicMock()
    async def mock_delete_document(file_path, token):
        raise HTTPException(status_code=404, detail="Document not found")
    
    fm.delete_document = mock_delete_document
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda x: fm)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete("/documents/delete_file?file_path=test.txt&token=pp")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}

@pytest.mark.asyncio
async def test_delete_exception_handling_500(monkeypatch):
    # Mock the file manager to raise an HTTPException
    fm = MagicMock()
    async def mock_delete_document(file_path, token):
        raise HTTPException(status_code=500, detail="Error in deleting file")
    
    fm.delete_document = mock_delete_document
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda x: fm)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete("/documents/delete_file?file_path=test.txt&token=pp")
    assert response.status_code == 500
    assert response.json() == {"detail": "Error in deleting file"}

@pytest.mark.asyncio
async def test_delete_exception_handling_http_default(monkeypatch):
    # Mock the file manager to raise an HTTPException
    fm = MagicMock()
    async def mock_delete_document(file_path, token):
        raise HTTPException(status_code=501, detail="Not Implemented")
      
    fm.delete_document = mock_delete_document
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda x: fm)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete("/documents/delete_file?file_path=test.txt&token=pp")
    assert response.status_code == 500
    assert response.json() == {"detail": "Error in deleting file"}

@pytest.mark.asyncio
async def test_delete_exception_handling_500_generic(monkeypatch):
    # Mock the file manager to raise an HTTPException
    fm = MagicMock()
    async def mock_delete_document(file_path, token):
        raise Exception("Generic error")
    
    fm.delete_document = mock_delete_document
    monkeypatch.setattr("app.routes.documents.get_file_manager_by_extension", lambda x: fm)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.delete("/documents/delete_file?file_path=test.txt&token=pp")
    assert response.status_code == 500
    assert response.json() == {"detail": "Error in deleting file"}