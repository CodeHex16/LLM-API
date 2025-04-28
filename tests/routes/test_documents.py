import pytest
from httpx import AsyncClient, ASGITransport

from fastapi import FastAPI, HTTPException

from app.routes.documents import router
from unittest.mock import MagicMock, patch 
from io import BytesIO
from fastapi import UploadFile

app = FastAPI()
app.include_router(router)

transport = ASGITransport(app=app)



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