import pytest
from app.services.file_manager_service import get_file_manager, get_file_manager_by_extension, TextFileManager, PdfFileManager
from fastapi import Depends, File, UploadFile
from io import BytesIO
from unittest.mock import MagicMock, patch
import os
import asyncio

def test_txt_file_manager_get_full_path(monkeypatch):
    MyTxtFileManager = TextFileManager()
    file_name = "test.txt"
    file_path = MyTxtFileManager._get_full_path(file_name)

  
    expected_path = os.path.join("/data/documents", "test.txt")
    assert file_path == expected_path, "Should return the correct full path for the .txt file"



def test_txt_file_manager_save_file(monkeypatch):
    MyTxtFileManager = TextFileManager()
    file_name = "test.txt"
    file_content = b"Test content"
    
    # Mock file
    file = MagicMock(spec=UploadFile)
    file.filename = file_name

    # Simulate async read method
    async def mock_read():
        return file_content

    # Simulate seek method 
    def mock_seek(position):
        pass  # Do nothing, just simulate the method

    # Assign mocked methods
    file.read.return_value = mock_read
    file.seek.return_value = mock_seek

    # Mock _get_full_path method
    monkeypatch.setattr(MyTxtFileManager, "_get_full_path", lambda x: os.path.join(".cache", x))

    # pass   with open(file_path, "wb") as f: 
    with patch("builtins.open", MagicMock()):
        # Use asyncio.run to execute the async method
        file_path = asyncio.run(MyTxtFileManager._save_file(file))

    # Check if the path is correct
    expected_path = os.path.join(".cache", "test.txt")
    assert file_path == expected_path, "Should return the correct full path for the saved .txt file"

@pytest.mark.asyncio
async def test_text_file_manager_load_split_file():
    MyTxtFileManager = TextFileManager()
    file_name = "test.txt"
    file_path = os.path.join(".cache", file_name)

    # Mock the file content
    mock_file_content = "This is a test content for the text file."
    
    # create the file
    with open(file_path, "w") as f:
        f.write(mock_file_content)

    result = await MyTxtFileManager._load_split_file(file_path)

    assert isinstance(result, list), "Should return a list of documents"
    assert len(result) > 0, "Should return a non-empty list of documents"

@pytest.mark.asyncio
async def test_text_file_manager_add_document(monkeypatch):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Create mock implementations for _save_file and _load_split_file
    async def mock_save_file(file):
        return "/mock/path/to/test.txt"  # Return a mock file path
    
    async def mock_load_split_file(file_path):
        return ["chunk1", "chunk2", "chunk3"]  # Mock the chunks from file splitting

    # Use monkeypatch to replace the methods with the mock implementations
    monkeypatch.setattr(MyTxtFileManager, "_save_file", mock_save_file)
    monkeypatch.setattr(MyTxtFileManager, "_load_split_file", mock_load_split_file)

    # Mock vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "vector_database", MagicMock())

    # Mock HTTP request using patch to completely prevent the actual request
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=201)  # Mock response with status 201
        
        # Create a mock UploadFile instance
        file = MagicMock(spec=UploadFile)
        file.filename = "test.txt"
        file.file = BytesIO(b"Test content")  # Mock the file content

        # Call the add_document method
        result = await MyTxtFileManager.add_document(file, "test_token")

        # Check if the result is True, indicating success
        assert result is True, "Should return True if the document is added successfully"

        # You can also check if the vector_database.add_documents was called correctly
        MyTxtFileManager.vector_database.add_documents.assert_called_once_with(["chunk1", "chunk2", "chunk3"])

@pytest.mark.asyncio
async def test_text_file_manager_delete_document(monkeypatch):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Mock the file deletion logic
    monkeypatch.setattr(os, "remove", MagicMock())  # Mock os.remove to avoid actual file deletion
    monkeypatch.setattr(os.path, "exists", MagicMock(return_value=True))  # Mock os.path.exists to return True

    # Mock vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "vector_database", MagicMock())

    # Mock the HTTP request using patch
    with patch("requests.delete") as mock_delete:
        mock_delete.return_value = MagicMock(status_code=200)  # Mock response with status 200
        
        # Call the delete_document method
        file_path = "/mock/path/to/test.txt"
        monkeypatch.setattr(os.path, "isfile", MagicMock(return_value=True))  # Mock os.path.isfile to return True
        result = await MyTxtFileManager.delete_document(file_path, "test_token")

        # Check if the result is True, indicating success
        assert result is None, "The document should be deleted successfully"

        # Assert that the file removal from the filesystem was attempted
        os.remove.assert_called_once_with(file_path)

        # Check that the vector_database.delete_document was called with the correct file path
        MyTxtFileManager.vector_database.delete_document.assert_called_once_with(file_path)

def test_get_file_manager(monkeypatch):
   
    # Mock the UploadFile object
    txt_File = MagicMock(spec=UploadFile)
    txt_File.content_type = "text/plain"
    txt_File.file = BytesIO(b"Test text content")

    pdf_File = MagicMock(spec=UploadFile)
    pdf_File.content_type = "application/pdf"
    pdf_File.file = BytesIO(b"Test PDF content")

    exe_File = MagicMock(spec=UploadFile)
    exe_File.content_type = "application/x-msdownload"
    exe_File.file = BytesIO(b"Test EXE content")


    tfm = get_file_manager(txt_File)
    pfm = get_file_manager(pdf_File)
    with pytest.raises(ValueError):
        get_file_manager(exe_File)
    assert isinstance(tfm, TextFileManager), "Should return an instance of TextFileManager"
    assert isinstance(pfm, PdfFileManager), "Should return an instance of PdfFileManager"


def test_get_file_manager_by_extension():
    # Test for .txt file
    file_path = "test.txt"
    file_manager = get_file_manager_by_extension(file_path)
    assert isinstance(file_manager, TextFileManager), "Should return an instance of TextFileManager"

    # Test for .pdf file
    file_path = "test.pdf"
    file_manager = get_file_manager_by_extension(file_path)
    assert isinstance(file_manager, PdfFileManager), "Should return an instance of PdfFileManager"

    # Test for unsupported file type
    with pytest.raises(ValueError):
        get_file_manager_by_extension("test.exe")