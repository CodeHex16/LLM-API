import pytest
from app.services.file_manager_service import (
    get_file_manager,
    get_file_manager_by_extension,
    TextFileManager,
    PdfFileManager,
)
from fastapi import Depends, File, UploadFile
from io import BytesIO
from unittest.mock import MagicMock, patch
import os
import asyncio
from fastapi import HTTPException
import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader


@pytest.fixture(autouse=True)
def documents_dir(tmp_path, monkeypatch):
    # Create a temporary directory for the test
    temp_dir = tmp_path / "documents"
    temp_dir.mkdir()

    monkeypatch.setenv("DOCUMENTS_DIR", str(temp_dir))
    # Return the path to the temporary directory
    yield str(temp_dir)
    # Cleanup is handled by pytest's tmp_path fixture


def test_txt_file_manager_get_full_path(documents_dir, monkeypatch):
    MyTxtFileManager = TextFileManager()
    file_name = "test.txt"
    file_path = MyTxtFileManager.get_full_path(file_name)
    expected_path = os.path.join(documents_dir, "test.txt")
    assert (
        file_path == expected_path
    ), "Should return the correct full path for the .txt file"


def test_txt_file_manager_save_file(monkeypatch):
    MyTxtFileManager = TextFileManager()
    file_name = "test.txt"
    file_content = b"Test content"

    # Mock file
    file = MagicMock(spec=UploadFile)
    file.filename = file_name

    # Assign mocked methods
    file.read.return_value = file_content
    file.seek.return_value = MagicMock(side_effect=None)  # Mock seek method

    # Mock _get_full_path method
    monkeypatch.setattr(
        MyTxtFileManager, "get_full_path", lambda x: os.path.join(".cache", x)
    )

    # pass   with open(file_path, "wb") as f:
    with patch("builtins.open", MagicMock()):
        # Use asyncio.run to execute the async method
        file_path = asyncio.run(MyTxtFileManager._save_file(file))

    # Check if the path is correct
    expected_path = os.path.join(".cache", "test.txt")
    assert (
        file_path == expected_path
    ), "Should return the correct full path for the saved .txt file"


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
        return "/mock/path/to/test.txt"

    async def mock_load_split_file(file_path):
        return ["chunk1", "chunk2", "chunk3"]

    monkeypatch.setattr(MyTxtFileManager, "_save_file", mock_save_file)
    monkeypatch.setattr(MyTxtFileManager, "_load_split_file", mock_load_split_file)

    # CORRECTED: Mock the _vector_database attribute
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=201)

        file = MagicMock(spec=UploadFile)
        file.filename = "test.txt"
        file.file = BytesIO(b"Test content")

        result = await MyTxtFileManager.add_document(file, "test_token")

        assert result is True
        MyTxtFileManager._vector_database.add_documents.assert_called_once_with(
            ["chunk1", "chunk2", "chunk3"]
        )


@pytest.mark.asyncio
async def test_text_file_manager_add_document_error_400(monkeypatch):
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

    # Mock _vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Mock HTTP request using patch to completely prevent the actual request
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=400
        )  # Mock response with status 201

        # Create a mock UploadFile instance
        file = MagicMock(spec=UploadFile)
        file.filename = "test.txt"
        file.file = BytesIO(b"Test content")  # Mock the file content

        # Call the add_document method
        with pytest.raises(HTTPException) as exc_info:
            result = await MyTxtFileManager.add_document(file, "test_token")
        assert (
            exc_info.value.status_code == 400
        ), "Should raise HTTPException with status code 400"
        assert (
            exc_info.value.detail == "Documento già esistente"
        ), "Should raise HTTPException with the correct detail"


@pytest.mark.asyncio
async def test_text_file_manager_add_document_error_500(monkeypatch):
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

    # Mock _vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Mock HTTP request using patch to completely prevent the actual request
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=500
        )  # Mock response with status 201

        # Create a mock UploadFile instance
        file = MagicMock(spec=UploadFile)
        file.filename = "test.txt"
        file.file = BytesIO(b"Test content")  # Mock the file content

        # Call the add_document method
        with pytest.raises(HTTPException) as exc_info:
            result = await MyTxtFileManager.add_document(file, "test_token")
        assert (
            exc_info.value.status_code == 500
        ), "Should raise HTTPException with status code 400"
        assert (
            exc_info.value.detail == "Errore nel caricare e processare file"
        ), "Should raise HTTPException with the correct detail"


@pytest.mark.asyncio
async def test_text_file_manager_add_document_error_defalut(monkeypatch):
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

    # Mock _vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Mock HTTP request using patch to completely prevent the actual request
    with patch("requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=501
        )  # Mock response with status 201

        # Create a mock UploadFile instance
        file = MagicMock(spec=UploadFile)
        file.filename = "test.txt"
        file.file = BytesIO(b"Test content")  # Mock the file content

        # Call the add_document method
        with pytest.raises(HTTPException) as exc_info:
            result = await MyTxtFileManager.add_document(file, "test_token")
        assert (
            exc_info.value.status_code == 500
        ), "Should raise HTTPException with status code 400"
        assert (
            exc_info.value.detail == "Errore nel caricare e processare file"
        ), "Should raise HTTPException with the correct detail"


@pytest.mark.asyncio
async def test_delete_document_os_remove_error(monkeypatch):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Mock os level functions for the filesystem interaction part
    mock_os_error_message = "Simulated OS error during file removal"
    mock_os_error = OSError(mock_os_error_message)
    monkeypatch.setattr(os, "remove", MagicMock(side_effect=mock_os_error))
    monkeypatch.setattr(
        os.path, "exists", MagicMock(return_value=True)
    )  # Assume file exists
    monkeypatch.setattr(
        os.path, "isfile", MagicMock(return_value=True)
    )  # Assume it's a file

    # Mock _vector_database as it might be called if other parts succeed
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Mock the requests.delete call to the Database API
    # This is crucial: ensure this call is mocked to return a success (204)
    # so that the code proceeds to the os.remove part.
    with patch("requests.delete") as mock_requests_delete:
        mock_requests_delete.return_value = MagicMock(
            status_code=204
        )  # Simulate DB API success

        file_path = "/mock/path/to/test.txt"

        with pytest.raises(HTTPException) as exc_info:
            await MyTxtFileManager.delete_document(
                "idid", file_path, "test_token", "pwdpwd"
            )

    assert (
        exc_info.value.status_code == 404
    ), "Should raise HTTPException with status code 404 when os.remove fails"

    # Check that the detail message includes the original OSError message
    expected_detail = f"File {file_path} non trovato: {mock_os_error_message}"
    assert (
        exc_info.value.detail == expected_detail
    ), "Detail should include the original OS error message"


@pytest.mark.asyncio
async def test_text_file_manager_delete_document(monkeypatch):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Mock the file deletion logic
    monkeypatch.setattr(
        os, "remove", MagicMock()
    )  # Mock os.remove to avoid actual file deletion
    monkeypatch.setattr(
        os.path, "exists", MagicMock(return_value=True)
    )  # Mock os.path.exists to return True

    # Mock _vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Mock the HTTP request using patch
    with patch("requests.delete") as mock_delete:
        mock_delete.return_value = MagicMock(
            status_code=204
        )  # Mock response with status 200

        # Call the delete_document method
        file_path = "/mock/path/to/test.txt"
        monkeypatch.setattr(
            os.path, "isfile", MagicMock(return_value=True)
        )  # Mock os.path.isfile to return True
        result = await MyTxtFileManager.delete_document(
            "idid", file_path, "test_token", "pwdpwd"
        )

        # Check if the result is True, indicating success
        assert result is None, "The document should be deleted successfully"

        # Assert that the file removal from the filesystem was attempted
        os.remove.assert_called_once_with(file_path)

        # Check that the _vector_database.delete_document was called with the correct file path
        MyTxtFileManager._vector_database.delete_document.assert_called_once_with(
            file_path
        )


@pytest.mark.asyncio
async def test_text_file_manager_delete_document_not_found(monkeypatch):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Mock the file deletion logic
    monkeypatch.setattr(
        os, "remove", MagicMock()
    )  # Mock os.remove to avoid actual file deletion
    monkeypatch.setattr(
        os.path, "exists", MagicMock(return_value=True)
    )  # Mock os.path.exists to return True

    # Mock _vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Mock the HTTP request using patch
    with patch("requests.delete") as mock_delete:
        mock_delete.return_value = MagicMock(
            status_code=404
        )  # Mock response with status 200

        # Call the delete_document method
        file_path = "/mock/path/to/test.txt"
        monkeypatch.setattr(
            os.path, "isfile", MagicMock(return_value=True)
        )  # Mock os.path.isfile to return True
        with pytest.raises(HTTPException) as exc_info:
            result = await MyTxtFileManager.delete_document(
                "idid", file_path, "test_token", "pwdpwd"
            )
        assert (
            exc_info.value.status_code == 404
        ), "Should raise HTTPException with status code 404"
        assert (
            exc_info.value.detail == "Documento non trovato"
        ), "Should raise HTTPException with the correct detail"


@pytest.mark.asyncio
async def test_text_file_manager_delete_document_500_exception(monkeypatch):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Mock os level functions as they are called before the HTTP error if DB call was successful
    monkeypatch.setattr(os, "remove", MagicMock())
    monkeypatch.setattr(os.path, "exists", MagicMock(return_value=True))
    monkeypatch.setattr(os.path, "isfile", MagicMock(return_value=True))

    # Mock _vector_database as its methods might be called if other parts succeed
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Define the specific text we expect delete_req.text to be
    expected_api_error_text = "Database server encountered an error."

    with patch("requests.delete") as mock_delete_request:
        # This mock_response object will be 'delete_req' in your service code
        mock_response = MagicMock()
        mock_response.status_code = 500  # To hit the 'case 500:'
        # Set the .text attribute on the mock_response
        mock_response.text = expected_api_error_text

        mock_delete_request.return_value = mock_response

        file_path = "/mock/path/to/test.txt"

        with pytest.raises(HTTPException) as exc_info:
            await MyTxtFileManager.delete_document(
                "idid", file_path, "test_token", "pwdpwd"
            )

        assert (
            exc_info.value.status_code == 500
        ), "Should raise HTTPException with status code 500"

        # Construct the expected detail string using the same specific text
        expected_detail_message = (
            f"Errore nel caricare e processare file {expected_api_error_text}"
        )
        assert (
            exc_info.value.detail == expected_detail_message
        ), "Should raise HTTPException with the correct detail including the API error text"


@pytest.mark.asyncio
async def test_text_file_manager_delete_document_default_exception(
    documents_dir, monkeypatch
):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Mock the file deletion logic
    monkeypatch.setattr(
        os, "remove", MagicMock(return_value=True)
    )  # Mock os.remove to avoid actual file deletion
    monkeypatch.setattr(
        os.path, "exists", MagicMock(return_value=True)
    )  # Mock os.path.exists to return True

    # Mock _vector_database to avoid interaction with the actual database
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())
    # Mock the HTTP request using patch
    with patch("requests.delete") as mock_delete:
        mock_delete.return_value = MagicMock(
            status_code=501
        )  # Mock response with status 500

        # Call the delete_document method
        file_path = "/mock/path/to/test.txt"
        monkeypatch.setattr(
            os.path, "isfile", MagicMock(return_value=True)
        )  # Mock os.path.isfile to return True
        with pytest.raises(HTTPException) as exc_info:
            result = await MyTxtFileManager.delete_document(
                "idid", file_path, "test_token", "pwdpwd"
            )
        assert (
            exc_info.value.status_code == 500
        ), "Should raise HTTPException with status code 500"
        assert (
            exc_info.value.detail == "Errore nel caricare e processare file"
        ), "Should raise HTTPException with the correct detail"


@pytest.mark.asyncio
async def test_text_file_manager_delete_path_not_found(monkeypatch):
    # Create an instance of TextFileManager
    MyTxtFileManager = TextFileManager()

    # Mock os level functions for the filesystem interaction part
    monkeypatch.setattr(
        os, "remove", MagicMock()
    )  # Mock os.remove as it might be called if logic changes
    monkeypatch.setattr(
        os.path, "exists", MagicMock(return_value=False)
    )  # Simulate file not existing
    monkeypatch.setattr(
        os.path,
        "isfile",
        MagicMock(
            return_value=False
        ),  # if exists is False, isfile should also logically be False or not matter
    )

    # Mock _vector_database as it's not relevant for this specific failure path
    monkeypatch.setattr(MyTxtFileManager, "_vector_database", MagicMock())

    # Mock the requests.delete call to the Database API
    # This is crucial: ensure this call is mocked to return a success (204)
    # so that the code proceeds to the filesystem check part.
    with patch("requests.delete") as mock_requests_delete:
        mock_requests_delete.return_value = MagicMock(
            status_code=204
        )  # Simulate DB API success

        file_path = "/mock/path/to/test.txt"

        with pytest.raises(HTTPException) as exc_info:
            await MyTxtFileManager.delete_document(
                "idid", file_path, "test_token", "pwdpwd"
            )

    assert (
        exc_info.value.status_code == 404
    ), "Should raise HTTPException with status code 404 when path does not exist"
    assert (
        exc_info.value.detail == f"File {file_path} non trovato"
    ), "Should raise HTTPException with the correct detail for non-existent path"


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
    assert isinstance(
        tfm, TextFileManager
    ), "Should return an instance of TextFileManager"
    assert isinstance(
        pfm, PdfFileManager
    ), "Should return an instance of PdfFileManager"


def test_get_file_manager_by_extension():
    # Test for .txt file
    file_path = "test.txt"
    file_manager = get_file_manager_by_extension(file_path)
    assert isinstance(
        file_manager, TextFileManager
    ), "Should return an instance of TextFileManager"

    # Test for .pdf file
    file_path = "test.pdf"
    file_manager = get_file_manager_by_extension(file_path)
    assert isinstance(
        file_manager, PdfFileManager
    ), "Should return an instance of PdfFileManager"

    # Test for unsupported file type
    with pytest.raises(ValueError):
        get_file_manager_by_extension("test.exe")


@pytest.mark.asyncio
async def test_pdf_file_manager_load_split_file(monkeypatch):
    MyPdfFileManager = PdfFileManager()
    FakePdfFileManager = MagicMock()
    FakePdfFileManager.load = MagicMock(return_value=["chunk1", "chunk2", "chunk3"])

    FakeSplitter = MagicMock()
    FakeSplitter.split_documents = MagicMock(
        return_value=["chunk1", "chunk2", "chunk3"]
    )
    monkeypatch.setattr(MyPdfFileManager, "_splitter", FakeSplitter)
    # mock langchain_community.document_loaders.PyPDFLoader
    monkeypatch.setattr(
        "app.services.file_manager_service.PyPDFLoader", FakePdfFileManager
    )
    await MyPdfFileManager._load_split_file("test.pdf")
    assert (
        FakeSplitter.split_documents.called
    ), "Should call the split_documents method of the _splitter"
    assert isinstance(
        FakeSplitter.split_documents.return_value, list
    ), "Should return a list of documents"
