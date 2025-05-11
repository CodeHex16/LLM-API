from unittest.mock import MagicMock
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.services.vector_database_service import ChromaDB, get_vector_database
import uuid
from chromadb.errors import DuplicateIDError


def test_chromadb_initialization(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )

    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_db"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir
    assert vector_db is not None, "ChromaDB instance should be initialized"
    assert isinstance(vector_db, ChromaDB), "Expected an instance of ChromaDB class"

    # Ensure the database can be initialized before attempting to delete
    db_client = vector_db._get_db()
    assert db_client is not None, "Chroma client should be initialized"

    vector_db._delete()  # Clean up the database after test


def test_chromadb_initialization_with_persist_directory(tmp_path, monkeypatch):
    # Mock the environment variable for ChromaDB with a custom persist directory
    temp_dir = tmp_path / "hihi"
    temp_dir.mkdir()
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    vector_db = ChromaDB(str(temp_dir))
    assert vector_db is not None, "ChromaDB instance should be initialized"
    assert vector_db.persist_directory == str(
        temp_dir
    ), "Persist directory should be '{{str(temp_dir)}}'"
    assert isinstance(vector_db, ChromaDB), "Expected an instance of ChromaDB class"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_get_db(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_db_get"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir
    db_instance = vector_db._get_db()
    db_instance2 = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    assert isinstance(db_instance, Chroma), "Expected an instance of ChromaDB class"
    assert (
        db_instance is db_instance2
    ), "Should return the same instance on subsequent calls"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_generate_document_ids(tmp_path):
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_generate_ids"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir
    documents = []
    for i in range(5):
        doc = Document(
            page_content="Test content", metadata={"source": f"test_source_{i}"}
        )
        documents.append(doc)
    document_ids = vector_db._generate_document_ids(documents)
    assert len(document_ids) == len(
        documents
    ), "Should generate the same number of IDs as documents"
    for i, doc in enumerate(documents):
        assert document_ids[i] == str(
            uuid.uuid3(uuid.NAMESPACE_DNS, doc.page_content)
        ), f"ID for document {i} should match"

    vector_db._delete()  # Clean up the database after test


def test_chromadb_add_documents(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_add_documents"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir
    documents = []
    for i in range(2):
        doc = Document(
            page_content="Test content " + str(i),
            metadata={"source": f"test_source_{i}"},
        )
        documents.append(doc)
    vector_db.add_documents(documents)
    db_instance = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    assert vector_db.count() == len(
        documents
    ), "Should add the same number of documents to the database"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_add_documents_empty(monkeypatch, tmp_path):  # Add tmp_path fixture
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_add_empty"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir

    vector_db.add_documents([])
    db_instance = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    assert vector_db.count() == 0, "Should not add any documents to the database"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_add_documents_error(monkeypatch):
    # Mock the environment variable for ChromaDB
    mock_get_db = MagicMock(side_effect=Exception("Database error"))

    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    vector_db = ChromaDB()
    monkeypatch.setattr(vector_db, "_get_db", mock_get_db)
    documents = ["123", "222"]

    with pytest.raises(Exception) as excinfo:
        vector_db.add_documents(documents)


def test_chromadb_add_duplicate_document(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_add_duplicate"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir

    documents = []
    for i in range(5):
        doc = Document(
            page_content="Test content " + str(i),
            metadata={"source": f"test_source_{i}"},
        )
        documents.append(doc)
    vector_db.add_documents(documents)
    db_instance = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    with pytest.raises(DuplicateIDError):
        vector_db.add_documents([documents[0]])
    vector_db._delete()  # Clean up the database after test
    pass


def test_chromadb_delete_documents(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_delete_documents"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir

    documents = []
    for i in range(5):
        doc = Document(
            page_content="Test content " + str(i),
            metadata={"source": f"test_source_{i}"},
        )
        documents.append(doc)
    vector_db.add_documents(documents)
    db_instance = vector_db._get_db()
    assert vector_db.count() == len(
        documents
    ), "Should add the same number of documents to the database"
    vector_db._delete()  # Clean up the database after test
    assert vector_db.count() == 0, "Should delete all documents from the database"


def test_chromadb_delete_documents_exception(monkeypatch):
    # Mock the environment variable for ChromaDB
    def mock_get_db(self):
        raise Exception("Database error")

    monkeypatch.setattr(
        "app.services.vector_database_service.ChromaDB._get_db", mock_get_db
    )
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    vector_db = ChromaDB()

    with pytest.raises(Exception) as excinfo:
        vector_db.delete_document("test_source_1")


def test_chromadb_delete_documents_empty(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_delete_empty"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Initialize with temp_persist_dir

    # With a fresh temporary database, the count should initially be 0
    assert vector_db.count() == 0, "Database should be empty at the start of the test."

    vector_db.delete_document("")
    assert (
        vector_db.count() == 0
    ), "Database should remain empty after calling delete_document with an empty path on an initially empty database."

    vector_db._delete()  # Clean up the database after test


def test_chromadb_search_context(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_search_context"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Pass temp_persist_dir

    documents = []
    for i in range(5):
        doc = Document(
            page_content="Test content " + str(i),
            metadata={"source": f"test_source_{i}"},
        )
        documents.append(doc)
    vector_db.add_documents(documents)
    results = vector_db.search_context("Test content")
    assert len(results) > 0, "Should return search results"
    assert isinstance(
        results[0], Document
    ), "Search result should be a Document instance"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_search_context_none(monkeypatch):
    # Mock the environment variable for ChromaDB
    def mock_get_db(self):
        raise Exception("Database error")

    monkeypatch.setattr(
        "app.services.vector_database_service.ChromaDB._get_db", mock_get_db
    )
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    vector_db = ChromaDB()

    result = vector_db.search_context("test_source_1")
    print("result", result)
    assert (
        result == []
    ), "Should not return any results when the database is not available"


def test_chromadb_search_context_empty(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_search_empty"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Initialize with temp_persist_dir

    results = vector_db.search_context("Non-existent content")
    assert (
        len(results) == 0
    ), "Should return no search results for non-existent content in an empty database"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_is_empty(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_is_empty"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(persist_directory=str(temp_persist_dir))
    assert vector_db.is_empty() is True, "Database should be empty initially"
    documents = []
    for i in range(5):
        doc = Document(
            page_content="Test content " + str(i),
            metadata={"source": f"test_source_{i}"},
        )
        documents.append(doc)
    vector_db.add_documents(documents)
    assert (
        vector_db.is_empty() is False
    ), "Database should not be empty after adding documents"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_count(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_count"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Initialize with temp_persist_dir
    documents = []
    for i in range(5):
        doc = Document(
            page_content="Test content" + str(i),
            metadata={"source": f"test_source_{i}"},
        )
        documents.append(doc)
    vector_db.add_documents(documents)
    count = vector_db.count()
    assert count == len(
        documents
    ), "Should return the same number of documents in the collection"
    vector_db._delete()  # Clean up the database after test


def test_chromadb_delete(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_delete"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(
        persist_directory=str(temp_persist_dir)
    )  # Initialize with temp_persist_dir
    documents = []
    for i in range(5):
        doc = Document(
            page_content="Test content" + str(i),
            metadata={"source": f"test_source_{i}"},
        )
        documents.append(doc)
    vector_db.add_documents(documents)
    db_instance = vector_db._get_db()
    assert vector_db.count() == len(
        documents
    ), "Should add the same number of documents to the database"
    vector_db._delete()  # Clean up the database after test
    assert vector_db.is_empty() is True, "Database should be empty after deletion"


def test_get_vector_database_chromadb(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )
    # Create a temporary directory for ChromaDB persistence for this test
    temp_persist_dir = tmp_path / "chroma_test_get_vector_db"
    temp_persist_dir.mkdir()
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_DIRECTORY",
        str(temp_persist_dir),
    )

    vector_db = get_vector_database()
    assert isinstance(vector_db, ChromaDB), "Expected an instance of ChromaDB class"
    # Ensure the _delete operation can write to the temporary directory
    try:
        vector_db._delete()  # Clean up the database after test
    except Exception as e:
        pytest.fail(f"_delete failed with temporary directory: {e}")


def test_get_vector_database_invalid_provider(monkeypatch):
    # Mock the environment variable for an invalid provider
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER",
        "invalid_provider",
    )
    with pytest.raises(ValueError):
        get_vector_database()


def test_get_collection_count_with_client_none(monkeypatch, tmp_path):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr(
        "app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma"
    )

    # Create a temporary directory for ChromaDB persistence
    temp_persist_dir = tmp_path / "chroma_test_collection_count"
    temp_persist_dir.mkdir()

    vector_db = ChromaDB(persist_directory=str(temp_persist_dir)) # Initialize with temp_persist_dir
    
    fake_db_client = MagicMock()
    # Simulate that the collection is not available or accessible on the client
    fake_db_client._collection = None 

    monkeypatch.setattr(vector_db, "_db", fake_db_client) # vector_db._get_db() will now return fake_db_client

    count = vector_db._get_collection_count()
    assert count == 0, "Should return 0 when collection is not available via the client"