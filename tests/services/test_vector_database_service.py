from unittest.mock import MagicMock
import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.services.vector_database_service import ChromaDB, get_vector_database
import uuid
from chromadb.errors import DuplicateIDError


def test_chromadb_initialization(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    assert vector_db is not None, "ChromaDB instance should be initialized"
    assert isinstance(vector_db, ChromaDB), "Expected an instance of ChromaDB class"
    vector_db._delete() # Clean up the database after test

def test_chromadb_initialization_with_persist_directory(monkeypatch):
    # Mock the environment variable for ChromaDB with a custom persist directory
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB("/custom/path")
    assert vector_db is not None, "ChromaDB instance should be initialized"
    assert vector_db.persist_directory == "/custom/path", "Persist directory should be '/custom/path'"
    assert isinstance(vector_db, ChromaDB), "Expected an instance of ChromaDB class"
    vector_db._delete() # Clean up the database after test

def test_chromadb_get_db(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    db_instance = vector_db._get_db()
    db_instance2 = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    assert isinstance(db_instance, Chroma), "Expected an instance of ChromaDB class"
    assert db_instance is db_instance2, "Should return the same instance on subsequent calls"
    vector_db._delete() # Clean up the database after test

def test_chromadb_generate_document_ids():
    vector_db = ChromaDB()
    documents = []
    for i in range(5):
        doc = Document(page_content="Test content", metadata={"source": f"test_source_{i}"})
        documents.append(doc)
    document_ids = vector_db._generate_document_ids(documents)
    assert len(document_ids) == len(documents), "Should generate the same number of IDs as documents"
    for i, doc in enumerate(documents):
        assert document_ids[i] == str(uuid.uuid3(uuid.NAMESPACE_DNS, doc.page_content)), f"ID for document {i} should match"
    
    vector_db._delete() # Clean up the database after test

def test_chromadb_add_documents(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    documents = []
    for i in range(2):
        doc = Document(page_content="Test content " + str(i), metadata={"source": f"test_source_{i}"})
        documents.append(doc)
    vector_db.add_documents(documents)
    db_instance = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    assert vector_db.count() == len(documents), "Should add the same number of documents to the database"
    vector_db._delete() # Clean up the database after test

def test_chromadb_add_documents_error(monkeypatch):
    # Mock the environment variable for ChromaDB
    def mock_get_db(self):
        raise Exception("Database error")
   
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    monkeypatch.setattr(vector_db, "_get_db", mock_get_db)
    documents = ["123","222"]
 
    with pytest.raises(Exception) as excinfo:
        vector_db.add_documents(documents)

def test_chromadb_add_documents_empty(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    
    vector_db.add_documents([])
    db_instance = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    assert  vector_db.count() == 0, "Should not add any documents to the database"
    vector_db._delete() # Clean up the database after test

def test_chromadb_add_duplicate_document(monkeypatch):
    # Mock the environment variable for ChromaDB
    # monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    # vector_db = ChromaDB()
    
    # documents = []
    # for i in range(5):
    #     doc = Document(page_content="Test content " + str(i), metadata={"source": f"test_source_{i}"})
    #     documents.append(doc)
    # vector_db.add_documents(documents)
    # db_instance = vector_db._get_db()
    # assert db_instance is not None, "ChromaDB instance should be initialized"
    # with pytest.raises(DuplicateIDError):
    #     vector_db.add_documents([documents[0]])
    # vector_db._delete() # Clean up the database after test
    pass

def test_chromadb_delete_documents(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    
    documents = []
    for i in range(5):
        doc = Document(page_content="Test content " + str(i), metadata={"source": f"test_source_{i}"})
        documents.append(doc)
    vector_db.add_documents(documents)
    db_instance = vector_db._get_db()
    assert  vector_db.count()== len(documents), "Should add the same number of documents to the database"
    vector_db._delete() # Clean up the database after test
    assert  vector_db.count()== 0, "Should delete all documents from the database"
    

def test_chromadb_delete_documents(monkeypatch):
    # Mock the environment variable for ChromaDB
    def mock_get_db(self):
        raise Exception("Database error")
    monkeypatch.setattr("app.services.vector_database_service.ChromaDB._get_db", mock_get_db)
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()

    with pytest.raises(Exception) as excinfo:
        vector_db.delete_document("test_source_1")

def test_chromadb_delete_documents_empty(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    
    vector_db.delete_document("")
    db_instance = vector_db._get_db()
    assert db_instance is not None, "ChromaDB instance should be initialized"
    assert  vector_db.count()== 0, "Should not delete any documents from the database"
    vector_db._delete() # Clean up the database after test

def test_chromadb_search_context(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    
    documents = []
    for i in range(5):
        doc = Document(page_content="Test content " + str(i), metadata={"source": f"test_source_{i}"})
        documents.append(doc)
    vector_db.add_documents(documents)
    results = vector_db.search_context("Test content")
    assert len(results) > 0, "Should return search results"
    assert isinstance(results[0], Document), "Search result should be a Document instance"
    vector_db._delete() # Clean up the database after test

def test_chromadb_search_context_none(monkeypatch):
    # Mock the environment variable for ChromaDB
    def mock_get_db(self):
        raise Exception("Database error")
    monkeypatch.setattr("app.services.vector_database_service.ChromaDB._get_db", mock_get_db)
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()


    result = vector_db.search_context("test_source_1")
    print("result",result)
    assert result == [], "Should not return any results when the database is not available"

def test_chromadb_search_context_empty(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    
    results = vector_db.search_context("Non-existent content")
    assert len(results) == 0, "Should return no search results for non-existent content"
    vector_db._delete() # Clean up the database after test

def test_chromadb_is_empty(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    assert vector_db.is_empty() is True, "Database should be empty initially"
    documents = []
    for i in range(5):
        doc = Document(page_content="Test content " +str(i), metadata={"source": f"test_source_{i}"})
        documents.append(doc)
    vector_db.add_documents(documents)
    assert vector_db.is_empty() is False, "Database should not be empty after adding documents"
    vector_db._delete() # Clean up the database after test

def test_chromadb_count(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    documents = []
    for i in range(5):
        doc = Document(page_content="Test content" + str(i), metadata={"source": f"test_source_{i}"})
        documents.append(doc)
    vector_db.add_documents(documents)
    count = vector_db.count()
    assert count == len(documents), "Should return the same number of documents in the collection"
    vector_db._delete() # Clean up the database after test

def test_chromadb_delete(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = ChromaDB()
    documents = []
    for i in range(5):
        doc = Document(page_content="Test content" + str(i), metadata={"source": f"test_source_{i}"})
        documents.append(doc)
    vector_db.add_documents(documents)
    db_instance = vector_db._get_db()
    assert  vector_db.count()== len(documents), "Should add the same number of documents to the database"
    vector_db._delete() # Clean up the database after test
    assert vector_db.is_empty() is True, "Database should be empty after deletion"

def test_get_vector_database_chromadb(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    vector_db = get_vector_database()
    assert isinstance(vector_db, ChromaDB), "Expected an instance of ChromaDB class"
    vector_db._delete() # Clean up the database after test

def test_get_vector_database_invalid_provider(monkeypatch):
    # Mock the environment variable for an invalid provider
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "invalid_provider")
    with pytest.raises(ValueError):
        get_vector_database()

def test_get_collection_count_with_client_none(monkeypatch):
    # Mock the environment variable for ChromaDB
    monkeypatch.setattr("app.services.vector_database_service.settings.VECTOR_DB_PROVIDER", "chroma")
    fake_db = MagicMock()
    fake_db.get.return_value = None
    
    vector_db = ChromaDB()
    monkeypatch.setattr(vector_db,"_db", fake_db)

    count = vector_db._get_collection_count()
    assert count == 0, "Should return 0 for empty collection"