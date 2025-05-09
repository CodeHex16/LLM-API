import pytest

from app.services.embeddings_service import OpenAIEmbeddingProvider,EmbeddingProvider, get_embedding_provider
from langchain_openai import OpenAIEmbeddings


def test_openai_embedding_provider(monkeypatch):
    embedding_provider = OpenAIEmbeddingProvider("test_api_key", "text-embedding-ada-002")
    assert isinstance(embedding_provider, OpenAIEmbeddingProvider), "Should return an instance of OpenAIEmbeddingProvider"
    assert embedding_provider.api_key == "test_api_key", "Should use the mocked API key"
    assert embedding_provider.model_name == "text-embedding-ada-002", "Should use the mocked model name"

def test_openai_embedding_function(monkeypatch):
    embedding_provider = OpenAIEmbeddingProvider()
    embedding_function = embedding_provider.get_embedding_function()
    embedding_function2 = embedding_provider.get_embedding_function()
    
    assert embedding_function is not None, "Should return a valid embedding function"
    assert isinstance(embedding_function, OpenAIEmbeddings), "Should return an instance of OpenAIEmbeddings"
    assert embedding_function == embedding_function2, "Should return the same instance of the embedding function"

def test_openai_embedding_function_no_api_key(monkeypatch):
    embedding_provider = OpenAIEmbeddingProvider(None, "text-embedding-ada-002")
    monkeypatch.setattr("os.environ", {"OPENAI_API_KEY": None})
    with pytest.raises(ValueError):
        embedding_provider.get_embedding_function()

def test_get_embeddings_provider(monkeypatch):
    # Mock the environment variable for embedding provider
    monkeypatch.setattr("app.services.embeddings_service.settings.LLM_PROVIDER", "openai")
    
    embedding_provider = get_embedding_provider()
    assert isinstance(embedding_provider, OpenAIEmbeddingProvider), "Should return an instance of EmbeddingsService"
    
def test_get_embedding_provider_invalid(monkeypatch):
    # Mock the environment variable for an invalid provider
    monkeypatch.setattr("app.services.embeddings_service.settings.LLM_PROVIDER", "invalid_provider")
    
    with pytest.raises(ValueError):
        get_embedding_provider()