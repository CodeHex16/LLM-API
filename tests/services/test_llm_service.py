import pytest

from app.services.llm_service import LLM, OpenAI, Ollama, get_llm_model

def test_openai_initialization(monkeypatch):
    # Mock the environment variable for OpenAI API key
    monkeypatch.setattr("app.services.llm_service.settings.LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    llm = OpenAI(model_name="gpt-4")
    assert llm.model_name == "gpt-4", "Model name should be 'gpt-4'"
    assert llm.model is not None, "Model should be initialized"
    assert isinstance(llm, OpenAI), "Expected an instance of OpenAI class"

def test_ollama_initialization(monkeypatch):
    pass

def test_get_llm_model_openai(monkeypatch):
    # Mock the environment variable for OpenAI
    monkeypatch.setattr("app.services.llm_service.settings.LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    llm = get_llm_model()
    assert isinstance(llm, OpenAI), "Expected an instance of OpenAI class"

def test_get_llm_model_ollama(monkeypatch):
    # Mock the environment variable for Ollama
    monkeypatch.setattr("app.services.llm_service.settings.LLM_PROVIDER", "ollama")
    llm = get_llm_model()
    assert isinstance(llm, Ollama), "Expected an instance of Ollama class"

def test_get_llm_model_invalid_provider(monkeypatch):
    # Mock the environment variable for an invalid provider
    monkeypatch.setattr("app.services.llm_service.settings.LLM_PROVIDER", "invalid_provider")
    with pytest.raises(ValueError):
        get_llm_model()
