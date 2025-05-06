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

def test_openai_check_environment_empty(monkeypatch):
    # Mock the environment variable for OpenAI API key
    MyOpenAi = OpenAI(model_name="gpt-4")
    monkeypatch.setenv("OPENAI_API_KEY", "hihihi")
    MyOpenAi._check_environment()


def test_openai_check_environment_empty(monkeypatch):
    # Mock the environment variable for OpenAI API key
    MyOpenAi = OpenAI(model_name="gpt-4")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValueError) as excinfo:
        MyOpenAi._check_environment()
    assert str(excinfo.value) == "API key mancante per OpenAI", "Should raise ValueError for missing API key"

def test_openai_inizialization_model_with_no_model(monkeypatch):
    MyOpenAi = OpenAI(model_name="gpt-4")
    def fake_init_chat_model(model, model_provider):
        assert model == "gpt-4", "Model name should be 'gpt-4'"
        return None # Simulate a failure in model initialization

    monkeypatch.setattr("app.services.llm_service.init_chat_model", fake_init_chat_model)
    with pytest.raises(ValueError) as excinfo:
        MyOpenAi._initialize_model()
    assert str(excinfo.value) == "Invalid or unavailable model: gpt-4", "Should raise ValueError for invalid model"


def test_openai_inizialization_model_exception(monkeypatch):
    MyOpenAi = OpenAI(model_name="gpt-4")

    monkeypatch.setattr("app.services.llm_service.init_chat_model", lambda model, model_provider: 1/0) # Simulate a failure in model initialization
    with pytest.raises(ValueError) as excinfo:
        MyOpenAi._initialize_model()
    assert str(excinfo.value) == "Invalid or unavailable model: gpt-4", "Should raise ValueError for invalid model"

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

def test_ollama_inizialization_model_with_no_model(monkeypatch):
    MyOllama = Ollama(model_name="ollama")
    def fake_init_chat_model(model, model_provider):
        assert model == "ollama", "Model name should be 'ollama'"
        return None # Simulate a failure in model initialization

    monkeypatch.setattr("app.services.llm_service.init_chat_model", fake_init_chat_model)
    with pytest.raises(ValueError) as excinfo:
        MyOllama._initialize_model()
    assert str(excinfo.value) == "Invalid or unavailable model: ollama", "Should raise ValueError for invalid model"


def test_ollama_inizialization_model_exception(monkeypatch):
    MyOllama = Ollama(model_name="ollama")

    monkeypatch.setattr("app.services.llm_service.init_chat_model", lambda model, model_provider: 1/0) # Simulate a failure in model initialization
    with pytest.raises(ValueError) as excinfo:
        MyOllama._initialize_model()
    assert str(excinfo.value) == "Invalid or unavailable model: ollama", "Should raise ValueError for invalid model"


def test_get_llm_model_invalid_provider(monkeypatch):
    # Mock the environment variable for an invalid provider
    monkeypatch.setattr("app.services.llm_service.settings.LLM_PROVIDER", "invalid_provider")
    with pytest.raises(ValueError):
        get_llm_model()
