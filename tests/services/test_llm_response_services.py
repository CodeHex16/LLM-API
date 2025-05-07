import pytest
from app.services.llm_response_service import LLMResponseService, get_llm_response_service
from app.services.vector_database_service import VectorDatabase, get_vector_database
from app.services.llm_service import LLM, get_llm_model
from app.schemas import Question, Message
from starlette.responses import StreamingResponse
from unittest.mock import MagicMock
from fastapi import HTTPException

def test_llm_response_service_initialization():
    llm_response_service = LLMResponseService()
    assert isinstance(llm_response_service, LLMResponseService), "Should return an instance of LLMResponseService"
    assert isinstance(llm_response_service.LLM, LLM), "LLM should be an instance of LLM"
    assert isinstance(llm_response_service.vector_database, VectorDatabase), "Vector database should be an instance of VectorDatabase"

def test_llm_response_service_get_context(monkeypatch):
    llm_response_service = LLMResponseService()
    question = "What is the capital of France?"
    
    # Mock the vector database search_context method
    mock_context = "Paris is the capital of France."
    monkeypatch.setattr(llm_response_service.vector_database, "search_context", lambda q: mock_context)
    
    context = llm_response_service._get_context(question)
    assert context == mock_context, "Should return the mocked context"

def test_llm_response_service_get_context_false(monkeypatch):
    llm_response_service = LLMResponseService()
    question = "What is the capital of France?"
    
    # Mock the vector database search_context method
    mock_context = "Paris is the capital of France."
    monkeypatch.setattr(llm_response_service.vector_database, "search_context", lambda q: None)
    
    with pytest.raises(HTTPException) as excinfo:
        llm_response_service._get_context(question)
    assert excinfo.value.status_code == 500, "Should raise HTTPException with status code 500"
    assert str(excinfo.value.detail).startswith("Error getting context:"), "Should raise HTTPException with the correct error message"

def test_llm_response_service_get_context_exception(monkeypatch):
    llm_response_service = LLMResponseService()
    question = "What is the capital of France?"
    
    mock_search_context = MagicMock(side_effect=Exception("Database error"))
    monkeypatch.setattr(llm_response_service.vector_database, "search_context", mock_search_context)

    with pytest.raises(HTTPException) as excinfo:
        llm_response_service._get_context(question)
    assert excinfo.value.status_code == 500, "Should raise HTTPException with status code 500"
    assert str(excinfo.value.detail).startswith("Error getting context:"), "Should raise HTTPException with the correct error message"

def test_get_llm_response_service():
    llm_response_service = get_llm_response_service()
    assert isinstance(llm_response_service, LLMResponseService), "Should return an instance of LLMResponseService"

def test_generate_llm_chat_name(monkeypatch):
    response = MagicMock()
    response.content = "mocked response content"
    

    llm_response_service = LLMResponseService()
    llm_response_service.LLM = MagicMock()
    llm_response_service.LLM.model = MagicMock()

    llm_response_service._get_context = lambda question: "mocked context"
    llm_response_service.LLM.model.invoke = response

    llm_response_service.generate_llm_chat_name("mocked question")
    assert response.content == "mocked response content", "Should return the mocked response content"

def test_generate_llm_chat_name_exception(monkeypatch):
    response = MagicMock()
    def throw_exception(*args, **kwargs):
        raise Exception("Mocked exception")
    response.invoke = throw_exception


    llm_response_service = LLMResponseService()
    llm_response_service._get_context = lambda question: "mocked context"
    llm_response_service.LLM.model = response

    with pytest.raises(HTTPException) as excinfo:
        llm_response_service.generate_llm_chat_name("mocked question")
        assert excinfo.value.status_code == 500, "Should raise HTTPException with status code 500"


@pytest.mark.asyncio
async def test_generate_llm_response_with_messages_list(monkeypatch):
    def mock_search_context(question): return "Mocked context"
    class mockChunk:
        def __init__(self, content):
            self.content = content

    async def mock_astream(messages):
        for chunk in ["chunk1", "chunk2", {"content":"chunk3"}, mockChunk("chunk4"),""]:  # 模拟异步迭代
            yield chunk

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM.model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(question="Voglio pizza!", messages=[Message(sender="me", content="Hi")])
    monkeypatch.setattr(service, "LLM", mock_LLM)

    result = service.generate_llm_response(question)
    assert isinstance(result, StreamingResponse), "Should be a StreamingResponse instance"
    async for chunk in result.body_iterator:
        assert chunk.startswith("data:"), "Chunk should start with 'data:'"


@pytest.mark.asyncio
async def test_generate_llm_response_with_messages_list(monkeypatch):
    def mock_search_context(question): return "Mocked context"
    class mockChunk:
        def __init__(self, content):
            self.content = content

    async def mock_astream(messages):
        for chunk in ["chunk1", "chunk2", {"content":"chunk3"}, mockChunk("chunk4"),""]:  # 模拟异步迭代
            yield chunk

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM.model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(question="Voglio pizza!", messages=[Message(sender="me", content="Hi")])
    monkeypatch.setattr(service, "LLM", mock_LLM)

    result = service.generate_llm_response(question)
    assert isinstance(result, StreamingResponse), "Should be a StreamingResponse instance"
    async for chunk in result.body_iterator:
        assert chunk.startswith("data:"), "Chunk should start with 'data:'"

@pytest.mark.asyncio
async def test_generate_llm_response_with_messages_list(monkeypatch):
    def mock_search_context(question): return "Mocked context"
    class mockChunk:
        def __init__(self, content):
            self.content = content

    async def mock_astream(messages):
        for chunk in ["chunk1", "chunk2", {"content":"chunk3"}, mockChunk("chunk4"),""]:  # 模拟异步迭代
            yield chunk

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM.model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(question="Voglio pizza!", messages=[])
    monkeypatch.setattr(service, "LLM", mock_LLM)

    result = service.generate_llm_response(question)
    assert isinstance(result, StreamingResponse), "Should be a StreamingResponse instance"
    async for chunk in result.body_iterator:
        assert chunk.startswith("data:"), "Chunk should start with 'data:'"


@pytest.mark.asyncio
async def test_generate_llm_response_with_stream_adapter_error(monkeypatch):
    def mock_search_context(question): return "Mocked context"
    class mockChunk:
        def __init__(self, content):
            self.content = content

    def mock_astream(messages):
        return ""

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM.model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(question="Voglio pizza!", messages=[Message(sender="me", content="Hi")])
    monkeypatch.setattr(service, "LLM", mock_LLM)

    result = service.generate_llm_response(question)
    assert isinstance(result, StreamingResponse), "Should be a StreamingResponse instance"
    result2 = ""
    async for chunk in result.body_iterator:
        result2 += chunk
    assert "data: [ERROR]" in result2, "Should contain error message"

@pytest.mark.asyncio
async def test_generate_llm_response_with_stream_error(monkeypatch):
    def mock_search_context(question): return "Mocked context"
    class mockChunk:
        def __init__(self, content):
            self.content = content

    def mock_astream(messages):
        raise Exception("Mocked exception")

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM.model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(question="Voglio pizza!", messages=[Message(sender="me", content="Hi")])
    monkeypatch.setattr(service, "LLM", mock_LLM)

    with pytest.raises(HTTPException) as excinfo:
        service.generate_llm_response(question)
    assert excinfo.value.status_code == 500, "Should raise HTTPException with status code 500"
    assert str(excinfo.value.detail).startswith("Error in chat service:"), "Should raise HTTPException with the correct error message"