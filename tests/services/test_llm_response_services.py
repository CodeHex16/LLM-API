import pytest
from app.services.llm_response_service import (
    LLMResponseService,
    get_llm_response_service,
)
from app.services.vector_database_service import VectorDatabase, get_vector_database
from app.services.llm_service import LLM, get_llm_model
from app.schemas import Question, Message
from starlette.responses import StreamingResponse
from unittest.mock import MagicMock
from fastapi import HTTPException


def test_llm_response_service_initialization():
    llm_response_service = LLMResponseService()
    assert isinstance(
        llm_response_service, LLMResponseService
    ), "Should return an instance of LLMResponseService"
    assert isinstance(
        llm_response_service._LLM, LLM
    ), "LLM should be an instance of LLM"
    assert isinstance(
        llm_response_service._vector_database, VectorDatabase
    ), "Vector database should be an instance of VectorDatabase"


def test_llm_response_service_get_context(monkeypatch):
    llm_response_service = LLMResponseService()
    question = "What is the capital of France?"

    # Mock the vector database search_context method
    # It should return an iterable of objects, each having a 'page_content' attribute.
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Paris is the capital of France."
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "It is a beautiful city."

    mock_search_context_result = [mock_doc1, mock_doc2]

    monkeypatch.setattr(
        llm_response_service._vector_database,
        "search_context",
        lambda q: mock_search_context_result,
    )

    # _get_context returns a list of page_content strings
    context_list = llm_response_service._get_context(question)

    expected_context_list = [
        "Paris is the capital of France.",
        "It is a beautiful city.",
    ]
    assert (
        context_list == expected_context_list
    ), "Should return a list of page_content strings from the mocked documents"


def test_llm_response_service_get_context_false(monkeypatch):
    llm_response_service = LLMResponseService()
    question = "What is the capital of France?"

    # Mock the vector database search_context method
    mock_context = "Paris is the capital of France."
    monkeypatch.setattr(
        llm_response_service._vector_database, "search_context", lambda q: None
    )

    with pytest.raises(HTTPException) as excinfo:
        llm_response_service._get_context(question)
    assert (
        excinfo.value.status_code == 404
    ), "Should raise HTTPException with status code 404"
    assert str(excinfo.value.detail).startswith(
        "Error getting context:"
    ), "Should raise HTTPException with the correct error message"


def test_llm_response_service_get_context_exception(monkeypatch):
    llm_response_service = LLMResponseService()
    question = "What is the capital of France?"

    # Mock the vector database search_context method to raise an exception
    error_message = "Database connection error"
    monkeypatch.setattr(
        llm_response_service._vector_database,
        "search_context",
        MagicMock(side_effect=Exception(error_message)),
    )

    with pytest.raises(HTTPException) as excinfo:
        llm_response_service._get_context(question)
    assert (
        excinfo.value.status_code == 500
    ), "Should raise HTTPException with status code 500"

    # Corrected assertion for the detail message
    expected_detail = f"Unexpected error getting context: {error_message}"
    assert (
        excinfo.value.detail == expected_detail
    ), "Should raise HTTPException with the correct error message for generic exceptions"


def test_get_llm_response_service():
    llm_response_service = get_llm_response_service()
    assert isinstance(
        llm_response_service, LLMResponseService
    ), "Should return an instance of LLMResponseService"


def test_generate_llm_chat_name(monkeypatch):
    response = MagicMock()
    response.content = "mocked response content"

    llm_response_service = LLMResponseService()
    llm_response_service._LLM = MagicMock()
    llm_response_service._LLM.model = MagicMock()

    llm_response_service._get_context = lambda question: "mocked context"
    llm_response_service._LLM.model.invoke = response

    llm_response_service.generate_llm_chat_name("mocked question")
    assert (
        response.content == "mocked response content"
    ), "Should return the mocked response content"


def test_generate_llm_chat_name_exception(monkeypatch):
    llm_response_service = LLMResponseService()

    # Mock the _LLM attribute of the service instance
    mock_llm_attribute = MagicMock(name="mock_service_LLM_attribute")
    llm_response_service._LLM = mock_llm_attribute

    error_message = "Mocked LLM API error"
    llm_response_service._LLM._model.invoke.side_effect = Exception(error_message)

    with pytest.raises(HTTPException) as excinfo:
        llm_response_service.generate_llm_chat_name("mocked question")

    assert (
        excinfo.value.status_code == 500
    ), "Should raise HTTPException with status code 500"

    expected_detail = f"Error generating chat name: {error_message}"
    assert (
        excinfo.value.detail == expected_detail
    ), "Should raise HTTPException with the correct detail message"


@pytest.mark.asyncio
async def test_generate_llm_response_with_messages_list(monkeypatch):
    def mock_search_context(question):
        return "Mocked context"

    class mockChunk:
        def __init__(self, content):
            self.content = content

    async def mock_astream(messages):
        for chunk in [
            "chunk1",
            "chunk2",
            {"content": "chunk3"},
            mockChunk("chunk4"),
            "",
        ]:  # 模拟异步迭代
            yield chunk

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM.model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(
        question="Voglio pizza!", messages=[Message(sender="me", content="Hi")]
    )
    monkeypatch.setattr(service, "_LLM", mock_LLM)

    result = service.generate_llm_response(question)
    assert isinstance(
        result, StreamingResponse
    ), "Should be a StreamingResponse instance"
    async for chunk in result.body_iterator:
        assert chunk.startswith("data:"), "Chunk should start with 'data:'"


@pytest.mark.asyncio
async def test_generate_llm_response_with_empty_messages_list(monkeypatch):
    def mock_search_context(question):
        return "Mocked context"

    class mockChunk:
        def __init__(self, content):
            self.content = content

    async def mock_astream(messages):
        for chunk in [
            "chunk1",
            "chunk2",
            {"content": "chunk3"},
            mockChunk("chunk4"),
            "",
        ]:  # 模拟异步迭代
            yield chunk

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM.model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(question="Voglio pizza!", messages=[])
    monkeypatch.setattr(service, "_LLM", mock_LLM)

    result = service.generate_llm_response(question)
    assert isinstance(
        result, StreamingResponse
    ), "Should be a StreamingResponse instance"
    async for chunk in result.body_iterator:
        assert chunk.startswith("data:"), "Chunk should start with 'data:'"


@pytest.mark.asyncio
async def test_generate_llm_response_with_stream_error(monkeypatch):
    def mock_search_context(question):
        return "Mocked context"

    def mock_astream(messages):
        raise Exception("Mocked exception")

    mock_LLM = MagicMock()
    mock_LLM_MODEL = MagicMock()
    mock_LLM_MODEL.astream = mock_astream
    mock_LLM._model = mock_LLM_MODEL

    service = LLMResponseService()
    monkeypatch.setattr(service, "_get_context", mock_search_context)
    question = Question(
        question="Voglio pizza!", messages=[Message(sender="me", content="Hi")]
    )
    monkeypatch.setattr(service, "_LLM", mock_LLM)

    with pytest.raises(HTTPException) as excinfo:
        service.generate_llm_response(question)
    assert (
        excinfo.value.status_code == 500
    ), "Should raise HTTPException with status code 500"
    assert str(excinfo.value.detail).startswith(
        "Error in chat service:"
    ), "Should raise HTTPException with the correct error message"
