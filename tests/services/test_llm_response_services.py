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

@pytest.mark.xfail(reason="This test is expected to fail due to unimplemented functionality")
def test_llm_response_service_generate_response(monkeypatch):
    llm_response_service = LLMResponseService()
    question = Question(question="What is the capital of France?", messages=[])
    # Mock the vector database search_context method
    mock_context = "Paris is the capital of France."
    monkeypatch.setattr(llm_response_service.vector_database, "search_context", lambda q: mock_context)

    # Mock the LLM model stream method
    mock_stream_response = ["Paris is the capital of France."]
    
    monkeypatch.setattr(llm_response_service.LLM.model, "stream", lambda messages: mock_stream_response)
    
    # monkeypatch.setattr(llm_response_service, "generate_llm_response", lambda messages: mock_stream_response)
    result = llm_response_service.generate_llm_response(question)
    print(result)
    # Check if the response is as expected
    # assert isinstance(result, ), "Should return a StreamingResponse"
    assert result == mock_stream_response, "Should return the mocked stream response"

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


# @pytest.mark.asyncio
# async def test_generate_llm_response_with_messages_list(monkeypatch):
#     def mock_search_context(question): return "Mocked context"
#     async def mock_astream(messages): yield {"content": "Paris"}

#     mock_LLM = MagicMock()
#     mock_LLM.model.astream = mock_astream

#     service = LLMResponseService()
#     monkeypatch.setattr(service.vector_database, "search_context", mock_search_context)
#     monkeypatch.setattr(service, "LLM", mock_LLM)

#     question = Question(
#         question="Capital of France?",
#         messages=[Message(sender="user", content="Hello?")]
#     )

#     response = service.generate_llm_response(question)
#     assert isinstance(response, StreamingResponse)

#     body = "".join([chunk async for chunk in response.body_iterator])
#     assert "Paris" in body

# @pytest.mark.asyncio
# async def test_generate_llm_response_without_messages(monkeypatch):
#     def mock_search_context(question): return "Mocked context"
#     async def mock_astream(messages): yield {"content": "Paris"}

#     mock_LLM = MagicMock()
#     mock_LLM.model.astream = mock_astream

#     service = LLMResponseService()
#     monkeypatch.setattr(service.vector_database, "search_context", mock_search_context)
#     monkeypatch.setattr(service, "LLM", mock_LLM)

#     question = Question(question="Capital of France?")
#     response = service.generate_llm_response(question)
#     body = b"".join([chunk async for chunk in response.body_iterator])
#     assert b"Paris" in body

# @pytest.mark.asyncio
# async def test_generate_llm_response_with_astream_exception(monkeypatch):
#     def mock_search_context(question): return "Mocked context"
#     async def mock_astream(messages): raise Exception("LLM error")

#     mock_LLM = MagicMock()
#     mock_LLM.model.astream = mock_astream

#     service = LLMResponseService()
#     monkeypatch.setattr(service.vector_database, "search_context", mock_search_context)
#     monkeypatch.setattr(service, "LLM", mock_LLM)

#     question = Question(
#         question="Capital of France?",
#         messages=[Message(sender="user", content="Bonjour")]
#     )

#     response = service.generate_llm_response(question)
#     body = b"".join([chunk async for chunk in response.body_iterator])
#     assert b"[ERROR] LLM error" in body
