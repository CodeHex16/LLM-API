import pytest
from app.services.llm_response_service import LLMResponseService, get_llm_response_service
from app.services.vector_database_service import VectorDatabase, get_vector_database
from app.services.llm_service import LLM, get_llm_model
from app.schemas import Question
from starlette.responses import StreamingResponse

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

