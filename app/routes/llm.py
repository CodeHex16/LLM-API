from fastapi import APIRouter, Depends, HTTPException
from app.services.llm_response_service import ChatService
from app.services.llm_service import LLM, OpenAI

# from app.services.chroma_services import embedding
from app.services.vector_database_service import get_vector_database_service
from app.services.llm_response_service import LLMResponseService, get_llm_response_service

import app.schemas as schemas

router = APIRouter(
    tags=["llm"],
)


@router.post("/")
async def create_chat_response(
    question: schemas.Question, llm_response_service: LLMResponseService = Depends(get_llm_response_service)
):
    """
    Fornisce una risposta a una domanda utilizzando il contesto rilevante.

    Args:
        question (schemas.Question): La domanda e lo storico dei messaggi.
        chat_service: Servizio di chat per generare risposte.

    Returns:
        La risposta generata dal modello LLM.

    Raises:
        HTTPException: Se non viene fornita una domanda valida.
    """
    if not question.question or question.question.strip() == "":
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")
    
    llm_response_service = get_llm_response_service()
    return llm_response_service.generate_llm_response(question.question, question.messages)


@router.post("/chat_name")
async def generate_chat_name(
    context: schemas.Context
):
    """ "
    Genera un nome per una chat.
    :param context: Il contesto contenente i messaggi della chat.
    :return: Il nome generato per la chat.
    """
    if not context.context:
        raise HTTPException(status_code=400, detail="Nessun contesto fornito")

    llm_response_service = get_llm_response_service() 
    return llm_response_service.generate_llm_chat_name(
        context.context
    )


@router.get("/ping")
async def ping():
    import requests

    ris = requests.get("https://www.google.com")
    return {"status": "ok", "message": ris.text}
