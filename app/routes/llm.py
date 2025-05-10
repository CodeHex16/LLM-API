from fastapi import APIRouter, Depends, HTTPException

from app.services.llm_response_service import LLMResponseService, get_llm_response_service

import app.schemas as schemas

router = APIRouter(
    tags=["llm"],
)


@router.post("/")
async def generate_chat_response(
    question: schemas.Question, llm_response_service: LLMResponseService = Depends(get_llm_response_service)
):
    """
    Fornisce una risposta a una domanda utilizzando il contesto rilevante.

    ### Args:
        * **question (schemas.Question)**: La domanda e lo storico dei messaggi.

    ### Returns:
        * **response**: La risposta generata dal modello LLM.

    ### Raises:
        * **HTTPException.404_NOT_FOUND**: Se non viene trovato alcun contesto rilevante.
        * **HTTPException.400_BAD_REQUEST**: Se non viene fornita alcuna domanda.
        * **HTTPException.500_INTERNAL_SERVER_ERROR**: Se si verifica un errore interno del server.
    """
    if not question.question or question.question.strip() == "":
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")

    return llm_response_service.generate_llm_response(question)


@router.post("/chat_name")
async def generate_chat_name(
    context: schemas.Context, llm_response_service: LLMResponseService = Depends(get_llm_response_service)
):
    """
    Genera un nome per la chat in base al contesto fornito.

    ### Args:
        * **context (schemas.Context)**: Il contesto della chat.
    
    ### Returns:
        * **response**: Il nome generato per la chat.

    ### Raises:
        * **HTTPException.400_BAD_REQUEST**: Se non viene fornito alcun contesto.
        * **HTTPException.500_INTERNAL_SERVER_ERROR**: Se si verifica un errore interno del server.
    """
    if not context.context:
        raise HTTPException(status_code=400, detail="Nessun contesto fornito")

    return llm_response_service.generate_llm_chat_name(
        context.context
    )