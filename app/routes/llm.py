from fastapi import APIRouter, Depends, HTTPException
from app.services.chat_services import chat, get_chat_name
from app.services.chroma_services import embedding
import app.schemas as schemas

router = APIRouter(
    tags=["llm"],
)


@router.post("/")
async def create_chat_response(domanda: schemas.Question, Depends=Depends(chat)):
    """
    Fornisce una risposta a una domanda.
    :param domanda: La domanda da porre al chatbot.
    :return: La risposta del chatbot.
    """
    if not domanda.question:
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")

    contesto = embedding(domanda.question)
    # print(contesto)
    return chat(domanda.question, contesto)


@router.post("/chat_name")
async def generate_chat_name(context: schemas.Context):
    """ "
    Genera un nome per una chat.
    :param context: Il contesto contenente i messaggi della chat.
    :return: Il nome generato per la chat.
    """
    if not context.context:
        raise HTTPException(status_code=400, detail="Nessun contesto fornito")

    print(context.context)
    return get_chat_name(context.context)
