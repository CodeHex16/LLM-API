from fastapi import APIRouter, Depends, HTTPException
from app.services.chat_services import ChatService
from app.services.llm_services import LLM, OpenAI
from app.services.chroma_services import embedding
import app.schemas as schemas

router = APIRouter(
    tags=["llm"],
)


def get_llm_model():
    """Factory function per creare un'istanza di LLM"""
    return OpenAI("gpt-4o-mini")


def get_chat_service(llm: LLM = Depends(get_llm_model)):
    """Ritorna un ChatService con il modello LLM appropriato"""
    return ChatService(llm)


@router.post("/")
async def create_chat_response(
    domanda: schemas.Question, chat_service=Depends(get_chat_service)
):
    """
    Fornisce una risposta a una domanda.

    *Parametri*:
        - domanda: La domanda da porre al chatbot.
    *Return*:
        - stream della risposta del chatbot.
    """
    if not domanda.question:
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")

    contesto = embedding(domanda.question)
    # print(contesto)
    return chat_service.chat(domanda.question, contesto)


@router.post("/chat_name")
async def generate_chat_name(
    context: schemas.Context,
    chat_service=Depends(get_chat_service)
):
    """ "
    Genera un nome per una chat.
    :param context: Il contesto contenente i messaggi della chat.
    :return: Il nome generato per la chat.
    """
    if not context.context:
        raise HTTPException(status_code=400, detail="Nessun contesto fornito")
    print(context.context)

    return chat_service.get_chat_name(context.context).content


@router.get("/ping")
async def ping():
    import requests
    ris = requests.get("https://www.google.com")
    return {"status": "ok", "message": ris.text}