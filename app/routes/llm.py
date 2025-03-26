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
    question: schemas.Question, chat_service=Depends(get_chat_service)
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
    
    # contesto dalla domanda corrente
    context = embedding(question.question)
    
    # Se ci sono messaggi precedenti, considera gli ultimi 6 per cercare altro contesto
    if question.messages:
        last_messages = question.messages[-6:]

        conversation_history = " ".join([
            f"{msg.sender}: {msg.content}" 
            for msg in last_messages
            if msg.content.strip()
        ])
        
        messages_context = embedding(conversation_history)

        context = f"{context} {messages_context}" # Unire i contesti
    
    return chat_service.chat(question.question, context)


@router.post("/chat_name")
async def generate_chat_name(
    context: schemas.Context, chat_service=Depends(get_chat_service)
):
    """ "
    Genera un nome per una chat.
    :param context: Il contesto contenente i messaggi della chat.
    :return: Il nome generato per la chat.
    """
    if not context.context:
        raise HTTPException(status_code=400, detail="Nessun contesto fornito")

    return chat_service.get_chat_name(context.context).content


@router.get("/ping")
async def ping():
    import requests

    ris = requests.get("https://www.google.com")
    return {"status": "ok", "message": ris.text}
