from fastapi import APIRouter, Depends, HTTPException
from app.services.chat_services import chat, get_chat_name
from app.services.chroma_services import embedding
import app.schemas as schemas

router = APIRouter()


@router.post("/")
async def create_chat_response(question: schemas.Question):
    if not question.question:
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")

    if not question.messages:
        raise HTTPException(status_code=400, detail="Nessun messaggio fornito")

    #vengono passati solo gli ultimo 6 messaggi
    last_messages = question.messages[-6:]

    context = embedding(question.question)
    messages_context = embedding(" | ".join([f"{msg.sender}: {msg.content}" for msg in last_messages]))

    return chat(question.question, context, messages_context, last_messages)

@router.post("/chat_name")
async def generate_chat_name(context: schemas.Context):
	if not context.context:
		raise HTTPException(status_code=400, detail="Nessun contesto fornito")

	print(context.context)
	return get_chat_name(context.context)