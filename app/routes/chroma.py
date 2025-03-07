from fastapi import APIRouter, Depends, HTTPException
from app.services.chat_services import chat, get_chat_name
from app.services.chroma_services import embedding
import app.schemas as schemas

router = APIRouter()


@router.post("/")
async def create_chat_response(domanda: schemas.Question):
    if not domanda.question:
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")

    contesto = embedding(domanda.question)
    # print(contesto)
    return chat(domanda.question, contesto)

@router.post("/chat_name")
async def generate_chat_name(context: schemas.Context):
	if not context.context:
		raise HTTPException(status_code=400, detail="Nessun contesto fornito")

	print(context.context)
	return get_chat_name(context.context)