from fastapi import APIRouter, Depends, HTTPException
from app.services.chat_services import chat
from app.services.chroma_services import embedding
import app.schemas as schemas

router = APIRouter()


@router.post("/")
async def create_chat_response(domanda: schemas.Question):
    if not domanda.question:
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")

    contesto = embedding(domanda.question)
    # print(contesto)
    risposta = chat(domanda.question, contesto)
    return {"answer": risposta}
