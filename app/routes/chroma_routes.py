from fastapi import APIRouter, Depends, HTTPException
from app.services.chat_services import chat
from app.services.chroma_services import embedding

chroma_router = APIRouter()

@chroma_router.get("/")
async def get_chat(domanda: str):
    if not domanda:
        raise HTTPException(status_code=400, detail="Nessuna domanda fornita")
        
    contesto = embedding(domanda)
    risposta = chat(domanda, contesto)
    return {"risposta": risposta}

@chroma_router.get("/test")
async def get_test():
    return 200
