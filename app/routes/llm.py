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
    # chat_service=Depends(get_chat_service)
):
    """ "
    Genera un nome per una chat.
    :param context: Il contesto contenente i messaggi della chat.
    :return: Il nome generato per la chat.
    """
    if not context.context:
        raise HTTPException(status_code=400, detail="Nessun contesto fornito")
    print(context.context)

    return ChatService(OpenAI("gpt-4o-mini")).get_chat_name(context.context)


@router.get("/test")
async def test():
    try:
        import os
        import asyncio
        from langchain_openai import OpenAI
        from langchain.chat_models import init_chat_model

        print("starting...")
        llm = init_chat_model(
            model="gpt-3.5-turbo", model_provider="openai"
        )  # Modello più comune

        # Utilizzo di asyncio.wait_for per aggiungere timeout
        try:
            # Eseguire l'operazione con un timeout di 10 secondi
            abc = await asyncio.wait_for(
                llm.ainvoke("Hello how are you?"), timeout=10.0
            )
            print(abc)
            print("Done.")
            return {"response": str(abc)}
        except asyncio.TimeoutError:
            return {"error": "Timeout durante la chiamata all'API OpenAI"}

    except Exception as e:
        import traceback

        print(f"Errore: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}


@router.get("/ping")
async def ping():
    import requests
    ris = requests.get("https://www.google.com")
    return {"status": "ok", "message": ris.text}


@router.get("/check_api")
async def check_api():
    """Verifica la configurazione dell'API OpenAI"""
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        return {"status": "error", "message": "OPENAI_API_KEY non impostata"}

    # Mostra solo i primi 5 caratteri per sicurezza
    masked_key = api_key[:5] + "..." if api_key else ""

    return {
        "status": "configured",
        "api_key_preview": masked_key,
        "length": len(api_key),
    }


@router.get("/test_direct")
async def test_direct():
    """Test diretto con la libreria OpenAI senza LangChain"""
    try:
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Modello standard
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=50,
        )

        return {"response": response.choices[0].message.content}
    except Exception as e:
        import traceback

        print(f"Errore diretto: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}
