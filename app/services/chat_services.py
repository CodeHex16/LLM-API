import os
from dotenv import load_dotenv
from fastapi import HTTPException
from starlette.responses import StreamingResponse
import logging
import json

logger = logging.getLogger(__name__)

load_dotenv()
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chat(question, context, messages_context, messages):

    CHATBOT_INSTRUCTIONS = """
    Sei il chatbot di un'azienda.

    Missione:
    Assistere gli utenti nell'esplorazione dei prodotti forniti dall'azienda, informarli sulle caratteristiche del prodotto e consigliarne l'acquisto.

    Tratti della personalità:
    - Conoscenza: Fornisce risposte accurate dalla base di conoscenze.
    - Amichevole: cordiale e disponibile.
    - Trasparente: condivide solo informazioni convalidate.

    Capacità:
    - Educare: Spiegare i prodotti presenti, consigliarne i possibili usi, la storia dell'azienda e i suoi valori utilizzando la base di conoscenze.
    - Assistere: Consigliare prodotti e fornire informazioni rigorosamente basate sui dati approvati.
    - Ispirare: evidenziare i vantaggi e gli usi di ogni prodotto.
    - Coinvolgere: Rispondere alle domande in modo chiaro ed educato, reindirizzando gli utenti al supporto se le risposte non sono disponibili.

    Tono:
    - Positivo, professionale e privo di gergo.
    - Rispettoso ed empatico per garantire un'esperienza di supporto.

    Regole comportamentali:
    - Utilizzare solo la base di conoscenze fornita.
    - Se una risposta non è disponibile, informare l'utente e suggerire di consultare l'assistenza clienti.
    - Non fornire informazioni personali.
    """

    try:
        answer = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": CHATBOT_INSTRUCTIONS,
                },
                {"role": "user", "content": f"Contesto: {context}"},
                {"role": "user", "content": f"Messaggi precedenti: {messages}"},
                {"role": "user", "content": f"Contesto dei messaggi precedenti: {messages_context}"},
                {"role": "user", "content": f"Domanda: {question}"},
            ],
            stream=True,
        )

        def async_generator():
            for chunk in answer:
                logging.debug(f"Received chunk: {chunk}")

                response_data = {
                    "id": chunk.id,
                    "object": chunk.object,
                    "created": chunk.created,
                    "model": chunk.model,
                    "system_fingerprint": chunk.system_fingerprint,
                    "choices": [
                        {
                            "index": chunk.choices[0].index,
                            "delta": {
                                "content": getattr(
                                    chunk.choices[0].delta, "content", None
                                )
                            },
                            "finish_reason": chunk.choices[0].finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(response_data)}\n\n"

        return StreamingResponse(async_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Errore nel servizio chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Errore nel servizio chat: {str(e)}"
        )


def get_chat_name(context):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Genera un nome per la chat in base alle domande e risposte fornite, deve essere composto da massimo 40 caratteri, non deve contenere informazioni personali e deve essere professionale. Rispondi solo con il nome della chat. Evita di includere 'chatbot' o 'assistente'. Deve racchiudere gli argomenti trattati.",
                },
                {"role": "user", "content": f"Domande: {context}"},
            ],
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Errore nel servizio chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Errore nel servizio chat: {str(e)}"
        )
