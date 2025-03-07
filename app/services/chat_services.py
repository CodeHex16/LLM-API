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


def chat(domanda, contesto):
    
    CHATBOT_INSTRUCTIONS = """
    Sei il chatbot di un'azienda alimentare.

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

    logger.info(f"Richiesta di risposta alla domanda: {domanda}")
    logger.info(f"Contesto fornito: {contesto}")
    try:
        risposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": CHATBOT_INSTRUCTIONS,
                },
                {"role": "user", "content": f"Contesto: {contesto}"},
                {"role": "user", "content": f"Domanda: {domanda}"},
            ],
            stream=True,
        )

        def async_generator():
            for chunk in risposta:
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

        # testo_risposta = risposta.choices[0].message.content
        # return testo_risposta

    except Exception as e:
        logger.error(f"Errore nel servizio chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Errore nel servizio chat: {str(e)}"
        )
