import os
from dotenv import load_dotenv
from fastapi import HTTPException
from starlette.responses import StreamingResponse
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llm_services import LLM


logger = logging.getLogger(__name__)

load_dotenv()

CHATBOT_INSTRUCTIONS = """
Sei il chatbot di un'azienda.

Missione:
Assistere gli utenti nell'esplorazione dei prodotti forniti dall'azienda, informarli sulle caratteristiche del prodotto e consigliane l'acquisto.

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


class ChatService:
    def __init__(
        self, LLM_model: LLM, chatbot_instructions: str = CHATBOT_INSTRUCTIONS
    ):
        self.LLM = LLM_model
        self.chatbot_instructions = chatbot_instructions

    def chat(self, domanda: str, contesto: str) -> StreamingResponse:
        messages = [
            SystemMessage(self.chatbot_instructions),
            SystemMessage(
                f"Contesto: {contesto}"
            ),  # TODO: da capire se tenere system o human
            HumanMessage(domanda),
        ]
        try:
            stream_resp = self.LLM.model.stream(messages)
        except Exception as e:
            logger.error(f"Error in chat service streaming: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error in chat service: {str(e)}"
            )
        return StreamingResponse(stream_resp, media_type="text/event-stream")

    def get_chat_name(self, contesto: str) -> str:
        messages = [
            SystemMessage(
                "Genera un nome per la chat in base alle domande e risposte fornite, deve essere composto da massimo 40 caratteri, non deve contenere informazioni personali e deve essere professionale. Rispondi solo con il nome della chat. Evita di includere 'chatbot' o 'assistente'. Deve racchiudere gli argomenti trattati."
            ),
            HumanMessage(f"Domande: {contesto}"),
        ]
        try:
            print("self.LLM", self.LLM)
            print("self.LLM.model", self.LLM.model)
            
            risp = self.LLM.model.invoke(messages)
            print("risposta", risp)
            return risp
        except Exception as e:
            logger.error(f"Error generating chat name: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error generating chat name: {str(e)}"
            )
