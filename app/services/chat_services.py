import os
from dotenv import load_dotenv
from fastapi import HTTPException
from starlette.responses import StreamingResponse
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.llm_services import LLM
from app import schemas


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

    def chat(self, question: str, context: str, messages: str) -> StreamingResponse:
        messages = [
            SystemMessage(self.chatbot_instructions),
            SystemMessage(
                f"Contesto: {context}"
            ),  # TODO: da capire se tenere system o human
            SystemMessage(f"Conversazione precedente: {messages}"),
            HumanMessage(f"Domanda a cui devi rispondere: {question}"),
        ]
        try:
            stream_response = self.LLM.model.stream(messages)

            async def stream_adapter():
                try:
                    async for chunk in stream_response:
                        if hasattr(chunk, "content"):
                            content = chunk.content
                        elif isinstance(chunk, dict) and "content" in chunk:
                            content = chunk["content"]
                        else:
                            content = str(chunk)

                        if content:
                            yield f"data: {content}\n\n"

                    # Segnala la fine dello stream
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error in stream adapter: {str(e)}", exc_info=True)
                    yield f"data: [ERROR] {str(e)}\n\n"

        except Exception as e:
            logger.error(f"Error in chat service streaming: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error in chat service: {str(e)}"
            )

        return StreamingResponse(stream_adapter(), media_type="text/event-stream")

    def get_chat_name(self, contesto: str) -> str:
        messages = [
            SystemMessage(
                "Genera un nome per la chat in base alle domande e risposte fornite, deve essere composto da massimo 40 caratteri, non deve contenere informazioni personali e deve essere professionale. Rispondi solo con il nome della chat. Evita di includere 'chatbot' o 'assistente'. Deve racchiudere gli argomenti trattati."
            ),
            HumanMessage(f"Domande: {contesto}"),
        ]
        try:
            return self.LLM.model.invoke(messages)
        except Exception as e:
            logger.error(f"Error generating chat name: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error generating chat name: {str(e)}"
            )
