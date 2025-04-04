import os
from dotenv import load_dotenv
from fastapi import HTTPException, Depends
from starlette.responses import StreamingResponse
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app import schemas
from typing import List

from app.config import settings


from app.services.vector_database_service import VectorDatabase, get_vector_database
from app.services.llm_service import LLM, get_llm_model

logger = logging.getLogger(__name__)

load_dotenv()


class LLMResponseService:
    def __init__(
        self
    ):
        self.LLM = get_llm_model()
        self.vector_database = get_vector_database()
        self.CHATBOT_INSTRUCTIONS = settings.CHATBOT_INSTRUCTIONS

    def _get_context(self, question: str) -> str:
        """
        Get the context for the question from the vector database.
        """
        try:
            question_context = self.vector_database.search_context(question)
            if not question_context:
                raise ValueError("No context found for the question.")
            return question_context
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error getting context: {str(e)}"
            )

    def generate_llm_response(self, question: schemas.Question) -> StreamingResponse:        
        context = self._get_context(question.question)
        # TODO: gestire array messaggi
        formatted_messages = ""
        
        if question.messages:
            if isinstance(question.messages, list):
                formatted_messages = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in question.messages]
                )
            else:
                formatted_messages = question.messages
            context_messages = self._get_context(formatted_messages)

        messages = [
            SystemMessage(self.CHATBOT_INSTRUCTIONS),
            SystemMessage(
                f"Contesto: {context}\n{context_messages}",
            ),
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

    def generate_llm_chat_name(self, chat_history: str) -> str:
        messages_context = self._get_context(chat_history)

        messages = [
            SystemMessage(
                "Genera un nome per la chat in base alle domande e risposte fornite, deve essere composto da massimo 40 caratteri, non deve contenere informazioni personali e deve essere professionale. Rispondi solo con il nome della chat. Evita di includere 'chatbot' o 'assistente'. Deve racchiudere gli argomenti trattati."
            ),
            HumanMessage(f"Domande: {chat_history}"),
        ]

        try:
            return self.LLM.model.invoke(messages)
        except Exception as e:
            logger.error(f"Error generating chat name: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error generating chat name: {str(e)}"
            )


def get_llm_response_service() -> LLMResponseService:
    return LLMResponseService()
