from abc import ABC, abstractmethod
import os
import logging
from langchain.chat_models import init_chat_model
from app.config import settings

logger = logging.getLogger(__name__)

# abstract class
class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._check_environment()
        self._initialize_model()

    @abstractmethod
    def _check_environment(self):  # pragma: no cover
        """
        Controlla le variabili d'ambiente necessarie per il funzionamento del servizio
        """
        pass

    @abstractmethod
    def _initialize_model(self):  # pragma: no cover
        """
        Inizializza il modello LLM
        """
        pass


class OpenAI(LLM):
    # private method
    def _check_environment(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("API key mancante per OpenAI")
        pass

    def _initialize_model(self):
        try:
            self.model = init_chat_model(model=self.model_name, model_provider="openai")
            if self.model is None:
                raise ValueError(f"Failed to initialize model {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {str(e)}")
            raise ValueError(f"Invalid or unavailable model: {self.model_name}") from e


class Ollama(LLM):
    def _check_environment(self):
        pass

    def _initialize_model(self):
        try:
            self.model = init_chat_model(model=self.model_name, model_provider="ollama")
            if self.model is None:
                raise ValueError(f"Failed to initialize model {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {str(e)}")
            raise ValueError(f"Invalid or unavailable model: {self.model_name}") from e

def get_llm_model() -> LLM:
    """Factory function per creare un'istanza di LLM"""
    provider = settings.LLM_PROVIDER.lower()
    match provider:
        case "openai":
            return OpenAI(settings.LLM_MODEL_NAME)
        case "ollama":
            return Ollama(settings.LLM_MODEL_NAME)
        # aggiungere altri provider qui
        case _:
            raise ValueError(f"Provider LLM '{provider}' non supportato.")