from abc import ABC, abstractmethod
import os
import logging
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)

# abstract class
class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._check_environment()
        self._initialize_model()

    @abstractmethod
    def _check_environment(self):
        """
        Controlla le variabili d'ambiente necessarie per il funzionamento del servizio
        """
        pass

    @abstractmethod
    def _initialize_model(self):
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
    def _initialize_model(self):
        try:
            self.model = init_chat_model(model=self.model_name, model_provider="ollama")
            if self.model is None:
                raise ValueError(f"Failed to initialize model {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {str(e)}")
            raise ValueError(f"Invalid or unavailable model: {self.model_name}") from e
