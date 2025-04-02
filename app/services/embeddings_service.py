from abc import ABC, abstractmethod
from langchain_openai import OpenAIEmbeddings
import os
from app.config import settings


class EmbeddingProvider(ABC):
	"""Interfaccia per i provider di embedding."""

	@abstractmethod
	def get_embedding_function(self):
		"""Restituisce la funzione di embedding."""
		pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
	"""Provider di embedding di OpenAI."""
	def __init__(self, api_key: str = settings.OPENAI_API_KEY, model_name: str = settings.EMBEDDING_MODEL_NAME):
		self.api_key = api_key
		self.model_name = model_name
		self._embedding_function = None
	
	def get_embedding_function(self):
		"""Restituisce la funzione di embedding."""
		if os.environ.get("OPENAI_API_KEY") is None and self.api_key is None:
			raise ValueError("API key non trovata. Assicurati di averla impostata.")

		if self._embedding_function is None:
			self._embedding_function = OpenAIEmbeddings(
				openai_api_key=self.api_key,
				model=self.model_name
			)
		return self._embedding_function

def get_embedding_provider() -> EmbeddingProvider: 
	"""Restituisce il provider di embedding in base alla configurazione."""
	provider = settings.LLM_PROVIDER.lower()
	match provider:
		case "openai":
			return OpenAIEmbeddingProvider()
			# aggiungere altri provider qui
		case _:
			raise ValueError(f"Provider di embedding '{provider}' non supportato.")