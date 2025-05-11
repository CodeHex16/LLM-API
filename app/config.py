from pydantic_settings import SettingsConfigDict, BaseSettings
import os


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY")
    DOCUMENTS_FOLDER: str = "documenti"
    VECTOR_DB_PROVIDER: str = "chroma"
    VECTOR_DB_DIRECTORY: str = "chroma_db"
    EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_PROVIDER: str = "openai"
    CHATBOT_INSTRUCTIONS: str = """
        Sei il chatbot di un'azienda.

        Missione:
        Assistere gli utenti nell'esplorazione dei prodotti forniti dall'azienda, informarli sulle caratteristiche del prodotto e consigliane l'acquisto.

        Regole comportamentali:
        - È essenziale che tu usi il più possibile le informazioni fornite dai documenti passati come contesto.
        - Se una risposta non è disponibile, informare l'utente e suggerire di consultare l'assistenza clienti.
        - Sii chiaro ed elenca metodicamente le informazioni richieste.
		- Non esprimere opinioni personali o fare supposizioni.
        - Non fornire informazioni personali.
        """
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 100
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
