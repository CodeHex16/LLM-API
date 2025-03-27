from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
	OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
	DOCUMENTS_FOLDER: str = "documenti"
	CHROMA_DB_PATH: str = "chroma_db"
	EMBEDDING_MODEL_NAME: str = "text-embedding-ada-002"
	LLM_MODEL_NAME: str = "gpt-4o-mini"
	LLM_PROVIDER: str = "openai"

	class Config:
		env_file = ".env"

settings = Settings()