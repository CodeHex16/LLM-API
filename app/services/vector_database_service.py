from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from fastapi import Depends
import os
import logging

logger = logging.getLogger(__name__)

from app.services.embeddings_service import EmbeddingProvider, get_embedding_provider
from app.config import settings


class VectorDatabase(ABC):
    """Interfaccia per la gestione del database vettoriale."""

    @abstractmethod
    def add_documents(self, documents: List[Document]):
        pass

    @abstractmethod
    def search_context(self, query: str, results_number: int = 2) -> List[Document]:
        pass

    # metodi ausiliari
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def ensure_vectorized(self, documents_folder: str):
        """Metodo per caricare e vettorializzare se vuoto."""
        # TODO: capire se lasciare in produzione
        pass


# TODO: cistemare con i document loaders specifici
class ChromaDB(VectorDatabase):
    """Implementazione del database vettoriale ChromaDB."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_directory: str = settings.VECTOR_DB_PERSIST_DIRECTORY,
    ):
        self.embedding_provider = embedding_provider
        # TODO: verificare come salva i file, se all'interno del container o in locale, e poi vedere il modo ideale per gestirli
        self.persist_directory = persist_directory
        self._db = None

    # Singleton
    def _get_db(self):
        if self._db is None:
            self._db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_provider.get_embedding_function(),
            )
        return self._db

    # TODO: da spostare nel contextManager
    # def _load_and_split_docs(self, folder_path: str) -> List[Document]:
    

    def add_documents(self, documents_chunck: List[Document]):
        if not documents_chunck:
            logger.warning("Nessun documento fornito per l'aggiunta.")
            return
        try:
            self._db = Chroma.from_documents(
                documents=documents_chunck,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory,
            )
            logger.info(f"Aggiunti {len(documents_chunck)} documenti al vector store.")
        except Exception as e:
            logger.error(
                f"Errore durante l'aggiunta di documenti a Chroma: {e}", exc_info=True
            )
            raise

    def search_context(self, query: str, results_number: int = 2) -> List[Document]:
        # TODO: Non è detto che serva: Verifica se ci sono documenti
        # ensure_vectorized()
        try:
            results = self._db.similarity_search(query, k=results_number)
            return results
        except Exception as e:
            logger.error(f"Errore durante la similarity search: {e}", exc_info=True)
            return []

    # metodi ausiliari
    def _get_collection_count(self) -> int:
        """Helper per gestire accesso a dettagli Chroma."""
        try:
            client = self._db.get()
            if client and self._db._collection:
                return self._db._collection.count()
            return 0
        except Exception as e:
            logger.warning(
                f"Impossibile ottenere il count della collection (potrebbe non esistere ancora): {e}"
            )
            return 0

    def is_empty(self) -> bool:
        return self._get_collection_count() == 0

    def count(self) -> int:
        return self._get_collection_count()

    def ensure_vectorized(self, documents_folder: str):
        """Controlla se il DB è vuoto e, in caso, carica e vettorializza."""
        if self.is_empty():
            logger.info(
                f"Vector store in {self.persist_directory} è vuoto. Avvio vettorizzazione da {documents_folder}..."
            )

            # TODO: delegare al context manager
            texts_to_add = self._load_and_split_docs(documents_folder)
            if texts_to_add:
                self.add_documents(texts_to_add)
                logger.info(
                    f"Vettorizzazione completata. {self.count()} documenti nel DB."
                )
            else:
                logger.warning("Nessun documento da vettorializzare trovato.")
        else:
            logger.info(f"Vector store già inizializzato con {self.count()} documenti.")


def get_vector_database(
    embedding_provider: EmbeddingProvider = Depends(
        get_embedding_provider
    ),  # Inietta il provider
) -> VectorDatabase:
    match settings.VECTOR_DB_PROVIDER.lower():
        case "chroma":
            vdb = ChromaDB(embedding_provider, persist_directory=settings.VECTOR_DB_PERSIST_DIRECTORY)
            vdb.ensure_vectorized(settings.DOCUMENTS_FOLDER)
            return vdb
        case _:
            raise ValueError(
                f"Provider di database vettoriale '{settings.VECTOR_DB_PROVIDER}' non supportato."
            )