from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from fastapi import Depends
import os
import logging
import uuid
from chromadb.errors import DuplicateIDError

logger = logging.getLogger(__name__)

from app.services.embeddings_service import EmbeddingProvider, get_embedding_provider
from app.config import settings


class VectorDatabase(ABC):
    """Interfaccia per la gestione del database vettoriale."""

    @abstractmethod
    def __init__(self):
        self._embedding_provider = get_embedding_provider()
        self._persist_directory = settings.VECTOR_DB_DIRECTORY
        self._db = None

    @abstractmethod
    def _get_db(self):
        pass

    @abstractmethod
    def __init__(self):
        self._embedding_provider = get_embedding_provider()
        self._persist_directory = settings.VECTOR_DB_DIRECTORY
        self._db = None

    @abstractmethod
    def _get_db(self):
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]):
        pass

    @abstractmethod
    def delete_document(self, document_path: str):
        pass

    @abstractmethod
    def search_context(self, query: str, results_number: int = 4) -> List[Document]:
        pass

    @abstractmethod
    def delete_all_documents(self):
        pass

    @abstractmethod
    def get_all_documents(self):
        pass

    # metodi ausiliari
    @abstractmethod
    def is_empty(self) -> bool:  # pragma: no cover
        pass

    @abstractmethod
    def count(self) -> int:  # pragma: no cover
        pass


class ChromaDB(VectorDatabase):
    """Implementazione del database vettoriale ChromaDB."""

    def __init__(
        self,
        persist_directory: str = settings.VECTOR_DB_DIRECTORY,
    ):
        self.embedding_provider = get_embedding_provider()
        self.persist_directory = persist_directory
        self._db = None

    # Singleton
    def _get_db(self):
        if self._db is None:

            logger.info(
                f"ChromaDB: Inizializzazione del database in {self.persist_directory}"
            )
            self._db = Chroma(
                collection_name="supplai_documents",
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_provider.get_embedding_function(),
            )
        return self._db

    def _generate_document_ids(self, documents: List[Document]) -> List[str]:
        """Estrae gli ID dei documenti."""
        return [
            str(uuid.uuid3(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents
        ]

    def add_documents(self, documents_chunk: List[Document]):
        print("document_chunks", documents_chunk)
        if not documents_chunk:
            logger.warning("Nessun documento fornito per l'aggiunta.")
            return
        try:
            db = self._get_db()

            ids_to_add = self._generate_document_ids(documents_chunk)

            collection = db._collection

            existing_data = collection.get(ids=ids_to_add)

            if existing_data and existing_data["ids"]:
                colliding_ids = existing_data["ids"]
                raise DuplicateIDError(
                    f"Attempted to add documents with an ID that already exists: {', '.join(colliding_ids)}"
                )

            db.add_documents(
                documents=documents_chunk,
                ids=ids_to_add,
            )

            print(
                f"ChromaDB: Aggiunti {len(documents_chunk)} documenti al vector store."
            )
            print("ChromaDB: numero di documenti presenti", self.count())
            logger.info(f"Aggiunti {len(documents_chunk)} documenti al vector store.")

        except DuplicateIDError as e:
            raise DuplicateIDError(
                f"ID duplicato trovato durante l'aggiunta di documenti: {e}"
            )
        except Exception as e:
            logger.error(
                f"Errore durante l'aggiunta di documenti a Chroma: {e}", exc_info=True
            )
            raise

    def delete_document(self, document_path: str):
        """Elimina un documento dal database."""
        try:
            db = self._get_db()
            db.delete(where={"source": document_path})
            print(f"[VECTOR DB] Documento con PATH {document_path} eliminato.")
            logger.info(f"Documento con PATH {document_path} eliminato.")
        except Exception as e:
            logger.error(
                f"Errore durante l'eliminazione del documento: {e}", exc_info=True
            )
            raise

    def delete_faq(self, faq_id: str):
        """Elimina una FAQ dal database."""
        try:
            db = self._get_db()
            db.delete(where={"faq_id": faq_id})
            print(f"[VECTOR DB] FAQ con ID {faq_id} eliminata.")
            logger.info(f"FAQ con ID {faq_id} eliminata.")
        except Exception as e:
            logger.error(f"Errore durante l'eliminazione della FAQ: {e}", exc_info=True)
            raise

    def search_context(self, query: str, results_number: int = 4) -> List[Document]:
        try:
            db = self._get_db()
            results = db.similarity_search(query, k=results_number)
            return results
        except Exception as e:
            logger.error(f"Errore durante la similarity search: {e}", exc_info=True)
            return []

    def delete_all_documents(self):
        """Elimina tutti i documenti dal database."""
        try:
            db = self._get_db()
            db.reset_collection()
            print("[VECTOR DB] Tutti i documenti eliminati.")
        except Exception as e:
            logger.error(
                f"Errore durante l'eliminazione di tutti i documenti: {e}",
                exc_info=True,
            )
            raise

    def get_all_documents(self):
        """Recupera tutti i documenti dal database."""
        try:
            db = self._get_db()
            results = db.get()
            return results
        except Exception as e:
            logger.error(
                f"Errore durante il recupero di tutti i documenti: {e}", exc_info=True
            )
            return []

    # metodi ausiliari
    def _get_collection_count(self) -> int:
        """Helper per gestire accesso a dettagli Chroma."""
        try:
            db_instance = self._get_db()
            if db_instance and db_instance._collection:
                return db_instance._collection.count()
            print(
                "Impossibile ottenere il count: istanza DB o _collection non disponibile dopo _get_db()."
            )
            return 0
        except Exception as e:
            print(f"Errore durante il recupero del count della collection: {e}")
            return 0

    def is_empty(self) -> bool:
        return self._get_collection_count() == 0

    def count(self) -> int:
        return self._get_collection_count()

    def _delete(self):
        return self._get_db().delete_collection()


def get_vector_database() -> VectorDatabase:
    match settings.VECTOR_DB_PROVIDER.lower():
        case "chroma":
            vdb = ChromaDB(persist_directory=settings.VECTOR_DB_DIRECTORY)
            # vdb.ensure_vectorized(settings.DOCUMENTS_FOLDER)
            return vdb
        case _:
            raise ValueError(
                f"Provider di database vettoriale '{settings.VECTOR_DB_PROVIDER}' non supportato."
            )
