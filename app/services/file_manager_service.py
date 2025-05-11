from abc import ABC, abstractmethod
from fastapi import Depends, File, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from fastapi import HTTPException
import os
import logging
import json
import requests
from datetime import datetime

from app.services.vector_database_service import get_vector_database
import app.schemas as schemas
from app.config import settings

logger = logging.getLogger(__name__)


class FileManager(ABC):
    def __init__(self):
        self._vector_database = get_vector_database()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def get_full_path(self, filename: str) -> str:
        """
        Restituisce il percorso completo del file.

        Param:
        - filename: str - Il nome del file.

        Returns:
        - str: Il percorso completo del file.
        """
        documents_dir = os.environ.get("DOCUMENTS_DIR", "/data/documents")
        return os.path.join(documents_dir, filename)

    def get_documents_number(self):
        """
        Restituisce le statistiche sui documenti.

        Returns:
        - dict: Un dizionario contenente le statistiche sui documenti.
        """
        # self.vector_database.delete_all_documents()
        return self._vector_database.count()

    def get_documents(self):
        """
        Restituisce i documenti dal database vettoriale.

        Param:
        - skip: int - Il numero di documenti da saltare.
        - limit: int - Il numero massimo di documenti da restituire.

        Returns:
        - list: Una lista di documenti.
        """
        return self._vector_database.get_all_documents()

    async def _save_file(self, file: File):
        """
        Salva il file nel filesystem.

        Param:
        - file: File - Il file da salvare.

        Returns:
        - str: Il percorso completo del file salvato.
        """
        await file.seek(0)
        contents = await file.read()

        path = file.filename
        file_path = self.get_full_path(path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)
            logger.info(f"File salvato in {file_path}")

        return file_path

    @abstractmethod
    def _load_split_file(self, file_path: str):
        """
        Carica il file e lo divide in chunk.

        Param:
        - file: File - Il file da caricare.
        - file_path: str - Il percorso completo del file salvato.

        Returns:
        - list: Una lista di chunk.
        """

    async def add_document(self, file: File, token: str):
        """
        Salva il file nel filesystem,
        lo carica e lo divide in chunk,
        lo salva nel database vettoriale,
        e invia una richiesta al database API per caricarne il riferimento.

        Param:
        - file: File - Il file da caricare.

        Returns:
        - bool: True se il file è stato caricato correttamente, False altrimenti.

        Raises:
        - HTTPException: Se il file è già esistente o se si verifica un errore durante il caricamento.
        """

        file_path = await self._save_file(file)
        chunks = await self._load_split_file(file_path)

        self._vector_database.add_documents(chunks)

        request_body = {
            "file_path": file_path,
            "title": file.filename,
            "owner_email": "test@test.it",
            "uploaded_at": datetime.now().isoformat(),
        }
        upload_request_response = requests.post(
            "http://database-api:8000/documents",
            data=json.dumps(request_body),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        print("upload_request_response:", upload_request_response.json())
        match upload_request_response.status_code:
            case 201:
                print(f"Documento caricato e splittato in {len(chunks)} chunk")
                return True
            case 400:
                raise HTTPException(
                    status_code=400,
                    detail=f"Documento già esistente",
                )
            case 500:
                raise HTTPException(
                    status_code=500,
                    detail=f"Errore nel caricare e processare file",
                )
            case _:
                raise HTTPException(
                    status_code=500,
                    detail=f"Errore nel caricare e processare file",
                )
        return False

    async def delete_document(
        self, file_id: str, file_path: str, token: str, current_password: str
    ):
        """
        Elimina il file dal filesystem e dal database vettoriale e dal database.

        Param:
        - file_path: str - Il percorso completo del file da eliminare.

        Returns:
        - bool: True se il file è stato eliminato correttamente, False altrimenti.
        """

        # rimuovi da Database API
        delete_req = requests.delete(
            f"http://database-api:8000/documents",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={
                "admin": {
                    "current_password": current_password,
                },
                "file": {
                    "id": file_id,
                },
            },
        )

        match delete_req.status_code:
            case 204:
                print(f"Documento eliminato correttamente")
            case 400:
                raise HTTPException(
                    status_code=400,
                    detail=f"Documento non trovato",
                )
            case 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Documento non trovato",
                )
            case 401:
                raise HTTPException(
                    status_code=401,
                    detail=f"Password errata",
                )
            case 500:
                raise HTTPException(
                    status_code=500,
                    detail=f"Errore nel caricare e processare file {delete_req.text}",
                )
            case _:  # Add this default case
                raise HTTPException(
                    status_code=500,  # As expected by the test for default errors
                    detail="Errore nel caricare e processare file",  # As expected by the test
                )

        # rimuovi da filesystem
        if os.path.isfile(file_path) and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"File {file_path} non trovato: {e}",
                )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_path} non trovato",
            )

        # rimuovi da database vettoriale
        self._vector_database.delete_document(file_path)


class TextFileManager(FileManager):
    async def _load_split_file(self, file_path: str):
        loader = TextLoader(file_path, encoding="utf-8")
        data = loader.load()
        chunks = self._splitter.split_documents(data)
        return chunks


class PdfFileManager(FileManager):
    async def _load_split_file(self, file_path: str):
        loader = PyPDFLoader(file_path, mode="single")
        data = loader.load()
        chunks = self._splitter.split_documents(data)
        return chunks


class StringManager(FileManager):
    async def _load_split_file(self, faq: schemas.FAQ):
        data = Document(
            page_content=f"Domanda: {faq.question}\nRisposta: {faq.answer}",
            metadata={"source": "faqs", "faq_id": faq.id},
        )
        print("[StringManager] data:", data)
        chunks = self._splitter.split_documents([data])
        return chunks

    async def add_faq(self, faq: schemas.FAQBase, token: str):
        """
        Divide la faq in chunk,
        la salva nel database vettoriale.

        Param:
        - faq: schemas.FAQ - La faq da caricare.
        """
        print("[StringManager] adding faq:", faq)

        ris = requests.post(
            "http://database-api:8000/faqs",
            headers={"Authorization": f"Bearer {token}"},
            json=faq.dict(),
        )
        faq_json = ris.json()

        if ris.status_code != 201:
            raise HTTPException(status_code=ris.status_code, detail=ris.json())

        faq_db = schemas.FAQ(
            id=faq_json["id"],
            title=faq.title,
            question=faq.question,
            answer=faq.answer,
        )
        chunks = await self._load_split_file(faq_db)
        self._vector_database.add_documents(chunks)

        return faq_db

    async def delete_faq(self, faq: schemas.FAQDelete, token: str):
        """
        Elimina la faq dal database vettoriale e dal database.

        Param:
        - faq: schemas.FAQDelete - La faq da eliminare.
        - token: str - Il token di autenticazione.
        """
        # rimuovi da Database API
        delete_req = requests.delete(
            f"http://database-api:8000/faqs/{faq.id}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={
                "current_password": faq.admin_password,
            },
        )

        match delete_req.status_code:
            case 204:
                print(f"FAQ eliminata correttamente")
            case 404:
                raise HTTPException(
                    status_code=400,
                    detail=f"FAQ non trovata",
                )
            case 401:
                raise HTTPException(
                    status_code=401,
                    detail=f"Password errata",
                )
            case _:
                raise HTTPException(
                    status_code=500,
                    detail=f"Errore nel caricare e processare file {delete_req.text}",
                )

        # rimuovi da database vettoriale
        self._vector_database.delete_faq(faq.id)

    async def update_faq(self, faq: schemas.FAQ, token: str):
        """
        Aggiorna la faq nel database.

        Param:
        - faq: schemas.FAQUpdate - La faq da aggiornare.
        - token: str - Il token di autenticazione.
        """
        # rimuovi da Database API
        update_req = requests.patch(
            f"http://database-api:8000/faqs/{faq.id}",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={
                "title": faq.title,
                "question": faq.question,
                "answer": faq.answer,
            },
        )

        match update_req.status_code:
            case 200:
                print(f"FAQ aggiornata correttamente")
            case 404:
                raise HTTPException(
                    status_code=400,
                    detail=f"FAQ non trovata",
                )
            case 401:
                raise HTTPException(
                    status_code=401,
                    detail=f"Password errata",
                )
            case _:
                raise HTTPException(
                    status_code=500,
                    detail=f"Errore nel caricare e processare file {update_req.text}",
                )

        self._vector_database.delete_faq(faq.id)
        chunks = await self._load_split_file(faq)
        self._vector_database.add_documents(chunks)


def get_file_manager(file: UploadFile = None):
    """
    Restituisce il file manager in base al tipo di file.

    Param:
    - file: File - Il file da gestire.

    Returns:
    - FileManager: Il file manager appropriato.
    """
    if file is None:
        return TextFileManager()
    match file.content_type:
        # TODO: capire se catcha anche le stringhe(=faq)
        case "text/plain":
            return TextFileManager()
        case "application/pdf":
            return PdfFileManager()
        case _:
            raise ValueError("Unsupported file type")


def get_file_manager_by_extension(file_path: str = None):
    """
    Restituisce il file manager in base all'estensione del file.

    Param:
    - file_path: str - Il percorso del file da gestire.

    Returns:
    - FileManager: Il file manager appropriato.
    """
    if file_path is None:
        return StringManager()
    _, ext = os.path.splitext(file_path)
    match ext:
        case ".txt":
            return TextFileManager()
        case ".pdf":
            return PdfFileManager()
        case _:
            raise ValueError("Unsupported file type")
