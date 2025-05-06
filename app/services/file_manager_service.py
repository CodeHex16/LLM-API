from abc import ABC, abstractmethod
from fastapi import Depends, File, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from fastapi import HTTPException
import os
import logging
import json
import requests
from datetime import datetime
from bson import ObjectId

from app.services.vector_database_service import get_vector_database, VectorDatabase

logger = logging.getLogger(__name__)


class FileManager(ABC):
    def __init__(self):
        self.vector_database = get_vector_database()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
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

        self.vector_database.add_documents(chunks)

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
        # rimuovi da filesystem
        print("INIZIO RIMOZIONE DOCUMENTO")
        print("file_path:", file_path)
        print("os.path.isfile(file_path):", os.path.isfile(file_path))
        print("ls -la /data/documents", os.listdir("/data/documents"))

        if os.path.isfile(file_path) and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                raise HTTPException(
                    status_code=404,
                    detail=f"File {file_path} non trovato: {e}",
                )
            logger.info(f"File {file_path} eliminato")
        else:
            raise HTTPException(
                status_code=404,
                detail=f"File {file_path} non trovato",
            )

        # rimuovi da database vettoriale
        self.vector_database.delete_document(file_path)

        # rimuovi da Database API
        print("[LLM API] file_id: pre DELETE: ", file_id, type(file_id))
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
            case 500:
                raise HTTPException(
                    status_code=500,
                    detail=f"Errore nel caricare e processare file",
                )


class TextFileManager(FileManager):
    async def _load_split_file(self, file_path: str):
        loader = TextLoader(file_path, encoding="utf-8")
        data = loader.load()
        chunks = self.splitter.split_documents(data)
        return chunks


class PdfFileManager(FileManager):
    async def _load_split_file(self, file_path: str):
        loader = PyPDFLoader(file_path, mode="single")
        data = loader.load()
        chunks = self.splitter.split_documents(data)
        return chunks


class StringManager(FileManager):
    pass


def get_file_manager(file: UploadFile):
    """
    Restituisce il file manager in base al tipo di file.

    Param:
    - file: File - Il file da gestire.

    Returns:
    - FileManager: Il file manager appropriato.
    """
    match file.content_type:
        case "text/plain":
            return TextFileManager()
        case "application/pdf":
            return PdfFileManager()
        case _:
            raise ValueError("Unsupported file type")


def get_file_manager_by_extension(file_path: str):
    """
    Restituisce il file manager in base all'estensione del file.

    Param:
    - file_path: str - Il percorso del file da gestire.

    Returns:
    - FileManager: Il file manager appropriato.
    """
    _, ext = os.path.splitext(file_path)
    match ext:
        case ".txt":
            return TextFileManager()
        case ".pdf":
            return PdfFileManager()
        case _:
            raise ValueError("Unsupported file type")
