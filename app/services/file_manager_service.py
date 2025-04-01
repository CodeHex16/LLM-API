from abc import ABC, abstractmethod
from fastapi import Depends, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,TextLoader

import logging
from datetime import datetime

from app.services.vector_database_service import get_vector_database, VectorDatabase

logger = logging.getLogger(__name__)
import os


class FileManager(ABC):
    def __init__(self):
        self.vector_database = get_vector_database()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
        )

    def _get_file_path(self, filename: str) -> str:
        documents_dir = os.environ.get("DOCUMENTS_DIR", "/data/documents")
        return os.path.join(documents_dir, filename)

    async def _save_file(self, file: File):
        await file.seek(0)
        contents = await file.read()

        path = file.filename
        file_path = self._get_file_path(path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)
            logger.info(f"File salvato in {file_path}")

        return file_path

    @abstractmethod
    def _prepare_file(self, file: File, file_path: str):
        """
        Prepares the file for processing.
        """

    async def add_document(self, file: File):
        """
        Salva il file nel filesystem, poi lo prepara per l'embedding nel database vettoriale e lo salva nel database.
        """

        file_path = await self._save_file(file)
        chunks = await self._prepare_file(file_path)

        self.vector_database.add_documents(chunks)

        #aggiunta al db mongo
        import requests
        upload_response = requests.post("http://database-api:8000/documents/upload",
                json={"id":1,       
                  "file_path": file_path, 
                    "title": file.filename.split(".")[0],
                    "owner_email": "uncazzo",
                    "uploaded_at": datetime.now().isoformat()
                    })
        print(upload_response.json())

        # TODO: da capire
        # chunks = text_splitter.split_documents(documents)
        print(f"Documento caricato e splittato in {len(chunks)} chunk")

        return True


class TextFileManager(FileManager):
    async def _prepare_file(self, file_path: str):

        loader = TextLoader(file_path, encoding="utf-8")
        data = loader.load()
        chunks = self.splitter.split_text(data)

        return chunks


class PdfFileManager(FileManager):
    async def _prepare_file(self, file_path: str):
        loader = PyPDFLoader(file_path, mode="single")
        data = loader.load()
        documents = self.splitter.split_documents(data)
        return documents


class StringManager(FileManager):
    pass


def get_file_manager(file: File):
    match file.content_type:
        case "text/plain":
            return TextFileManager()
        case "application/pdf":
            return PdfFileManager()
        case _:
            raise ValueError("Unsupported file type")


# class CsvFileManager(FileManager):
#     async def _prepare_file(self, file: File, file_path: str):
#         """
#         """
#         print("******* MEDOTO 1 ****************")
#         await file.seek(0)
#         contents = await file.read()
#         text = contents.decode("utf-8")
#         documents = self.splitter.split_text(text)
#         print("chunks", documents[:5])
#         print()
#         print()
#         print("******* MEDOTO 2 ****************")
#         loader = CSVLoader(file_path, encoding="utf-8")
#         data = loader.load()
#         documents = self.splitter.split_text(text)
#         print("chunks", documents[:5])

#         return documents
