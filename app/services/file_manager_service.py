from abc import ABC, abstractmethod
from fastapi import Depends, File
from langchain.text_splitter import CharacterTextSplitter
import logging

logger = logging.getLogger(__name__)
import os


from app.services.vector_database_service import get_vector_database


class FileManager(ABC):
    def __init__(self, vector_database=Depends(get_vector_database)):
        self.vector_database = vector_database
        self.splitter = CharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
        )

    
    def _get_file_path(filename: str) -> str:
        documents_dir = os.environ.get("DOCUMENTS_DIR", "/data/documents")
        return os.path.join(documents_dir, filename)
    
    async def _save_file(self, file: File):
        contents = await file.read()
        file_path = self._get_file_path(file.filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)
            logger.info(f"File salvato in {file_path}")

        return file_path

    @abstractmethod
    def _prepare_file(self, file: File):
        """
        Prepares the file for processing.
        """
        

    def add_document(self, file: File):
        """
        Prepares the context for the vector database.
        """
        
        #Salva il file
        self._save_file(file)        
        
        # convertire in text il documento
        self._prepare_file(file)

        # dividere in chunck il documento
        # aggiungere al database
    
        chunks = text_splitter.split_documents(documents)
        logger.info(
            f"Documento caricato e splittato in {len(chunks)} chunk"
        )
        return texts


class TextFileManager(FileManager):
    async def _prepare_file(self, file: File):
        """
        Prepares the text file for processing.
        """
        # Read the contents of the file
        contents = await file.read()
        
        # Convert the contents to text
        text = contents.decode("utf-8")
        
        # Split the text into chunks
        documents = self.splitter.split_text(text)
        
        return documents
    pass

class CsvFileManager(FileManager):
    async def _prepare_file(self, file: File):
        """
        Prepares the CSV file for processing.
        """
        # Read the contents of the file
        contents = await file.read()

        # Convert the contents to text
        text = contents.decode("utf-8")

        # Split the text into chunks
        documents = self.splitter.split_text(text)

        return documentsp

    pass

# class StringManager(FileManager):
#     pass


class PdfFileManager(FileManager):
    pass

def get_file_manager(file: File):
    match file.content_type:
        case "text/plain":
            return TextFileManager()
        case "application/pdf":
            return PdfFileManager()
        case "text/csv":
            return CsvFileManager()
        case _:
            raise ValueError("Unsupported file type")
