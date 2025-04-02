from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from app.services.llm_service import LLM, OpenAI
import os

# from app.services.chroma_services import embedding
from app.services.file_manager_service import get_file_manager

import app.schemas as schemas

router = APIRouter(
    tags=["documents"],
    prefix="/documents",
)

# TODO: gestire auth????????? siamo pazzi


@router.post("/upload_file")
async def upload_file(file: UploadFile):
    """
    Carica il file nel database vettoriale
    
    Args:
        file (UploadFile): Il file da caricare. Deve essere un file di testo o PDF.

    Raises:
        HTTPException: Se il file non è valido o se si verifica un errore durante il caricamento.
        HTTPException: Se il file esiste già nel database vettoriale.
        HTTPException: Se si verifica un errore durante il caricamento e l'elaborazione del file.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if (
        not file.filename.endswith(".txt")
        and not file.filename.endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400, detail="Only txt/pdf files are allowed"
        )
    if (
        file.content_type != "text/plain"
        and file.content_type != "application/pdf"
    ):
        raise HTTPException(
            status_code=400, detail="Only txt/pdf content type is allowed"
        )

    try:
        # manda il file ad una funzione
        file_manager = get_file_manager(file)
        await file_manager.add_document(file)
    except HTTPException as e:
        match e.status_code:
            case 400:
                print("error detail:", e.detail)
                raise HTTPException(
                    status_code=400,
                    detail="Document already exists",
                )
            case 500:
                print("error detail:", e.detail)
                raise HTTPException(
                    status_code=500,
                    detail="Error in uploading and processing file",
                )

    return {"message": "File uploaded successfully"}


@router.delete("/delete_file")
async def delete_file(filename: str):
    # file_path = get_file_path(filename)

    # if os.path.exists(file_path):
    #     os.remove(file_path)
    #     return {"message": "File deleted successfully"}
    # else:
    #     raise HTTPException(status_code=404, detail="File not found")
    pass
