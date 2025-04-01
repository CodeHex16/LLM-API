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
        #manda il file ad una funzione
        file_manager = get_file_manager(file)
        await file_manager.add_document(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel caricare e processare file: {e}")

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
