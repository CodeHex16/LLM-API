from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from app.services.llm_service import LLM, OpenAI
from typing import List
import os

# from app.services.chroma_services import embedding
from app.services.file_manager_service import (
    get_file_manager,
    get_file_manager_by_extension,
)

import app.schemas as schemas


router = APIRouter(
    tags=["documents"],
    prefix="/documents",
)


@router.post("/upload_file")
async def upload_file(files: List[UploadFile], token: str):
    """
    Carica il file nel database vettoriale

    Args:
    - files (List[UploadFile]): I file da caricare. Devono essere file di testo o PDF.

    Raises:
    - HTTPException: Se il file non è valido o se si verifica un errore durante il caricamento.
    - HTTPException: Se il file esiste già nel database vettoriale.
    - HTTPException: Se si verifica un errore durante il caricamento e l'elaborazione del file.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    processed_files_count = 0
    errors = []

    for file in files:
        if not file.filename:
            errors.append({"filename": "N/A", "detail": "No filename provided"})
            continue
        if not file.filename.endswith((".txt", ".pdf")):
            errors.append(
                {"filename": file.filename, "detail": "Only txt/pdf files are allowed"}
            )
            continue
        if file.content_type not in ("text/plain", "application/pdf"):
            errors.append(
                {
                    "filename": file.filename,
                    "detail": f"Invalid content type: {file.content_type}",
                }
            )
            continue

        try:
            print(f"Processing file: {file.filename}")
            file_manager = get_file_manager(file)
            await file_manager.add_document(file, token)  # Passa il singolo file
            processed_files_count += 1
        except HTTPException as e:
            print(f"HTTPException processing {file.filename}: {e.detail}")
            errors.append(
                {
                    "filename": file.filename,
                    "detail": e.detail,
                    "status_code": e.status_code,
                }
            )
        except Exception as e:
            print(f"Exception processing {file.filename}: {e}")
            errors.append(
                {
                    "filename": file.filename,
                    "detail": f"Internal server error during processing: {str(e)}",
                }
            )

    if processed_files_count == 0 and errors:
        first_error = errors[0]
        raise HTTPException(
            status_code=first_error.get("status_code", 400),
            detail=f"Failed to process any files. First error on '{first_error.get('filename', 'N/A')}': {first_error.get('detail', 'Unknown error')}",
        )
    elif errors:
        return {
            "message": f"Processed {processed_files_count} files with {len(errors)} errors.",
            "processed_count": processed_files_count,
            "errors": errors,
        }
    else:
        return {
            "message": f"Successfully uploaded and processed {processed_files_count} files."
        }


@router.delete("/delete_file")
async def delete_file(fileDelete: schemas.DocumentDelete):
    print("delete file title:", fileDelete)
    file_manager = get_file_manager_by_extension(fileDelete.title)
    if file_manager is None:
        raise HTTPException(status_code=400, detail="File manager not found")
    try:
        file_path = file_manager.get_full_path(fileDelete.title)
        print("file path:", file_path)
        await file_manager.delete_document(
            fileDelete.id, file_path, fileDelete.token, fileDelete.current_password
        )
    except HTTPException as e:
        match e.status_code:
            case 404:
                print("error detail:", e.detail)
                raise HTTPException(
                    status_code=404,
                    detail="Document not found",
                )
            case 500:
                print("error detail:", e.detail)
                raise HTTPException(
                    status_code=500,
                    detail="Error in deleting file",
                )
            case _:
                print("error detail:", e.detail)
                raise HTTPException(
                    status_code=500,
                    detail="Error in deleting file",
                )

    except Exception as e:
        print("error detail:", e)
        raise HTTPException(
            status_code=500,
            detail="Error in deleting file",
        )

    return {"message": "File deleted successfully"}


@router.get("/get_documents")
def get_documents():
    """
    Ottiene la lista dei documenti dal database.

    Args:
    - token (str): Il token di autenticazione dell'utente.

    Raises:
    - HTTPException: Se si verifica un errore durante il recupero dei documenti.
    """

    return os.listdir("/data/documents")
