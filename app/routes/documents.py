from fastapi import APIRouter, HTTPException, UploadFile
from typing import List
import os

from app.services.file_manager_service import (
    get_file_manager,
    get_file_manager_by_extension,
)

import app.schemas as schemas


router = APIRouter(
    tags=["documents"],
    prefix="/documents",
)


@router.post("")
async def upload_file(files: List[UploadFile], token: str):
    """
    Carica il file nel database vettoriale.

    ### Args:
    * **files (List[UploadFile])**: I file da caricare. Devono essere file di testo o PDF.

    ### Raises:
    * **HTTPException.400_BAD_REQUEST**: Se non sono stati forniti file o se i file non sono di tipo testo o PDF.
    * **HTTPException.500_INTERNAL_SERVER_ERROR**: Se si verifica un errore durante il caricamento dei file.
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


@router.delete("")
async def delete_file(fileDelete: schemas.DocumentDelete):
    """
    Elimina un file dal database.

    ### Args:
    * **fileDelete (schemas.DocumentDelete)**: Il file da eliminare. Deve contenere il titolo, il token e la password corrente.

    ### Raises:
    * **HTTPException.400_BAD_REQUEST**: Se il file non esiste o se si verifica un errore durante l'eliminazione.
    * **HTTPException.500_INTERNAL_SERVER_ERROR**: Se si verifica un errore durante l'eliminazione del file.
    """
    print("delete file title:", fileDelete)
    file_manager = get_file_manager_by_extension(fileDelete.title)
    if file_manager is None:
        raise HTTPException(status_code=400, detail="File manager not found")

    file_path = file_manager.get_full_path(fileDelete.title)
    print("file path:", file_path)
    await file_manager.delete_document(
        fileDelete.id, file_path, fileDelete.token, fileDelete.current_password
    )

    return {"message": "File deleted successfully"}


@router.get("")
def get_documents():
    """
    Restituisce il numero di documenti e i loro nomi.

    ### Returns:
    * **int**: Il numero di documenti.
    * **List[str]**: I nomi dei documenti.
    * **List[str]**: I nomi dei file nella directory /data/documents.
    """
    file_manager = get_file_manager()
    return file_manager.get_documents_number(), file_manager.get_documents(),os.listdir("/data/documents")
