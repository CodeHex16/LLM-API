from fastapi import APIRouter, HTTPException

from app.services.file_manager_service import (
    get_file_manager_by_extension,
)

import app.schemas as schemas


router = APIRouter(
    tags=["faqs"],
    prefix="/faqs",
)


@router.post("")
async def create_faq(faq: schemas.FAQBase, token: str):
    """
    Crea una nuova FAQ.

    ### Args:
    * **faq (schemas.FAQCreate)**: I dati della FAQ da creare.
    * **token (str)**: Il token di autenticazione.

    ### Raises:
    * **HTTPException.400_BAD_REQUEST**: Se non sono stati forniti dati per la creazione della FAQ.
    * **HTTPException.500_INTERNAL_SERVER_ERROR**: Se si verifica un errore durante la creazione della FAQ.
    """
    if not faq:
        raise HTTPException(status_code=400, detail="No data provided for creation")

    file_manager = get_file_manager_by_extension()
    if file_manager is None:
        raise HTTPException(status_code=500, detail="File manager not found")

    faq_db = await file_manager.add_faq(faq, token)

    return {"faq": faq_db, "message": "FAQ created successfully"}


@router.delete("")
async def delete_faq(faq: schemas.FAQDelete, token: str):
    """
    Elimina una FAQ esistente.

    ### Args:
    * **faq (schemas.FAQDelete)**: I dati della FAQ da eliminare.

    ### Raises:
    * **HTTPException.400_BAD_REQUEST**: Se non sono stati forniti dati per l'eliminazione.
    * **HTTPException.404_NOT_FOUND**: Se la FAQ non esiste.
    * **HTTPException.500_INTERNAL_SERVER_ERROR**: Se si verifica un errore durante l'eliminazione della FAQ.
    """
    if not faq:
        raise HTTPException(status_code=400, detail="No data provided for deletion")

    file_manager = get_file_manager_by_extension()
    if file_manager is None:
        raise HTTPException(status_code=500, detail="File manager not found")

    await file_manager.delete_faq(faq, token)
    return {"message": "FAQ deleted successfully"}


@router.put("")
async def update_faq(faq: schemas.FAQ, token: str):
    """
    Aggiorna una FAQ esistente.

    ### Args:
    * **faq (schemas.FAQUpdate)**: I dati della FAQ da aggiornare.
    * **faq_id (str)**: L'ID della FAQ da aggiornare.

    ### Raises:
    * **HTTPException.400_BAD_REQUEST**: Se non sono stati forniti dati per l'aggiornamento.
    * **HTTPException.404_NOT_FOUND**: Se la FAQ non esiste.
    * **HTTPException.500_INTERNAL_SERVER_ERROR**: Se si verifica un errore durante l'aggiornamento della FAQ.
    """
    if not faq:
        raise HTTPException(status_code=400, detail="No data provided for update")

    file_manager = get_file_manager_by_extension()
    if file_manager is None:
        raise HTTPException(status_code=500, detail="File manager not found")

    faq_db = await file_manager.update_faq(faq, token)
    return {"faq": faq_db, "message": "FAQ updated successfully"}
