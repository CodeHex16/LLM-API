import pytest
from unittest.mock import patch, MagicMock
import os
import shutil
import pytest
import time
import uuid

from app.services.chroma_services import (
    set_chroma_db_path,
    set_documents_folder,
    vectorize_documents,
    embedding,
    has_documents,
    count_documents,
)

DOCUMENTS_FOLDER = "tests/.cache/documenti"
CHROMA_DB_PATH =   "tests/.cache/chroma_db"
set_documents_folder(DOCUMENTS_FOLDER)
set_chroma_db_path(CHROMA_DB_PATH)


SAMPLE_TEXT = """Formaggio Semi Stagionato Tumarrano 'Latte misto' - Valsana FormaggiCarni & SalumiPesceVegetaliCereali & CoGastronomiaCondimentiDolci & Abbinamenti Diventa cliente Toggle Nav 0 IT EN FR IT â–¾ IT EN FR  Diventa cliente Toggle Nav 0 IT EN FR IT â–¾ IT EN FR FormaggiCarni & SalumiPesceVegetaliCereali & CoGastronomiaCondimentiDolci & Abbinamenti Home Formaggi Formaggio Semi Stagionato Tumarrano 'Latte misto' Il prodotto Formaggio Semi Stagionato Tumarrano 'Latte misto' Formaggio a latte misto ovino e vaccino prodotto in Sicilia Codice: 21436 Paese e Luogo di Origine: Italia - Sicilia Tipo di Latte: Ovino, Vaccino   Peso: 13 kg circa Ordine Minimo: 1/8 Aggiungi ai preferiti Scheda in PDF Condividi via mail Caratteristiche del prodotto Maggiori Informazioni Descrizione Latte ovino e latte vaccino pastorizzati raccolto presso le stalle dei soci della Cooperativa Agricola Tumarrano di Cammarata (AG) Aspetto La pasta si presenta di colore avorio, la crosta è canestrata e di colore marrone chiaro Sapore Dolce, leggermente piccante con notevole complessità . Ha note agrumate, di burro cotto, di fieno e frutta tostata. Stagionatura Almeno 3 mesi Curiosità La Cooperativa Agricola Tumarrano nasce nel 1971 a Cammarata (AG), dall'esigenza di riunire diversi piccoli allevatori di ovini e bovini, principalmente per la macellazione della carne. Nel 2002 viene introdotta anche la raccolta e la commercializzazione del latte. Nel 2009, con la costruzione del nuovo caseificio di proprietà , inizia la caseificazione e viene abbandonata la macellazione. Oggi i soci sono 60, tutti che conferiscono latte ovino di Razza Sarda, con cui sii producono soprattutto ricotta e pecorino semistagionato. Lo storico presidente della cooperativa Francesco Madonia racconta che il Pecorino Tumarrano è chiamato dalla gente del luogo l'Oro di Cammarata per il colore giallo carico della pasta Maggiori Informazioni Maggiori Informazioni Ingredienti LATTE ovino, LATTE vaccino, sale, caglio, fermenti lattici Allergeni ingredienti Latte e prodotti a base di latte Altri allergeni Uovo e prodotti a base di uovo Peso 13 kg circa Confezione Confezionata sottovuoto Modalità di Conservazione (prodotto confezionato) Consevare a temperatura uguale o inferiore a +4 Â°C Condizioni di impiego Asportare la crosta prima del consumo. Da vendere previo frazionamento Paese di origine ingrediente primario Italia Dichiarazione Nutrizionale Valore energetico: 1916 kJ / 390 kcal Grassi: 32 g di cui saturi: 24 g Carboidrati: 1,6 g di cui zuccheri: 0 g Proteine: 24 g Sale: 1,9 g Valori riferiti a 100 g di prodotto Il produttore Cooperativa Agricola Tumarrano - Cammarata (AG) Selezionato perchè La Cooperativa Agricola Tumarrano nasce nel 1971 a Cammarata (AG), dall'esigenza di riunire diversi piccoli allevatori di ovini e bovini. Nel 2009, con la costruzione del nuovo caseificio di proprietà , inizia la produzione di ricotta e pecorino semistagionato, esclusivamente con latte raccolto dai soci. Oggi i soci sono 60, tutti che conferiscono latte ovino di Razza Sarda. Lo storico presidente della cooperativa Francesco Madonia racconta che il Pecorino Tumarrano è chiamato dalla gente del luogo "l'Oro di Cammarata" per il colore giallo carico della pasta. Dello stesso produttore Formaggio Semi Stagionato Tumarrano 'Latte misto' Vedi tutti i prodotti Tutte le categorie Formaggi Carni & Salumi Pesce Vegetali Cereali & Co Gastronomia Condimenti Dolci & Abbinamenti Tutte le categorieâ–¾Tutte le categorieFormaggiCarni & SalumiPesceVegetaliCereali & CoGastronomiaCondimentiDolci & Abbinamenti Cercati di recente Prodotti Vedi Tutti 0 No Result Cerca Diventa cliente Newsletter Iscriviti per ricevere i nostri aggiornamenti su nuovi prodotti, eventi e promozioni Newsletter Iscriviti Chi siamo Azienda Contatti Blog I nostri partner Produttori Punti vendita Termini e condizioni Condizioni di acquisto Privacy Policy Cookie Policy Scelta lingua EN FR Scelta lingua â–¾ Scelta lingua EN FR Newsletter Esprimo il consenso al trattamento dei dati personali per le finalità descritte nell'informativa sulla privacy Invia Â© VALSANA S.R.L. - All rights reserved - Tel. (+39) 0438 1883 125 - Fax (+39) 0438 64976 - email: valsana@valsana.it - Via degli Olmi 16 - 31010 Godega di Sant'Urbano (TV) - P.IVA 01949710261 - C.F. 00548700244 - Reg. Imprese Treviso N. 00548700244 - Cap.Soc. 52.000 i.v. Le tue preferenze relative al consenso per le tecnologie di tracciamento."""

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    # Pulizia e preparazione dell'ambiente
    # if os.path.exists(DOCUMENTS_FOLDER):
    #     shutil.rmtree(DOCUMENTS_FOLDER)

    # if os.path.exists(CHROMA_DB_PATH):
    #     shutil.rmtree(CHROMA_DB_PATH)

    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    sample_file_path = os.path.join(DOCUMENTS_FOLDER, "sample.txt")
    with open(sample_file_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)

    yield
    # Cleanup after tests

def test_vectorize_documents_creates_chroma_db():
    vectorize_documents()
    assert os.path.exists(CHROMA_DB_PATH), "La cartella del DB non è stata creata"
    assert has_documents() is True, "Il database dovrebbe contenere documenti"
    assert count_documents() > 0, "Il database dovrebbe avere almeno un documento"

def test_embedding_returns_results():
    query = "esempio documento"
    results = embedding(query)
    assert isinstance(results, list)
    assert len(results) > 0, "La funzione embedding dovrebbe restituire dei risultati"

def test_has_documents_and_count_documents():
    assert has_documents() is True
    assert count_documents() > 0

def test_has_documents_empty_db():
    # Crea un nuovo database vuoto
    new_db_path = "tests/.cache/"+str(uuid.uuid4())+"_chroma_db"
    set_chroma_db_path(new_db_path)
    assert has_documents() is False, "Il database dovrebbe essere vuoto"
    assert count_documents() == 0, "Il conteggio dei documenti dovrebbe essere zero"