import os
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Percorso dove salvare i dati di Chroma
DOCUMENTS_FOLDER = "documenti"
CHROMA_DB_PATH = "chroma_db"


def vectorize_documents():
    embedding_function = OpenAIEmbeddings()

    documents = []
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)

    if not documents:
        print("Nessun file .txt trovato.")
        return

    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(
        texts, embedding_function, persist_directory=CHROMA_DB_PATH
    )

    print(f"✅ {len(texts)} documenti vettorializzati e salvati in Chroma.")


def embedding(query):
    # Verifica se ci sono documenti
    if has_documents():
        print(f"Il database contiene {count_documents()} documenti.")
    else:
        print("Il database è vuoto. Esecuzione della vettorizzazione...")
        vectorize_documents()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())
    results = db.similarity_search(query, k=3)
    return results


def has_documents():
    """Verifica se il database Chroma contiene documenti."""
    try:
        db = Chroma(
            persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings()
        )
        count = db._collection.count()
        return count > 0
    except Exception as e:
        print(f"Errore nel verificare il database: {e}")
        return False


def count_documents():
    """Restituisce il numero di documenti nel database Chroma."""
    try:
        db = Chroma(
            persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings()
        )
        return db._collection.count()
    except Exception as e:
        print(f"Errore nel conteggio dei documenti: {e}")
        return 0
