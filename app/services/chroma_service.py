import os
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from config import settings

# TODO: DA CANCELLARE


def vectorize_documents():
    embedding_function = OpenAIEmbeddings()

    documents = []
    for filename in os.listdir(settings.settings.DOCUMENTS_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(settings.DOCUMENTS_FOLDER, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)

    if not documents:
        print("Nessun file .txt trovato.")
        return

    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    db = Chroma.from_documents(
        texts,
        embedding_function,
        persist_directory=settings.VECTOR_DB_PERSIST_DIRECTORY,
    )

    print(f"✅ {len(texts)} documenti vettorializzati e salvati in Chroma.")
