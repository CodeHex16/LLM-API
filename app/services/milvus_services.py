import os
from pymilvus import MilvusClient, model
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Configurazione del database Milvus
MILVUS_DB_FILE = "milvus_demo.db"
MILVUS_COLLECTION_NAME = "document_embeddings"

# Cartella dei documenti
DOCUMENTS_FOLDER = "documenti"

def connection():
    client = MilvusClient(MILVUS_DB_FILE)

    if client.has_collection(collection_name=MILVUS_COLLECTION_NAME):
        client.drop_collection(collection_name=MILVUS_COLLECTION_NAME)

    client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        dimension=1536
    )

    return client

def vectorize_documents():
    client = connection()

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

    vectors = embedding_function.embed_documents([t.page_content for t in texts])
    
    data = [
        {"id": i, "vector": vectors[i], "text": texts[i].page_content}
        for i in range(len(vectors))
    ]
    client.insert(collection_name=MILVUS_COLLECTION_NAME, data=data)

    print(f"{len(texts)} documenti vettorializzati e salvati in Milvus.")

def embedding(query):
    client = MilvusClient(MILVUS_DB_FILE)
    embedding_function = OpenAIEmbeddings()
    query_vector = embedding_function.embed_query(query)
    res = client.search(
        collection_name=MILVUS_COLLECTION_NAME,
        data=[query_vector],
        limit=3,
        output_fields=["text"]
    )
    return res