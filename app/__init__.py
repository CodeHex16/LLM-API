from fastapi import FastAPI
from app.routes.milvus_routes import milvus_router
from app.routes.chroma_routes import chroma_router

app = FastAPI()

app.include_router(milvus_router, prefix="/milvus")
app.include_router(chroma_router, prefix="/chroma")
