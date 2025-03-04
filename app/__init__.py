from fastapi import FastAPI
from app.routes import chroma

app = FastAPI()

app.include_router(chroma.router)
