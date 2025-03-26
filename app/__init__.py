from fastapi import FastAPI
from app.routes import llm

app = FastAPI()

app.include_router(llm.router)
