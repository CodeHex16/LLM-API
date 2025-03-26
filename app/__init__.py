from fastapi import FastAPI
from app.routes import llm

app = FastAPI(
    title="LLM API",
    description="LLM API for the Suppl-AI project",
    version="0.2",
)

app.include_router(llm.router)
