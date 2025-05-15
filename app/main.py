from fastapi import FastAPI
from app.routes import llm, documents, faq
import os

if os.environ.get("OPENAI_API_KEY") is None:
    print("WARNING: OPENAI_API_KEY environment variable is not set. Please set it before running the application.")
    exit(1)

    
app = FastAPI(
    title="LLM API",
    description="LLM API for the Suppl-AI project",
    version="0.2",
)

app.include_router(llm.router)
app.include_router(documents.router)
app.include_router(faq.router)
