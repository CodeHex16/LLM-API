from pydantic import BaseModel, field_validator
from typing import List
from datetime import datetime
from uuid import UUID


class Message(BaseModel):
    sender: str
    content: str


class Question(BaseModel):
    question: str
    messages: List[Message] = []


class Context(BaseModel):
    context: str


class Document(BaseModel):
    id: UUID
    title: str
    updated_at: datetime
    content: str


class DocumentDelete(BaseModel):
    id: str
    title: str
    token: str
    current_password: str


class FAQ(BaseModel):
    id: str
    title: str
    question: str
    answer: str

class FAQCreate(BaseModel):
    title: str
    question: str
    answer: str
    
class FAQDelete(BaseModel):
    id: str
    admin_password: str