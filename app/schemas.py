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


class DocumentBase(BaseModel):
    id: str
    title: str


class Document(DocumentBase):
    updated_at: datetime
    content: str


class DocumentDelete(DocumentBase):
    token: str
    current_password: str


class FAQBase(BaseModel):
    title: str
    question: str
    answer: str


class FAQ(FAQBase):
    id: str


class FAQDelete(BaseModel):
    id: str
    admin_password: str
