from pydantic import BaseModel
from typing import List
from datetime import datetime


class Message(BaseModel):
	sender: str
	content: str

class Question(BaseModel):
	question: str
	messages: List[Message] = []

class Context(BaseModel):
	context: str