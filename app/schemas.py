from pydantic import BaseModel, EmailStr, UUID4, Field, field_validator
from typing import Union, Optional, List
from datetime import datetime


class Question(BaseModel):
	question: str