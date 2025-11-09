from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ToolRecord(BaseModel):
    id: str             # unique id, e.g. sha or uuid
    name: Optional[str]
    language: str
    code: str
    created_at: datetime
    metadata: Optional[dict] = {}

class ErrorRecord(BaseModel):
    id: str
    error_text: str
    stderr: Optional[str]
    context: Optional[str]
    created_at: datetime

class DocRecord(BaseModel):
    id: str
    title: str
    content: str
    created_at: datetime

class PatternRecord(BaseModel):
    id: str
    pattern: str
    description: Optional[str]
    created_at: datetime
