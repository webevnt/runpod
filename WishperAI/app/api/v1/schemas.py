from datetime import datetime
from typing import Optional, List, Annotated

import pydantic
from fastapi import Form
from pydantic import BaseModel, Json, validator
from fastapi import FastAPI,APIRouter, Depends, File, Form, UploadFile, Query


class WhisperJaxSchema(BaseModel):
    file: Optional[UploadFile]
    return_timestamps: Optional[bool]
    
    @validator('return_timestamps')
    def set_return_timestamps(cls, timestamp):
        return timestamp or False
    
    @classmethod
    def as_form(
        cls,
        file: UploadFile = File(None),
        return_timestamps: bool = Form(None)
    ):
        return cls(file=file, return_timestamps=return_timestamps)
    


class WhisperFormSchema(BaseModel):
    file: Optional[Annotated[bytes, File()]]
    return_timestamps: Optional[bool]
    
    @validator('return_timestamps')
    def set_return_timestamps(cls, timestamp):
        return timestamp or False
    
    @classmethod
    def as_form(
        cls,
        file: Annotated[bytes, File()],
        return_timestamps: Annotated[bool, Form()]
    ):
        return cls(file=file, return_timestamps=return_timestamps)