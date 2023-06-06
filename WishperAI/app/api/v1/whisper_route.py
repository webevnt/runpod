import json
import pathlib
from typing import Any, Dict, List, Union

import requests
from app.api.v1 import crud, models, schemas
from app.api.v1.auth_handler import AuthHandler
from app.api.v1.models import *
from app.config import settings
from app.logs.log_handler import LibLogger
from app.status_messages import response_status
from fastapi import APIRouter, Depends, File, Response, UploadFile, status, Request, Form
import aiofiles
import numpy as np
import librosa
from streaming_form_data import StreamingFormDataParser
from streaming_form_data.targets import FileTarget, ValueTarget
from streaming_form_data.validators import MaxSizeValidator
import streaming_form_data
from starlette.requests import ClientDisconnect 
from typing import Annotated, Optional


JSONObject: Dict = Dict[str, Any]
JSONArray: List = List[Any]
JSONStructure: Union[List, Dict] = Union[JSONArray, JSONObject]

router: APIRouter = APIRouter()
log: LibLogger = LibLogger()
auth_handler: AuthHandler = AuthHandler()

MAX_FILE_SIZE = 1024 * 1024 * 1024 * 1  # = 1GB
MAX_REQUEST_BODY_SIZE = MAX_FILE_SIZE + 1024


class MaxBodySizeException(Exception):
    def __init__(self, body_len: str):
        self.body_len = body_len

class MaxBodySizeValidator:
    def __init__(self, max_size: int):
        self.body_len = 0
        self.max_size = max_size

    def __call__(self, chunk: bytes):
        self.body_len += len(chunk)
        if self.body_len > self.max_size:
            raise MaxBodySizeException(body_len=self.body_len)


async def whisper_transcribe_audio(form_data):
    try:
        task = form_data.get('task')
        file = form_data.get('file')
        return_timestamps = form_data.get('return_timestamps')
        from app.ml_services.whisper_ai import transcribe
        outputs = ""
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            contents = await file.read()
            outputs = transcribe(contents)
        return True, outputs
    except Exception as error:
        return False, str(error)


async def whisperx_transcribe_and_diarize_audio(form_data):
    try:
        CHUNK_SIZE = 1024 * 1024
        file = form_data.get('file')
        FILE_PATH = f'model_data/{file.filename}'
        return_timestamps = form_data.get('return_timestamps') 
        from app.ml_services.whisper_ai import transcribe_and_diarize
        outputs = ""
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            contents = await file.read()
            outputs = transcribe_and_diarize(contents,FILE_PATH)
        # async with aiofiles.open(FILE_PATH, 'wb') as out_file:
        #     content = await file.read()  # async read
        #     await out_file.write(content)  # async write
        return True, outputs
    except Exception as error:
        print(str(error),"---===----=------==========---------========")
        return False, str(error)
   

@router.post("/whisper/transcribe",
    summary="Use Whisper to transcribe audio",
    description="Use Whisper to `transcribe` or `translate` task."
)
async def use_whisper(
    response: Response,
    form_data: schemas.WhisperJaxSchema = Depends(schemas.WhisperJaxSchema.as_form),
):
    """
    Endpoint are for use of creating new user.
    """
    try:
        form_data: Dict = form_data.dict()
        is_processed, data = await whisper_transcribe_audio(form_data)
        if is_processed:
            response.status_code = status.HTTP_200_OK
            return data
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": data}  
    except Exception as error:
        log.error(error)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"message": str(error)}



@router.post("/whisperx/diarize",
    summary="Use Whisperx to transcribe and diarize",
    description="Use Whisperx to `transcribe` and `diarize`."
)
async def use_speaker_diarization(
    response: Response,
    form_data: schemas.WhisperJaxSchema = Depends(schemas.WhisperJaxSchema.as_form),
):
    """
    Endpoint are for use of creating new user.
    """
    try:
        form_data: Dict = form_data.dict()
        is_processed, data = await whisperx_transcribe_and_diarize_audio(form_data)
        if is_processed:
            response.status_code = status.HTTP_200_OK
            return data
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"message": data}
    except Exception as error:
        log.error(error)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"message": str(error)}


