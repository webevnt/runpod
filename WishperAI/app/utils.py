import random
import uuid
from datetime import datetime, timedelta
from typing import Union

from fastapi import Response, status
from jose import JWTError, jwt
from app.config import settings
from app.logs.log_handler import LibLogger
from passlib.context import CryptContext

log: LibLogger = LibLogger()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def exception_decorator(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            response: Response = Response()
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            log.error(error)
            return False, error

    return inner

def generate_uuid() -> str:
    return str(uuid.uuid4())

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_decode_token(token):
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    return payload

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7,minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode,settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

