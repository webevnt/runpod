from typing import Dict

import requests
from app.config import settings
from app.logs.log_handler import LibLogger
from app.status_messages import response_status
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.status import HTTP_403_FORBIDDEN,HTTP_401_UNAUTHORIZED
from app.utils import get_decode_token

log: LibLogger = LibLogger()


class AuthHandler:
    security: HTTPBearer = HTTPBearer()

    def decode_token(self, token: str):
        """
        This method is used for decoding jwt token to get payload data.
        """
        message: str = response_status.INVALID_TOKEN
        try:
            headers: Dict = {
                "Authorization": f"Bearer {token}",
            }
            data = get_decode_token(token)
            if data:
                return data
            else:
                raise HTTPException(
                    status_code=401, detail="Not authenticated"
                )
        except HTTPException as e:
            raise e
        except Exception as error:
            log.error(error)
            raise HTTPException(status_code=500, detail=message)

    def auth_wrapper(self, auth: HTTPAuthorizationCredentials = Security(security)):
        """
        This is method is used by all the API's as autorization wrapper for API security
        """
        return self.decode_token(auth.credentials)
