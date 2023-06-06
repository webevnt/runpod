from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.v1 import whisper_route, views
from app.config import settings

app = FastAPI(title=settings.API_NAME, version=settings.VERSION)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

import mimetypes
mimetypes.init()

templates = Jinja2Templates(directory="templates")

mimetypes.add_type('application/javascript', '.js')
app.mount("/static", StaticFiles(directory="static"), name="static")



app.include_router(whisper_route.router, prefix="/api", tags=["whispers"])
app.include_router(views.router)
