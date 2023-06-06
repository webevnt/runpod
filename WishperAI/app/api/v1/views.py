from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from app.logs.log_handler import LibLogger
from app.api.v1.auth_handler import AuthHandler
from fastapi import APIRouter, Depends, File, Response, UploadFile, status, Request
from app.api.v1 import crud, models, schemas
from app.api.v1.models import *




router: APIRouter = APIRouter()
log: LibLogger = LibLogger()
auth_handler: AuthHandler = AuthHandler()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
	return templates.TemplateResponse("index.html", {"request": request})


@router.get("/auth", response_class=HTMLResponse)
async def auth(request: Request):
	return templates.TemplateResponse("auth.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def show_dashboard(request: Request):
	return templates.TemplateResponse("dashboard.html", {"request": request})

