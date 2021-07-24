
__all__ = [
    "create_app"
    ]

import sys
import logging
from typing import Optional
from datetime import datetime as dt

# Third-party imports
import databases
from loguru import logger
from fastapi import APIRouter, FastAPI
from starlette.requests import Request
from starlette.config import environ

from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from fastapi.responses import ORJSONResponse
from starlette.config import Config
from starlette.datastructures import CommaSeparatedStrings, Secret
from starlette.middleware.cors import CORSMiddleware


# Local imports
# from .char_counter.char_counter import count_chars
# from .config import prod_settings, dev_settings
# from .logging.handlers import InterceptHandler
# from .logging.utils import get_log_level


# API  init
api = APIRouter(prefix="")


# Current settings
settings = dev_settings


# .env config
config = Config(".env")
DEBUG: bool = config("DEBUG", cast=bool, default=False)
SECRET_KEY: str  = config("SECRET_KEY", cast=Secret)
ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=CommaSeparatedStrings)


# Docker log level
if DEBUG:
    LOGGING_LEVEL = get_log_level("debug")
else:
    LOGGING_LEVEL = get_log_level(environ["LOG_LEVEL"].lower())
# ENV_LOG_LEVEL = get_log_level(ENV_LOG_LEVEL.lower())


# Configure logging.
# LOGGING_LEVEL = get_log_level(logging.DEBUG if DEBUG else ENV_LOG_LEVEL)
LOGGERS = ("uvicorn.asgi", "uvicorn.access")

logging.getLogger().handlers = [InterceptHandler()]
for logger_name in LOGGERS:
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = [InterceptHandler(level=LOGGING_LEVEL)]

log_handlers=[{
    "sink": sys.stderr,
    "level": LOGGING_LEVEL,
    }]

logger.configure(handlers=log_handlers)




# Instance app
# app = FastAPI(title = settings.APP_NAME, debug = settings.DEBUG)


# Errors
async def http_error_handler(request: Request, exc: HTTPException) -> ORJSONResponse:
    return ORJSONResponse({"errors": [exc.detail]}, status_code=exc.status_code)


# Creat app function.
def create_app():
    _app = FastAPI(title = settings.APP_NAME, debug = settings.DEBUG)
    _app.include_router(api)
    _app.add_exception_handler(HTTPException, http_error_handler)

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],                
        )
    return _app


@api.get("/", response_class = ORJSONResponse)
async def home() -> ORJSONResponse:
    msg = {
        "message": "Backend is running!",
        "current_dttm": dt.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return ORJSONResponse(status_code=200, content = msg)



@api.get("/count/{phrase}", response_class = ORJSONResponse)
async def counter(phrase: Optional[str] = None) -> ORJSONResponse:
    if phrase is None:
        raise HTTPException(status_code=404, detail = "String parameter not detected.")
    else:
        # --- Phase 4 --- #
        msg = count_chars(phrase)   
        if LOGGING_LEVEL <= logging.INFO:
            logger.opt(colors = True).debug(f"<white>{phrase}</white> <white>=></white> <magenta>{msg}</magenta>")
    return ORJSONResponse(status_code=200, content = {"result": msg})


@api.get("/logs/")
async def counter(phrase: Optional[str] = None) -> ORJSONResponse:
    if phrase is None:
        raise HTTPException(status_code=404, detail = "String parameter not detected.")
    else:
        # --- Phase 4 --- #
        msg = count_chars(phrase)   
        if LOGGING_LEVEL <= logging.INFO:
            logger.opt(colors = True).debug(f"<white>{phrase}</white> <white>=></white> <magenta>{msg}</magenta>")
    return ORJSONResponse(status_code=200, content = {"result": msg})


# Instance app
app = create_app()
