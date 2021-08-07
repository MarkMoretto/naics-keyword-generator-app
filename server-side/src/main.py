
__all__ = [
    "create_app"
    ]

import sys
import logging
from random import choices, sample as rand_sample
from typing import Optional
from datetime import datetime as dt

# Third-party imports
from fastapi import APIRouter, FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.exceptions import RequestValidationError

from pydantic import BaseModel

from starlette.config import Config
from starlette.config import environ
from starlette.requests import Request
from starlette.exceptions import HTTPException
from starlette.datastructures import CommaSeparatedStrings, Secret
from starlette.middleware.cors import CORSMiddleware


# Local imports
from .config import prod_settings, dev_settings
from ._preprocess import get_stopwords
from ._embeddings import (
    get_or_create_w2v_model,
    make_tokens,
    W2v_MODEL_PATH,
    W2v_MODEL_PATH_ABS,
    Word2Vec,
    )


# API  init
api = APIRouter(prefix="")


# Current settings
settings = dev_settings


# .env config
config = Config(".env")
DEBUG: bool = config("DEBUG", cast=bool, default=False)
SECRET_KEY: str  = config("SECRET_KEY", cast=Secret)
ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=CommaSeparatedStrings)



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


# --- STOPWORDS --- #
STOPWORDS = get_stopwords()

@api.get("/stopwords", response_class = ORJSONResponse)
async def counter(sample_size: Optional[int] = -1) -> ORJSONResponse:
    """
    # Retrieve list of common English stopwords.
    &nbsp;
    ## Parameters
    \----------------
    - **sample_size** {_int_} - Number of random stopwords to retrieve.
    """
    _sw = stopwords
    if sample_size > 0:
        # _sw = choices(STOPWORDS, k = sample_size)
        _sw = rand_sample(STOPWORDS, k = sample_size)
    msg = {
        "stopwords": _sw
    }
    return ORJSONResponse(status_code=200, content = msg)
    

# --- SIMILARITY --- #

w2v_model = get_or_create_w2v_model()

class SimilarityItem(BaseModel):
    text_input: str
    num_results: int = 10
    negative_terms: str = ""


@api.post("/similarity", response_class = ORJSONResponse)
async def counter(item: SimilarityItem) -> ORJSONResponse:
    """Return top N words similar to each of a given set of keywords.
    Handles both positive and negative terms.
    """
    if not item.text_input is None:
        # Return object and type.
        res: KeyedVecList = None
        _positive_terms: StrList = []        
        _negative_terms: StrList = []

        _positive_terms = make_tokens(item.text_input.strip())

        if len(item.negative_terms) > 0:
            _negative_terms = make_tokens(item.negative_terms.strip())
            res = w2v_model.wv.most_similar_cosmul(
                positive = _positive_terms,
                negative = _negative_terms,
                topn = item.num_results,
                )

        else:
            res = w2v_model.wv.most_similar(positive = _positive_terms, topn = item.num_results)

        return ORJSONResponse(status_code=200, content = dict(res))
    raise HTTPException(status_code = 404, detail = "String parameter not detected.")    


@api.post("/similarity-ext", response_class = ORJSONResponse)
async def counter(item: SimilarityItem) -> ORJSONResponse:
    """Return top N words similar to each of a given set of keywords.

    Handles both positive and negative terms.

    Extended -> Returns two sets of results.
    """
    if not item.text_input is None:
        # Return object and type.
        res: KeyedVecList = None
        res_cosmul: KeyedVecList = None
        _positive_terms: StrList = []        
        _negative_terms: StrList = []

        _positive_terms = make_tokens(item.text_input.strip())

        if len(item.negative_terms) > 0:
            _negative_terms = make_tokens(item.negative_terms.strip())

        res = w2v_model.wv.most_similar(positive = _positive_terms, negative = _negative_terms, topn = item.num_results)
        res_cosmul = w2v_model.wv.most_similar_cosmul(positive = _positive_terms, negative = _negative_terms, topn = item.num_results,)

        return ORJSONResponse(status_code=200, content = [dict(most_similar = dict(res)), dict(most_similar_cosmul = dict(res_cosmul))])

    raise HTTPException(status_code = 404, detail = "String parameter not detected.")    


# @api.get("/logs/")
# async def counter(phrase: Optional[str] = None) -> ORJSONResponse:
#     if phrase is None:
#         raise HTTPException(status_code=404, detail = "String parameter not detected.")
#     else:
#         # --- Phase 4 --- #
#         msg = count_chars(phrase)   
#         if LOGGING_LEVEL <= logging.INFO:
#             logger.opt(colors = True).debug(f"<white>{phrase}</white> <white>=></white> <magenta>{msg}</magenta>")
#     return ORJSONResponse(status_code=200, content = {"result": msg})


# Instance app
app = create_app()
