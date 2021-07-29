

from typing import Optional, List

from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from fastapi.responses import ORJSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

from gensim.similarities import MatrixSimilarity

from src import DATA_DIR, Path

from src.main import df_nlp


token_router = APIRouter(prefix="/keywords")


def load_pretrained_model(url: Path):
    """Load a pre-trained model from local storage."""
    return MatrixSimilarity.load(str(url.absolute()))


ldam_path = DATA_DIR.joinpath("ldam-model.index")

ldam_model = load_pretrained_model(ldam_path)

# Retrieve data
# documents = df_for_nlp(min_frequency=1)


# @token_router.post("/related/", response_class = ORJSONResponse)
# async def get_top_keywords(request: Request, query_string: str, top_n: Optional[int] = 10) -> ORJSONResponse:
#     ..


@token_router.get("/stopwords", response_class = ORJSONResponse)
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
        # _sw = choices(stopwords, k = sample_size)
        _sw = rand_sample(stopwords, k = sample_size)
    msg = {
        "stopwords": _sw
    }
    return ORJSONResponse(status_code=200, content = msg)