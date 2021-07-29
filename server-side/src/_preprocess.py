__all__ = [
    "df_for_nlp",
    "create_and_clean_df",
    "create_word_corpus",
    "corpus_effects",
    "DataFrame",
]


import re
import logging
from io import BytesIO

import requests
import numpy as np
from pandas import DataFrame, factorize, read_excel, set_option as pd_set_option

from src._types import (
    Dict,
    List,
    Number,
    Vector,
    MutVector,
    Iterables,
    StrDict,
    Tuple,
)

from src import DATA_DIR, Path


DEBUG: bool = True
NAICS_YEAR: int = 2017
LOCAL_NAICS_DESC_XLSX_PATH: Path = DATA_DIR.joinpath(f"naics-desc-{NAICS_YEAR}.xlsx")
LOCAL_CSV_DATA_PATH: Path = DATA_DIR.joinpath("naics-main.csv")
STOPWORD_PATH = DATA_DIR.joinpath("stopwords.txt")

# --- Pandas options --- #
pd_set_option("io.excel.xlsx.reader", "openpyxl")
pd_set_option("mode.chained_assignment", None)

if DEBUG:
    pd_set_option("display.max_colwidth", 80)
    pd_set_option("display.max_columns", 25)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



urlmap: Dict = dict(
    code_index = f"https://www.census.gov/naics/{NAICS_YEAR}NAICS/{NAICS_YEAR}_NAICS_Index_File.xlsx",
    code_description = f"https://www.census.gov/naics/{NAICS_YEAR}NAICS/{NAICS_YEAR}_NAICS_Descriptions.xlsx"
)


def _set_stopwords(output_path: Path) -> None:
    _url = "https://gist.github.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
    with requests.Session() as s:
        resp = s.get(_url)
        sw_raw = resp.text
        sw_out = [i for i in sw_raw.split("\n") if len(i) > 1]
        with output_path.open(mode="w") as outf:
            outf.write("\n".join(sw_out))
            


def get_stopwords(min_stopword_len: int = 1) -> List:
    """Return list of stopwords.
    
    Parameters
    ----------
    min_stopword_len : int
        Minimum character count for a word to remain included in the final list.
    
    Returns
    -------
    List
        List of NLTK English stop words.
    """
    _sw_path = DATA_DIR.joinpath("stopwords.txt")
    _sw_string: str = ""
    if not _sw_path.exists():
        _set_stopwords(_sw_path)

    # Read and return.
    with _sw_path.open() as f:
        _sw_string = f.read()
        return [i for i in re.split(r"\n", _sw_string) if len(i) > min_stopword_len]


# --- Retrieve data from census.gov --- #
def retrieve_data(url: str = urlmap["code_description"]) -> DataFrame:
    """Return BytesIO object representing MS Excel workbook."""
    _resp = requests.get(url)
    _bio = BytesIO(_resp.content)
    if _bio:
        _bio.seek(0)
        _df = read_excel(_bio, engine="openpyxl")
        _df.columns = list(map(str.lower, _df.columns))
        return _df


def clean_data(DF: DataFrame, exclude_stopwords = True) -> DataFrame:
    """Return pandas DaraFrame following "clean-up" of text data."""

    cols = ["title", "description"]

    for c in cols:
        DF.loc[:, c] = DF.loc[:, c].str.lower()

        # Some titles end with 'T'; Remove that.
        if c == "title":
            DF.loc[:, c] = DF.loc[:, c].str.replace(r"t\s*?$", "", regex=True).str.strip()
            DF.loc[DF[c].isna(), c] = ""

        # Replace newline and non alphanumeric characters with double whitespace.
        DF.loc[:, c] = DF.loc[:, c].str.replace(r"(\r?\n+)", "  ", regex=True)
        DF.loc[:, c] = DF.loc[:, c].str.replace(r"[^a-zA-Z- ]", "  ", regex=True)

        if c == "description":
            DF.loc[:, c] = DF.loc[:, c].replace(r"\s{2,}", " ", regex=True)

            # Remove stopwords
            if exclude_stopwords:
                _sw = get_stopwords()
                _sw_dict = {rf"\b{w}\b": "" for w in _sw if len(w) > 1}            
                DF.loc[:, c] = DF.loc[:, c].replace(_sw_dict, regex=True)   
                        
            DF.loc[:, c] = DF.loc[:, c].str.strip()
            DF.loc[DF[c].isna(), c] = ""

    return DF


def create_and_clean_df(url: str = urlmap["code_description"], remove_stopwords = True) -> DataFrame:

    # Get data from data source
    _df = retrieve_data(url)

    # Set dependent column data set (if we were going to predict class)
    df_dependent = _df.loc[:, "code"]

    # Set independent (predictor) values dataframe.
    df_independent = clean_data(_df.drop("code", axis=1), remove_stopwords)

    return df_dependent, df_independent


def create_word_corpus(DF: DataFrame, col_name: str = "description", min_word_length: int = 1) -> DataFrame:
    """Return pandas DaraFrame representing corpus of tokenized words."""

    _corpus = (
        DF.loc[:, col_name].str.split(r"\s+")           # Split on white space
        .apply(lambda r: [w for w in r if len(w) > min_word_length])  # take workds with character counts over a given minimum.
        .explode()
        .to_frame("token")
        ).sort_values(by=["token"])


    # Clean-up words with hyphens at the start or items that are only hyphens.
    _corpus.loc[_corpus["token"].str.startswith("-", na=False), "token"] = (
        _corpus.loc[_corpus["token"].str.startswith("-", na=False), "token"].str.replace(r"[-]+", "", regex=True)
    )

    # Remove row(s) with blank value and return results.
    return _corpus.loc[_corpus["token"] != "", :].copy()


def corpus_effects(corp: DataFrame) -> Tuple[StrDict, StrDict, int]:
    # Word count for entire corpus
    _word_count = corp.groupby("token").size()
    _word_count = _word_count.reset_index().rename(columns={0:"freq"})

    # Value encoding.
    codes, uniques = factorize(_word_count["token"], sort=True)


    _word_to_index = dict(zip(uniques, codes))
    _index_to_word = dict(zip(codes, uniques))
    _vocab_size = len(_word_to_index)
    return _word_count, _word_to_index, _vocab_size


def df_for_nlp(min_frequency: int = 1):
    """Data is """
    # --- Set some variables -- #
    df_dep, _df = create_and_clean_df()

    _df = _df.loc[:, "description"].str.split(r"\s+").apply(lambda r: [w for w in r if len(w) > 1])

    dfx = _df.explode().to_frame("token").reset_index(drop=True)

    # dfq = (dft.explode().to_frame("token")).sort_values(by=["token"])


    # Clean-up words with hyphens at the start or items that are only hyphens.
    dfx.loc[dfx["token"].str.startswith("-", na=False), "token"] = (
        dfx.loc[dfx["token"].str.startswith("-", na=False), "token"].str.replace(r"-+", "", regex=True)
    )
    dfx = dfx.loc[dfx["token"] != "", :]

    word_count = (dfx.groupby("token").size()).reset_index().rename(columns={0: "freq"})

    # Remove words with frequencies of min_frequency or below.
    freq_vals = (word_count.loc[word_count["freq"] > min_frequency, "token"]).values
    freq_map = {f"\b{w}\b":"" for w in freq_vals}
    _df = _df.replace(freq_map, regex=True)
    _df = _df.apply(lambda c: [i for i in c if not i.startswith("-")])

    return _df

