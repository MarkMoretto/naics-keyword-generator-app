

# import numpy as np
# if int("".join(np.__version__.split(".")[:2])) < 117:
#     raise ImportWarning("Please update numpy to version 1.17 or newer before running the script.")
import re
from functools import partial
# from datetime import datetime as dt


from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from gensim import corpora
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
# from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

# For plotting the word vectors
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

try:
    from src import DATA_DIR
    from src._preprocess import create_and_clean_df
    from src._types import List, StrList
except ImportError:
    from src import DATA_DIR
    from ._preprocess import create_and_clean_df
    from ._types import List, StrList    

# date_stamp = dt.utcnow().strftime("%Y-%m-%d")
# nlp.utils.check_version("0.10.0")
# assert (nlp.__version__ == "0.10.0"), "GluonNLP Version Error"


# context = mx.cpu()  # Enable this to run on CPU
# context = mx.gpu(0)  # Enable this to run on GPU

model_filename = "word2vec-latest.model"
W2v_MODEL_PATH = DATA_DIR.joinpath(model_filename)
W2v_MODEL_PATH_ABS = str(W2v_MODEL_PATH.absolute())

# vec_filename = f"word2vec-latest.wordvectors"
# vec_path = DATA_DIR.joinpath(vec_filename)


SNOWBALL_STEMMER_ENG = SnowballStemmer("english")
SNOWBALL_STEMMER_ENG_IGNORE_STOPWORDS = SnowballStemmer("english", ignore_stopwords=True)


def stem_words(words, stem_fn = SNOWBALL_STEMMER_ENG_IGNORE_STOPWORDS) -> StrList:
    """Returns list of tokens processed with specified word stemmer function."""
    return list(map(stem_fn.stem, words))


def get_data(min_word_length: int = 2) -> List[StrList]:
    _, data = create_and_clean_df(remove_stopwords=False)
    data = data.loc[~data.loc[:, "description"].str.startswith("see industry description"), :]
    df1 = data.apply(lambda a: " ".join(a), axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

    df1 = df1.to_frame("texts")
    _documents = (df1.loc[:, "texts"].apply(word_tokenize)).apply(lambda q: [word for word in q if len(word) >= min_word_length])
    return _documents

# documents = get_data(min_word_length = 3)


# --- mean words per line/sentance --- #
avg_words_per_line: float = lambda docs: sum([len(line) for line in docs]) / len(docs)


# --- Partial function for known parameters --- #
p_Word2Vec: Word2Vec = partial(
    Word2Vec,
    min_count = 1,
    vector_size = 300,
    workers = 4,
    epochs = 50,
    sg = 0,
    hs = 0,
    alpha = 0.020,
    min_alpha = 1e-3,
)


def get_or_create_w2v_model(create_new: bool = False) -> Word2Vec:
    """Return pre-trained model or create new one if no prior model found."""
    if not W2v_MODEL_PATH.exists() or create_new:
        _docs = get_data(min_word_length = 3)
        # Get average words per line and update window
        awpl = avg_words_per_line(_docs)
        _w2v_mod = p_Word2Vec(
            sentences = _docs.values,
            window = int(awpl ** 0.5),
        )

        _w2v_mod.save(W2v_MODEL_PATH_ABS)
    return Word2Vec.load(W2v_MODEL_PATH_ABS)


def make_tokens(text: str, fn = lambda w: re.split(r"`", re.sub(r"([\W\s]+)", "`", w), flags = re.I)) -> StrList:
    return fn(text)

# tst = "health|care   doctor"
# make_tokens(tst)
# w2v_model = get_or_create_w2v_model(create_new = True)


# Similarity
# w2v_model.wv.most_similar("health", topn=10)
# w2v_model.wv.most_similar("care", topn=10)
# w2v_model.wv.most_similar("doctor", topn=10)
# w2v_model.wv.most_similar_cosmul(positive=["health", "care", "doctor"], topn=10)

# try:
#     w2v_model.wv.most_similar("healthcare", topn=10)
# except KeyError as ke:
#     print("Word not found in data model!")

# Potential to "seek" parts of a word if error raised.
# [i for i in dir(w2v_model) if not i.startswith("_")]
# type(w2v_model.wv.vocab)

# w2v_model = Word2Vec(sentences=documents.values, min_count=1, vector_size=300, workers = 4)
# print(w2v_model)

# words = list(w2v_model.wv.index_to_key)

# w2v_model.save(str(model_path.absolute()))
# w2v_model = Word2Vec.load(str(model_path.absolute()))

# # --- Plot with PCA
# X = w2v_model.wv[w2v_model.wv.index_to_key]
# pca = PCA(n_components=2)
# pca_mod = pca.fit_transform(X)

# plt.scatter(pca_mod[:, 0], pca_mod[:, 1])
# plt.show()


# # Similarity
# w2v_model.wv.most_similar("health", topn=10)
# w2v_model.wv.most_similar("care", topn=10)
# w2v_model.wv.most_similar("doctor", topn=10)



# vec_filename = f"word2vec-{date_stamp}.wordvectors"
# vec_path = str(DATA_DIR.joinpath(vec_filename).absolute())

# vecs = w2v_model.wv
# vecs.save(vec_path)

# wv = KeyedVectors.load(vec_path, mmap="r")
# vector_ = wv["health"]

# # from gensim.test.utils import common_corpus, common_dictionary
# from gensim.models.hdpmodel import HdpModel


# def get_dict_corpus(iterable: list):
#     _dict = corpora.Dictionary(documents)
#     _corp = [_dict.doc2bow(r) for r in documents]
#     return _dict, _corp

# def query_to_bow_vector(keyword_string: str):
#     """Return query string into bag of words vector."""
#     return DICTIONARY.doc2bow(re.split(r"[^\w-]+", keyword_string.lower()))

# DICTIONARY, CORPUS = get_dict_corpus(documents.values)


# hdp = HdpModel(CORPUS, DICTIONARY)

# bow_vec = query_to_bow_vector("healthcare")
# doc_hdp = hdp[bow_vec]


# topic_info = hdp.print_topics(num_topics=20, num_words=10)

