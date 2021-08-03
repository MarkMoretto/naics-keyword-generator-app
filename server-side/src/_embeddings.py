

# import numpy as np
# if int("".join(np.__version__.split(".")[:2])) < 117:
#     raise ImportWarning("Please update numpy to version 1.17 or newer before running the script.")
import re
from datetime import datetime as dt

from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models.word2vec import Word2Vec

# For plotting the word vectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src import DATA_DIR
from src._preprocess import create_and_clean_df

date_stamp = dt.utcnow().strftime("%Y-%m-%d")
# nlp.utils.check_version("0.10.0")
# assert (nlp.__version__ == "0.10.0"), "GluonNLP Version Error"


# context = mx.cpu()  # Enable this to run on CPU
# context = mx.gpu(0)  # Enable this to run on GPU

model_filename = f"word2vec-{date_stamp}.model"
vec_filename = f"word2vec-{date_stamp}.wordvectors"
model_path = DATA_DIR.joinpath(model_filename)
vec_path = DATA_DIR.joinpath(vec_filename)

_, data = create_and_clean_df(remove_stopwords=False)


data = data.loc[~data.loc[:, "description"].str.startswith("see industry description"), :]
df1 = data.apply(lambda a: " ".join(a), axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

df1 = df1.to_frame("texts")
documents = df1.loc[:, "texts"].apply(word_tokenize)


w2v_model = Word2Vec(sentences=documents.values, min_count=1, vector_size=300, workers = 4)
print(w2v_model)

words = list(w2v_model.wv.index_to_key)

w2v_model.save(str(model_path.absolute()))
w2v_model = Word2Vec.load(str(model_path.absolute()))

# --- Plot with PCA
X = w2v_model.wv[w2v_model.wv.index_to_key]
pca = PCA(n_components=2)
pca_mod = pca.fit_transform(X)

plt.scatter(pca_mod[:, 0], pca_mod[:, 1])
plt.show()


# Similarity
w2v_model.wv.most_similar("health", topn=10)
w2v_model.wv.most_similar("care", topn=10)
w2v_model.wv.most_similar("doctor", topn=10)


from gensim.models import KeyedVectors
vec_filename = f"word2vec-{date_stamp}.wordvectors"
vec_path = str(DATA_DIR.joinpath(vec_filename).absolute())

vecs = w2v_model.wv
vecs.save(vec_path)

wv = KeyedVectors.load(vec_path, mmap="r")
vector_ = wv["health"]

# from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.hdpmodel import HdpModel


def get_dict_corpus(iterable: list):
    _dict = corpora.Dictionary(documents)
    _corp = [_dict.doc2bow(r) for r in documents]
    return _dict, _corp

def query_to_bow_vector(keyword_string: str):
    """Return query string into bag of words vector."""
    return DICTIONARY.doc2bow(re.split(r"[^\w-]+", keyword_string.lower()))

DICTIONARY, CORPUS = get_dict_corpus(documents.values)


hdp = HdpModel(CORPUS, DICTIONARY)

bow_vec = query_to_bow_vector("healthcare")
doc_hdp = hdp[bow_vec]


topic_info = hdp.print_topics(num_topics=20, num_words=10)

