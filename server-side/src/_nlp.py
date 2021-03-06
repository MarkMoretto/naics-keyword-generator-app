#!/bin/python3


# __all__ = ["load_model"]

import re

# --- Gensim action --- #

from gensim import corpora, similarities
from gensim.similarities import MatrixSimilarity
from gensim.models import LsiModel, LdaModel, LdaMulticore, TfidfModel

from src._preprocess import df_for_nlp
from src import DATA_DIR, Path


df = df_for_nlp(min_frequency=1)

# df = df.apply(lambda c: [i for i in c if not i.startswith("-")])

VERBOSE: bool = False

# Number of results to retireve.
N_RESULTS: int = 10

# Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()
# df_stem = df.loc[:15, "token"].apply(lambda r: [p_stemmer.stem(i) for i in r])

documents = df.values

def get_dict_corpus(iterable: list):
    _dict = corpora.DICTIONARY(documents)
    _corp = [_dict.doc2bow(r) for r in documents]
    return _d, _c


DICTIONARY, CORPUS = get_dict_corpus(documents)


def load_pretrained_model(url: Path):
    """Load a pre-trained model from local storage."""
    return similarities.MatrixSimilarity.load(str(url.absolute()))

def query_to_bow_vector(keyword_string: str):
    return DICTIONARY.doc2bow(re.split(r"[^\w-]+", keyword_string.lower()))

TEST_QUERY = "healthcare providers"
LSI_FILEPATH: Path = DATA_DIR.joinpath("lsi-model.index")
LDA_FILEPATH: Path = DATA_DIR.joinpath("lda-model.index")
LDAM_FILEPATH: Path = DATA_DIR.joinpath("ldam-model.index")
TFIDF_LDAM_FILEPATH: Path = DATA_DIR.joinpath("tfidf-ldam-model.index")

vec_bow = DICTIONARY.doc2bow(re.split(r"[^\w-]+", TEST_QUERY.lower()))

# Latent Semantic indexing
# https://radimrehurek.com/gensim/models/lsimodel.html
# https://github.com/susanli2016/NLP-with-Python
lsi = LsiModel(CORPUS, id2word=DICTIONARY, num_topics=10)
vec_lsi = lsi[vec_bow]

if VERBOSE:
    lsi.print_topics(10)
    print(vec_lsi)


lsi_index = similarities.MatrixSimilarity(lsi[CORPUS])

 # Save model
lsi_index.save(str(LSI_FILEPATH.absolute()))

# lsi_index = similarities.MatrixSimilarity.load(str(LSI_FILE.absolute()))

# --- Cosine Similarity using LSI Vector from Test Query --- #
# Range is (-1, 1), with -1 being the least similar and 1 being the most similar #
lsi_similarities = lsi_index[vec_lsi]


sim_sort = lambda sims: sorted(enumerate(sims), key = lambda Q: -Q[1])
sorted_similarities = sim_sort(lsi_similarities)

def print_sims(sims, N_RESULTS: int = 10):
    i = 0
    for doc_position, doc_score in sims:
        if i == N_RESULTS:
            break
        tmp = ",".join(df.values[doc_position])
        print(f"{doc_score:.6f}\t{tmp}\n")
        i += 1

print_sims(sorted_similarities)

# https://towardsdatascience.com/latent-semantic-analysis-deduce-the-hidden-topic-from-the-document-f360e8c0614b
# generate LDA model
lda = LdaModel(CORPUS, num_topics = 4, id2word = DICTIONARY, alpha="auto", passes=5)
vec_lda = lda[vec_bow]
print(vec_lda)

print(lda.print_topics(num_topics=2, num_words=3))

print(lda.print_topics(num_topics=3, num_words=3))

lda_index = similarities.MatrixSimilarity(lda[CORPUS])

 # Save model
lda_index.save(str(LDA_FILEPATH.absolute()))

### Print similarities
lda_similarities = lda_index[vec_lda]
sorted_lda_similarities = sim_sort(lda_similarities)
print_sims(sorted_lda_similarities)



#######################
### LDA multicore
ldam = LdaMulticore(CORPUS, num_topics = 10, id2word = DICTIONARY, passes=2, workers = 2)
vec_ldam = ldam[vec_bow]
print(vec_ldam)

for topic, words in ldam.print_topics(-1):
    print(f"Topic #: {topic}\nWords: {words}")

# print(ldam.print_topics(num_topics=2, num_words=3))
# print(ldam.print_topics(num_topics=3, num_words=3))

ldam_index = similarities.MatrixSimilarity(ldam[CORPUS])

 # Save model
ldam_index.save(str(LDAM_FILEPATH.absolute()))

### Print similarities
ldam_similarities = ldam_index[vec_ldam]
sorted_ldam_similarities = sim_sort(ldam_similarities)
print_sims(sorted_ldam_similarities)


#### Tf-Idf
tfidf = TfidfModel(CORPUS)
corpus_tfidf = tfidf[CORPUS]
ldam_tfidf = LdaMulticore(corpus_tfidf, num_topics=10, id2word=DICTIONARY, passes=2, workers=4)
for topic, words in ldam_tfidf.print_topics(-1):
    print(f"Topic #: {topic}\nWords: {words}")


test_idx: int = 100
sample_docs = documents[test_idx]

for index, score in sorted(ldam[CORPUS[test_idx]], key=lambda X: -1 * X[1]):
    print(f"\nScore: {score}\t \nTopic: {ldam.print_topic(index, 10)}")


for index, score in sorted(ldam_tfidf[CORPUS[test_idx]], key=lambda X: -1 * X[1]):
    print(f"\nScore: {score}\t \nTopic: {ldam_tfidf.print_topic(index, 10)}")