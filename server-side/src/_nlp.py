# server-side\src\_nlp.py

# --- Gensim action --- #
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem.porter import PorterStemmer
import re
from gensim import corpora, similarities
from gensim.models import LsiModel, LdaModel

# from ._preprocess import df_for_nlp
from src._preprocess import df_for_nlp


df = df_for_nlp(min_frequency=1)

VERBOSE: bool = False
# Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()
# df_stem = df.loc[:15, "token"].apply(lambda r: [p_stemmer.stem(i) for i in r])

documents = df.values

dictionary = corpora.Dictionary(df.values)
corpus = [dictionary.doc2bow(r) for r in documents]


TEST_QUERY = "health care"
LSI_FILEPATH: Path = DATA_DIR.joinpath("lsi-model.index")
LDA_FILEPATH: Path = DATA_DIR.joinpath("lda-model.index")

vec_bow = dictionary.doc2bow(re.split(r"[^\w-]+", TEST_QUERY.lower()))

# Latent Semantic indexing
# https://radimrehurek.com/gensim/models/lsimodel.html
# https://github.com/susanli2016/NLP-with-Python
lsi = LsiModel(corpus, id2word=dictionary, num_topics=10)
vec_lsi = lsi[vec_bow]

if VERBOSE:
    lsi.print_topics(10)
    print(vec_lsi)


lsi_index = similarities.MatrixSimilarity(lsi[corpus])

 # Save model
lsi_index.save(str(LSI_FILEPATH.absolute()))

# lsi_index = similarities.MatrixSimilarity.load(str(LSI_FILE.absolute()))

# --- Cosine Similarity using LSI Vector from Test Query --- #
# Range is (-1, 1), with -1 being the least similar and 1 being the most similar #
lsi_similarities = lsi_index[vec_lsi]

# Number of results to retireve.
N_RESULTS: int = 10
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
lda = LdaModel(corpus, num_topics = 4, id2word = dictionary, alpha="auto", passes=5)
vec_lda = lda[vec_bow]
print(vec_lda)

print(lda.print_topics(num_topics=2, num_words=3))

print(lda.print_topics(num_topics=3, num_words=3))

lda_index = similarities.MatrixSimilarity(lda[corpus])

 # Save model
lda_index.save(str(LDA_FILEPATH.absolute()))

### Print similarities
lda_similarities = lda_index[vec_lda]
sorted_lda_similarities = sim_sort(lda_similarities)
print_sims(sorted_lda_similarities)