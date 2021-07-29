

from functools import partial

# local
from src._types import List, Optional, Union


from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
    )

# from sklearn.discriminant_analysis import (
#     LinearDiscriminantAnalysis as LinDA,
#     QuadraticDiscriminantAnalysis as QDA,
#     )

from sklearn.decomposition import (
    LatentDirichletAllocation as LDA,
    NMF, # Non-Negative Matrix Factorization
    )

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


from src._preprocess import create_and_clean_df

DecompModel = Union[LDA, NMF]

N_TOPICS: int = 10

df_target, df = create_and_clean_df(remove_stopwords = False)
documents = df["description"].values

def get_top_n_words(model: DecompModel, feature_names: List, n_top_words: Optional[int] = 10) -> None:
    """Print ranking of significant words for a given number of topics."""

    output = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features_idx = topic.argsort()[:-n_top_words-1:-1]
        top_features = [feature_names[idx] for idx in top_features_idx]
        weights = topic[top_features_idx]

        output[f"topic_{topic_idx + 1}"] = "\n\t".join([f"{tf}  -  {w:.3f}" for w, tf in zip(weights, top_features)])

    for k, v in output.items():
        print(f"\n{k} top features:\n\t{v}")



tfidf_vec = TfidfVectorizer(
    max_df = 0.95,
    min_df = 2, # Ignore frequency below this threshold
    stop_words="english",
    max_features= len(documents) // 2, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
)
tfidf_mod = tfidf_vec.fit_transform(documents)

# Extract LDA features
cv_vec = CountVectorizer(
    max_df = 0.95,
    min_df = 2, # Ignore frequency below this threshold
    stop_words="english",
    max_features= len(documents) // 4, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
)
cv_mod = cv_vec.fit_transform(documents)

 

# Core feature names to use when viewing/outputting results.
FEATURE_NAMES = tfidf_vec.get_feature_names()


# Lock some params in place.
base_NMF = partial(
    NMF,
    n_components=N_TOPICS,
    max_iter=500,    
    alpha = 0.1,
    l1_ratio = 0.5,    
    )

# NMF model (Frobenius norm) fitted to Tf-Idf features
nmf_vec_fn = base_NMF(
    init="nndsvda",
    beta_loss="frobenius",
    solver="cd",
)
nmf_mod_fn = nmf_vec_fn.fit(tfidf_mod)


# NMF model (generalized Kullback-Leibler divergence) fitted to Tf-Idf features
nmf_vec_kl = base_NMF(
    init="nndsvda",
    beta_loss="kullback-leibler",
    solver="mu",
)
nmf_mod_kl = nmf_vec_kl.fit(tfidf_mod)


# Latent Dirichlet Allocation model with Tf-Idf features
lda_vec = LDA(
    n_components=N_TOPICS,
    max_iter = 5,
    learning_method="online",
    learning_offset=50., # A (positive) parameter that downweights early iterations in online learning. 
)
lda_mod = lda_vec.fit(cv_mod)



nmf_fn_clf = Pipeline([
    ("tfidf", tfidf_vec),
    ("nmf", nmf_vec_fn),
    ])


nmf_kl_clf = Pipeline([
    ("tfidf", tfidf_vec),
    ("nmf", nmf_vec_kl),
    ])


lda_clf = Pipeline([
    ("tf", cv_vec),
    ("lda", lda_vec),
    ])  




# Print results
# get_top_n_words(nmf_mod_kl, FEATURE_NAMES, 10)
# get_top_n_words(lda_mod, FEATURE_NAMES, 10)
# import re
# from gensim.test.utils import common_texts
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# vals = df["description"].values

# p = re.compile(r"([^a-z0-9- ]+)", flags = re.I)

# res = [w for w in re.split(r"\s+", p.sub("  ", vals[0])) if len(w) > 2]

# tag_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
# doc_model = Doc2Vec(tag_docs, vector_size=5, window=2, min_count=1, workers=4)

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer
# from math import log10, log
# corpus_ = [
#     "this is the first document",
#     "this document is the second document",
#     "and this is the third one",
#     "is this the first document",
#     ]

# # Document count
# n_docs = len(corpus_)


# TARGET = "first"

# line_metrics = dict(
#     target_count=[],
#     word_count=[],
#     cum_tot_cnt=[0],
#     )

# for line in corpus_:
#     line_metrics["target_count"].append(line.count(TARGET))
#     line_metrics["word_count"].append(len(line.split(" ")))

#     if line_metrics["cum_tot_cnt"][-1] == 0:
#         line_metrics["cum_tot_cnt"][0] = line_metrics["word_count"][-1]
#     else:
#         tmp = line_metrics["cum_tot_cnt"][-1] + line_metrics["word_count"][-1]
#         line_metrics["cum_tot_cnt"].append(tmp)

# n_docs_w_target = sum([1 for i in line_metrics["target_count"] if i > 0])
# tf = sum([i for i in line_metrics["target_count"]])
# idf = log(n_docs / n_docs_w_target)
# tf_df = tf*idf

# idf_smooth = log10((n_docs+1) / (n_docs_w_target+1)) + 1
# tf_idf_smooth = tf*idf_smooth