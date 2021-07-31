

import re
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

from functools import partial
from typing import Iterator, List, Union
import concurrent.futures as ccf

import pandas as pd

from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS, Phrases, Phraser

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.malletcorpus import MalletCorpus



from nltk import pos_tag # https://www.nltk.org/book/ch05.html
from nltk.stem import WordNetLemmatizer as Lemmatizer


from src import DATA_DIR, Path
from src._preprocess import create_and_clean_df

Numbers = Union[int, float]
StrList = List[str]

_, df = create_and_clean_df(remove_stopwords = False)
# df = df["description"]

df = df.loc[~df.loc[:, "description"].str.startswith("see industry description"), :]
df1 = df.apply(lambda a: " ".join(a), axis=1).str.replace(r"\s+", " ", regex=True).str.strip()

# documents = df["description"].values
df1 = df1.to_frame("texts")
# sw_map = dict.fromkeys(map(lambda w: rf"\b{w}\b", STOPWORDS), "")
# df1 = df1.replace(sw_map, "", regex=True)

def gen_tokens(iterable: List[str]) -> Iterator[str]:
    for line in iterable:
        yield simple_preprocess(f"{line}")

texts = list(gen_tokens(df1.values))


# Collocation detection
# Bigrams and Trigrams
# Higher thresholds == fewer phrases
bigrams = Phrases(texts, min_count=3, threshold=10, connector_words=ENGLISH_CONNECTOR_WORDS)
trigrams = Phrases(bigrams[texts], threshold=10)

bigram_mod = Phraser(bigrams)
trigram_mod = Phraser(trigrams)

# Sample
print(trigram_mod[bigram_mod[texts[0]]])


def remove_stopwords(iterable: List[str], fn = lambda q: not q in STOPWORDS) -> List[StrList]:
    return [list(filter(fn, line)) for line in iterable]

# texts1 = remove_stopwords(texts)
def create_bigrams(iterable: List[str]) -> List[str]:
    return [bigram_mod[text] for text in iterable]

def create_trigrams(iterable: List[str]) -> List[str]:
    return [trigram_mod[bigram_mod[text]] for text in iterable]


patterns = [
    (r".*ing$", "VBG"),                # gerunds
    (r".*ed$", "VBD"),                 # simple past
    (r".*es$", "VBZ"),                 # 3rd singular present
    (r".*ould$", "MD"),                # modals
    (r".*\'s$", "NN$"),                # possessive nouns
    (r".*s$", "NNS"),                  # plural nouns
    (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),  # cardinal numbers
    (r".*", "NN")                      # nouns (default)
]

def lemma_by_tags(iterable: List[str], allowed_postags: List[str] = ["NN", "NNS", "NN$", "ADJ", "JJ", "VERB", "ADV", "RB"]) -> List[str]:
    output = []

    for text in iterable:
        output.append([w[0] for w in pos_tag(text) if w[1] in allowed_postags])
    return output

texts1 = remove_stopwords(texts)
bi_texts = create_bigrams(texts1)
# tri_texts = create_trigrams(texts1)
lemma_texts = lemma_by_tags(bi_texts)



# dict and corpus
dictionary = corpora.Dictionary(lemma_texts)
corpus = [dictionary.doc2bow(text) for text in lemma_texts]
text_corpus = [[(dictionary[i], freq) for i, freq in item] for item in corpus]


# Build LDA Model
lda_mod = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    update_every=1,
    chunksize=len(corpus) // 4, 
    passes = 10, 
    alpha = "auto",
    per_word_topics=True,
    )
# lda_mod.print_topics()
lda_vec = lda_mod[corpus]


# Get model perplexity
lda_mod_perplexity = lda_mod.log_perplexity(corpus)

# Get model coherence score
# from gensim.models.coherencemodel import CoherenceModel
lda_mod_coherence = CoherenceModel(model = lda_mod, texts = lemma_texts, dictionary=dictionary, coherence="c_v")
coherence_lda = lda_mod_coherence.get_coherence()

msg = []
msg.append(f"Perplexity: {lda_mod_perplexity:.4f}")
msg.append(f"Coherence Score: {coherence_lda:.4f}")

print("\n".join(msg))


# Paritals
p_LdaModel = partial(LdaModel, corpus = corpus, id2word = dictionary, eta="auto", eval_every=5, distributed = False, num_topics = 100, decay=0.5, iterations=50)
p_CoherenceModel = partial(CoherenceModel, texts=lemma_texts, dictionary=dictionary, coherence="c_v", topn = 20, processes = 1)



def simple_run(num_topics: int):
    lda_model = p_LdaModel(num_topics = num_topics)

    coherence_model = p_CoherenceModel(model = lda_model)

    return lda_model, coherence_model.get_coherence()


def compute_coherence_values(limit, start = 2, step = 3):
    _coherence_list = []
    _model_list = []

    with ccf.ThreadPoolExecutor(max_workers=4) as out_executor:
        mod_futures = {out_executor.submit(simple_run, n):f"{n}" for n in range(start, limit, step)}

        for k in ccf.as_completed(mod_futures):
            try:
                current = mod_futures[k]
                _lda, _coherence = k.result()
                _model_list.append(_lda)
                _coherence_list.append(_coherence)
                print(f"Completed with item: {current}")
            except ccf.BrokenExecutor as be:
                print(f"Generated an exception: {be}")
            except ccf.thread.BrokenThreadPool as bte:
                print(f"Generated BrokenThreadPool exception: {bte}")

    return _model_list, _coherence_list

LIMIT = 60
START = 2
STEP = 4

model_list, coherence_values = compute_coherence_values(limit = LIMIT, step = STEP)


for m, cv in zip(range(START, LIMIT, STEP), coherence_values):
    print(f"Num Topics = {m} has Coherence Value of {cv:.4f}")

pre_flatten_max = 0.2959
optimal_model = model_list[[i for i, v in enumerate(coherence_values) if round(v, 4) == pre_flatten_max][0]]
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=10))
print(optimal_model.print_topics(num_topics=10))


# Domnant topic in each sentence
def format_topics_sentences(model = lda_mod, corpus = corpus, texts = texts1):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break

    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


df_topic_sents_keywords = format_topics_sentences(model = optimal_model)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ["document_no", "dominant_topic", "topic_perc_contrib", "keywords", "text",]

# Show
df_dominant_topic.head(10)








### --- Topic distribution across documents --- ###

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords["Dominant_Topic"].value_counts()

topic_counts_df = topic_counts.to_frame("topic_count")

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ["dominant_topic", "topic_keywords", "num_documents", "pct_documents"]

# Show
df_dominant_topics.loc[~df_dominant_topics["num_documents"].isna(), :]
