

import re
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

from functools import partial
from collections import defaultdict
from typing import Iterator, List, Union


# --- Third-party packages --- #
import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Token
# import pandas as pd

# --- Local imports --- #

from src import DATA_DIR, Path
from src._preprocess import create_and_clean_df

# Types
from src._types import Number, StrList, Union
TokenObj = Union[Union[Doc, Token], Span]

# Loading spaCy library.
nlp = spacy.load("en_core_web_sm")



# _, df = create_and_clean_df(remove_stopwords = False)



matcher = Matcher(nlp.vocab)

# https://spacy.io/usage/rule-based-matching
# https://course.spacy.io/en/chapter2
pattern = [
    {"TEXT": "iPhone"},
    {"TEXT": "X"}
    ]
matcher.add("IPHONE_PATTERN", [pattern])

text = "Upcoming iPhone X release date leaked"
doc = nlp(text)
matches = matcher(doc)
print(doc.text)
matcher = Matcher(nlp.vocab)


#######################
# --- Hash Values --- #
text = "David Bowie is a PERSON"
doc = nlp(text)

TEST: str = "PERSON"
# nlp.vocab.strings[TEST]
# doc.vocab.strings[TEST]
hash_test = nlp.vocab.strings[TEST]
nlp.vocab.strings[hash_test]

lexeme = nlp.vocab[TEST]
print(f"\nMetrics for the test word: {lexeme.text}")
print(f"\n\tHash - {lexeme.orth}\n\tIs Alpha? - {lexeme.is_alpha}\n")


################################
# --- Doc and Span objects --- #
words = ["Hello", "world", "!"]

# The spaces are a list of boolean values indicating
# whether the word is followed by a space.
spaces = [True, False, False] 

doc = Doc(nlp.vocab, words=words, spaces=spaces)

span = Span(doc, 0, 2)
labeled_span = Span(doc, 0, 2, label="GREETING")

# Add span to the doc.ents
doc.ents = [labeled_span]


# Analyze text and collect proper nouns
doc = nlp("Berlin looks like a nice city")

# Get all tokens and part-of-speech tags
token_texts = [token.text for token in doc]
pos_tags = [token.pos_ for token in doc]

# Bad because it uses strings vs tokens
for index, pos in enumerate(pos_tags):
    # Check if the current token is a proper noun
    if pos == "PROPN":
        # Check if the next token is a verb
        if pos_tags[index + 1] == "VERB":
            result = token_texts[index]
            print("Found proper noun before a verb:", result)


# Using tokens instead
for token in doc:
    if token.pos_ == "PROPN":
        if doc[token.i + 1].pos_ == "VERB":
            print(f"Found proper noun before a verb: {token.text}")



#####################
# --- Semantics --- #

import numpy as np
# Using larger model with word vectors
nlp = spacy.load("en_core_web_md")

doc1 = nlp("I like fast food.")
doc2 = nlp("I like pizza.")


def calc_similarity(doc_1: TokenObj, doc_2: TokenObj) -> float:
    d1_vec, d2_vec = doc_1.vector, doc_2.vector

    numer = np.dot(d1_vec, d2_vec)
    denom = np.linalg.norm(d1_vec) * np.linalg.norm(d2_vec)

    return numer / denom

similarity = calc_similarity(doc1, doc2)

print(f"spaCy similarity: {doc1.similarity(doc2):.5f}")
print(f"Calculated similarity: {similarity:.5f}")


# Compare tokens
doc = nlp("I like pizza and pasta")
token1: Token = doc[2]
token2: Token = doc[4]
print(token1.similarity(token2))

################################################
#  ---Phrase matching with PhraseMatcher() --- #
ph_matcher = PhraseMatcher(nlp.vocab)

pattern = nlp("Golden Retriever")
ph_matcher.add("DOG", None, pattern)
doc = nlp("I have a Golden Retriever")

for match_id, start, end in ph_matcher(doc):
    span = doc[start:end]
    print(f"Matched span: {span}")


pattern = [{"LOWER": "silicon"}, {"TEXT": " "}, {"LOWER": "valley"}]
doc = nlp("Can Silicon Valley workers rein in big tech from within?")


#  --- Debugging --- #

# * Edit pattern1 so that it correctly matches all case-insensitive mentions 
# of "Amazon" plus a title-cased proper noun.
# * Edit pattern2 so that it correctly matches all case-insensitive mentions
# of "ad-free", plus the following noun.

doc = nlp(
    "Twitch Prime, the perks program for Amazon Prime members offering free "
    "loot, games and other benefits, is ditching one of its best features: "
    "ad-free viewing. According to an email sent out to Amazon Prime members "
    "today, ad-free viewing will no longer be included as a part of Twitch "
    "Prime for new members, beginning on September 14. However, members with "
    "existing annual subscriptions will be able to continue to enjoy ad-free "
    "viewing until their subscription comes up for renewal. Those with "
    "monthly subscriptions will have access to ad-free viewing until October 15."
)
# Create the match patterns
pattern1 = [
    {"LOWER": "amazon"},
    {"IS_TITLE": True, "POS": "PROPN"}
    ]
# pattern2 = [{"LOWER": "ad-free"}, {"POS": "NOUN"}]
# doc[28].is_punct
pattern2 = [{"LOWER": "ad"}, {"IS_PUNCT": True}, {"LOWER": "free"}, {"POS": "NOUN"}]

# Initialize the Matcher and add the patterns
matcher = Matcher(nlp.vocab)
matcher.add("PATTERN1", [pattern1])
matcher.add("PATTERN2", [pattern2])

# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

# for token in doc:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Iterate over the matches
for match_id, start, end in matcher(doc):
    # Print pattern string name and text of matched span
    print(doc.vocab.strings[match_id], doc[start:end].text)


# --- Efficient phrase matching --- #



