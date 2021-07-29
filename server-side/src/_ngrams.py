
# https://nbviewer.jupyter.org/github/sgsinclair/alta/blob/master/ipynb/RepeatingPhrases.ipynb

# import nltk
# from src._preprocess import df_for_nlp, create_and_clean_df, create_word_corpus
# from src import DATA_DIR, Path

# _, df1 = create_and_clean_df(remove_stopwords = False)
# df2 = df1.drop("Title", axis = 1, inplace = False)
# df3 = create_word_corpus(df2)
# df = df3.reset_index().groupby("index")["token"].apply(list)


# # df = df_for_nlp()


# # Sample data
# dfx = df[:15]

# # Bigrams
# # df_bigrams = [list(nltk.ngrams(r, n = 2)) for r in dfx]


# # 4-grams
# df_4grams = [list(nltk.ngrams(r, n = 4)) for r in dfx]
# freq_4grams = [nltk.FreqDist(r) for r in df_4grams]

# # 4-grams with fuller list
# all_tokens = [x for y in dfx for x in y] # Flatten list of lists.
# all_4grams = list(nltk.ngrams(all_tokens, n = 4))
# all_freq_4grams = nltk.FreqDist(all_4grams)

# for words, freq in all_freq_4grams.most_common(10):
#     tmp = ", ".join(words)
#     print(f"{freq}: {tmp}")
