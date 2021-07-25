# server-side\src\datadata.py

__all__ = [
    "cosine_similarity",
    "get_similar_words",
]

import re
from io import BytesIO
import concurrent.futures as ccf
from typing import Dict, Iterable, List, Optional, Tuple, Union

import requests
import numpy as np
from pandas import DataFrame, factorize, read_excel, set_option as pd_set_option

from src import DATA_DIR, Path


# --- Pandas options --- #
pd_set_option("display.max_colwidth", 80)
pd_set_option("display.max_columns", 25)
pd_set_option("io.excel.xlsx.reader", "openpyxl")
pd_set_option("mode.chained_assignment", None)


# --- Types --- #
Number = Union[int, float]
Vector = List[Number]
Iterables = Union[Iterable, Iterable]
StrStrDict = Dict[str, str]




naics_year: int = 2017

urlmap: Dict = dict(
    code_index = f"https://www.census.gov/naics/{naics_year}NAICS/{naics_year}_NAICS_Index_File.xlsx",
    code_description = f"https://www.census.gov/naics/{naics_year}NAICS/{naics_year}_NAICS_Descriptions.xlsx"
)


def _set_stopwords(output_path: Path) -> None:
    _url = "https://gist.github.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
    with requests.Session() as s:
        resp = s.get(_url)
        stopwords_raw = resp.text
        with output_path.open(mode="w") as outf:
            outf.write(stopwords_raw)


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
def retrieve_data(url: str) -> DataFrame:
    """Return BytesIO object representing MS Excel workbook."""
    _resp = requests.get(url)
    _bio = BytesIO(_resp.content)
    if _bio:
        _bio.seek(0)
        return read_excel(_bio)



def clean_data(DF: DataFrame, remove_stopwords = True) -> DataFrame:
    """Return pandas DaraFrame following "clean-up" of text data."""

    cols = ["Title", "Description"]

    if remove_stopwords:
        _sw = get_stopwords()
        _sw_dict = {rf"\b{w}\b": "" for w in _sw if len(w) > 1}

    for c in cols:
        DF.loc[:, c] = DF.loc[:, c].str.lower()

        # Some titles end with 'T'; Remove that.
        if c == "Title":
            DF.loc[:, c] = DF.loc[:, c].str.replace(r"t\s*?$", "", regex=True).str.strip()
            DF.loc[DF[c].isna(), c] = ""

        # Replace newline and non alphanumeric characters with double whitespace.
        DF.loc[:, c] = DF.loc[:, c].str.replace(r"(\r?\n+)", "  ", regex=True)
        DF.loc[:, c] = DF.loc[:, c].str.replace(r"[^a-zA-Z- ]", "  ", regex=True)

        if c == "Description":
            DF.loc[:, c] = DF.loc[:, c].replace(r"\s{2,}", " ", regex=True)
            DF.loc[:, c] = DF.loc[:, c].replace(_sw_dict, regex=True)            
            DF.loc[:, c] = DF.loc[:, c].str.strip()
            DF.loc[DF[c].isna(), c] = ""

    return DF


def create_and_clean_df(url: str = urlmap["code_description"]) -> DataFrame:

    # Get data from data source
    _df = retrieve_data(url)

    # Set dependent column data set (if we were going to predict class)
    # df_dependent = _df.loc[:, "Code"]

    # Set independent (predictor) values dataframe.
    df_independent = clean_data(_df.drop("Code", axis=1))

    return df_independent


def create_word_corpus(DF: DataFrame, col_name: str = "Description") -> DataFrame:
    """Return pandas DaraFrame representing corpus of tokenized words."""

    _corpus = (
        DF.loc[:, col_name].str.split(r"\s+")
        .apply(lambda r: [w for w in r if len(w) > 1])
        .explode()
        .to_frame("token")
        ).sort_values(by=["token"])


    # Clean-up words with hyphens at the start or items that are only hyphens.
    _corpus.loc[_corpus["token"].str.startswith("-", na=False), "token"] = (
        _corpus.loc[_corpus["token"].str.startswith("-", na=False), "token"].str.replace(r"-+", "", regex=True)
    )

    # Remove row(s) with blank value and return results.
    return _corpus.loc[_corpus["token"] != "", :].copy()


def corpus_effects(corp: DataFrame) -> Tuple[StrStrDict, StrStrDict, int]:
    # Word count for entire corpus
    word_count = corp.groupby("token").size()
    word_count = word_count.reset_index().rename(columns={0:"freq"})

    # Value encoding.
    codes, uniques = factorize(word_count["token"], sort=True)


    _word_to_index = dict(zip(uniques, codes))
    _index_to_word = dict(zip(codes, uniques))
    _vocab_size = len(_word_to_index)
    return _word_to_index, _index_to_word, _vocab_size


# --- Set some variables -- #
df = create_and_clean_df()
corpus = create_word_corpus(df)
word_to_index, index_to_word, vocab_size = corpus_effects(corpus)




# --- Activators --- #
# arr = np.array([1,3,7,2,8,0,12,2,2,20], dtype = np.int32)

def softmax(X: np.array) -> np.ndarray:
    numer = np.exp(X - np.max(X))
    return numer / numer.sum(axis = 0)


def relu(X: np.array) -> np.ndarray:
    return X * (X > 0)


def softplus(X) -> np.ndarray:
    return np.logaddexp(1.0, X)


# --- Processing --- #
def label_vectorizer(target_word: str, context_words: Iterable) -> Tuple[Vector, Vector]:
    """Return two lists: one for binary representation of target word within unique values collection
    and the other representing context words in relation to the target word within a unique values collection.
    """
    _target_vec: np.array = np.zeros(vocab_size, dtype = np.int32)
    _context_vec: np.array = np.zeros_like(_target_vec)

    word_index: int = word_to_index.get(target_word)
    if word_index:
        _target_vec[word_index] = 1
    
    for w in _context_vec:
        w_index = word_to_index.get(w)
        _context_vec[w_index] = 1

    return _target_vec, _context_vec


def create_training_data(window_size: int = 2, create_sample: bool = False) -> Iterables:

    training_data: List = []
    training_sample_words: List =  []

    corp_arr = corpus["token"].values
    length_of_corpus: int = len(corp_arr)

    for i, word in enumerate(corp_arr):
        index_target_word = i
        target_word = word
        context_words = []

        # when target word is the first word
        if i == 0:  
            # trgt_word_index:(0), ctxt_word_index:(1,2)
            context_words = [corp_arr[x] for x in range(window_size + 1)] 


        # when target word is the last word
        elif i == length_of_corpus - 1:
            # trgt_word_index:(9), ctxt_word_index:(8,7), length_of_corpus = 10
            context_words = [corp_arr[x] for x in range(length_of_corpus - 2, length_of_corpus - 2 - window_size, -1)]

        # When target word is the middle word
        else:
            # Before the middle target word
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size, -1):
                if x >= 0:
                    context_words.extend([corp_arr[x]])

            # After the middle target word
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < length_of_corpus:
                    context_words.extend([corp_arr[x]])


        trgt_word_vec, ctxt_word_vec = label_vectorizer(target_word, context_words)

        training_data.append([trgt_word_vec, ctxt_word_vec])   
        
        # Create and return sample data set.
        if create_sample:
            training_sample_words.append([target_word, context_words])   
        
    return training_data, training_sample_words    



# --- Propogation --- #

def forward_propogation(hidden_wt_in: np.array, hidden_wt_out: np.array, target_vector: np.array) -> Tuple[np.ndarray, np.array, np.array]:
    # Input layer weights
    hidden_layer = np.dot(hidden_wt_in.T, target_vector)

    # Output layer weights
    out_hidden_layer = np.dot(hidden_wt_out.T, hidden_layer)

    # Predicted dependent values
    y_pred = softmax(out_hidden_layer)

    return y_pred, hidden_layer, out_hidden_layer


def backward_prop(hidden_wt_in, hidden_wt_out, total_error, hidden_layer, target_word_vector, learning_rate: float = 0.01):
    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(hidden_wt_in, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)
    
    # Update weights
    weight_inp_hidden = hidden_wt_in - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = hidden_wt_out - (learning_rate * dl_weight_hidden_output)
    
    return weight_inp_hidden, weight_hidden_output


def calculate_error(y_pred, context_words) -> np.array:
    
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}
    
    for i in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update ({i : 'yes'})
        
    number_of_1_in_context_vector = len(index_of_1_in_context_words)
    
    for i, value in enumerate(y_pred):
        
        if index_of_1_in_context_words.get(i) != None:
            total_error[i]= (value-1) + ( (number_of_1_in_context_vector -1) * value)
        else:
            total_error[i]= (number_of_1_in_context_vector * value)
            
            
    return np.array(total_error)




def calculate_loss(u, ctx):
    sum_1 = 0

    for i in np.where(ctx == 1)[0]:
        sum_1 = sum_1 + u[i]
    
    sum_1 = -sum_1
    sum_2 = len(np.where(ctx == 1)[0]) * np.log(np.sum(np.exp(u)))
    
    total_loss = sum_1 + sum_2
    return total_loss



# --- Model training --- #
def train(word_embedding_dimension, window_size, epochs, training_data, learning_rate, disp = "no", interval = -1):
    
    weights_input_hidden = np.random.uniform(-1, 1, (vocab_size, word_embedding_dimension))
    weights_hidden_output = np.random.uniform(-1, 1, (word_embedding_dimension, vocab_size))


    # For analysis purposes
    epoch_loss = []
    weights_1 = []
    weights_2 = []
    
    for epoch in range(epochs):
        loss = 0

        for target,context in training_data:
            y_pred, hidden_layer, u = forward_prop(weights_input_hidden,weights_hidden_output,target)

            total_error = calculate_error(y_pred, context)

            weights_input_hidden,weights_hidden_output = backward_prop(
                weights_input_hidden,weights_hidden_output ,total_error, hidden_layer, target,learning_rate
            )

            loss_temp = calculate_loss(u,context)
            loss += loss_temp
        
        epoch_loss.append( loss )
        weights_1.append(weights_input_hidden)
        weights_2.append(weights_hidden_output)
        
        if disp == 'yes':
            if epoch ==0 or epoch % interval ==0 or epoch == epochs -1:
                print('Epoch: %s. Loss:%s' %(epoch,loss))
    return epoch_loss,np.array(weights_1),np.array(weights_2)


# [corpus.loc[x, "token"] for x in range(i + 1, window_size + 1)]
# [corpus.loc[x, "token"] for x in range(1, 3)]

window_size = 2
epochs = 100
learning_rate = 0.01

training_data, training_sample_words = create_training_data(2, True)


# --- Evaluation --- #
def cosine_similarity(word, weight):
    
    #Get the index of the word from the dictionary
    index = word_to_index[word]
    
    #Get the correspondin weights for the word
    word_vector_1 = weight[index]
    
    
    word_similarity = {}

    for i in range(vocab_size):
        
        word_vector_2 = weight[i]
        
        theta_sum = np.dot(word_vector_1, word_vector_2)
        theta_den = np.linalg.norm(word_vector_1) * np.linalg.norm(word_vector_2)
        theta = theta_sum / theta_den
        
        word = index_to_word[i]
        word_similarity[word] = theta
    
    return word_similarity #words_sorted


def similarity_score():




def get_similar_words(top_n_words: int, weight, msg, words_subset) -> DataFrame:
    
    columns = []
    
    for i in range(0, len(words_subset)):
        columns.append("similar: " + str(i + 1))
        
    _df = DataFrame(columns=columns,index=words_subset)
    # _df.head()
    
    row = 0
    for word in words_subset:
        
        # Get the similarity matrix for the word: word
        similarity_matrix = cosine_similarity(word,weight,word_to_index,vocab_size,index_to_word)
        col = 0
        
        # Sort the top_n_words
        words_sorted = dict(sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)[1:top_n_words+1])
        
        # Create a dataframe to display the similarity matrix
        for similar_word,similarity_value in words_sorted.items():
            _df.iloc[row][col] = (similar_word, round(similarity_value, 2))
            col += 1
        row += 1

    styles = [dict(selector='caption',

    props=[('text-align', 'center'),('font-size', '20px'),('color', 'red')])] 

    _df = _df.style.set_properties(**{'color': 'green','border-color': 'blue','font-size':'14px'}).set_table_styles(styles).set_caption(msg)

    return _df




loss_epoch = {}
dataframe_sim = []

epoch_loss,weights_1,weights_2 = train(dimension,window_size,epochs,training_data,learning_rate,'yes',50)
loss_epoch.update( {'yes': epoch_loss} )





# https://towardsdatascience.com/a-word2vec-implementation-using-numpy-and-python-d256cf0e5f28

# --- Retrieve both assets simultaneously --- #
# with ccf.ThreadPoolExecutor(max_workers = len(urlmap) * 2) as e:
#     url_futures = {e.submit(retrieve_data, url):url_name for url_name, url in urlmap.items()}
#     for fut in ccf.as_completed(url_futures):
#         if url_futures[fut] == "code_index":
#             df_code = data_to_df(fut.result())
#         else:
#             _df = data_to_df(fut.result())
#             df = clean_data(_df)







