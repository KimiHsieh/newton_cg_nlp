import os
import time
import numpy as np

# import matplotlib as mpl
# import matplotlib.pyplot as plt
from pprint import pprint
from IPython.display import clear_output
import unicodedata
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


print(tf.__version__)


def data_load(dataset_name):
    """This function downloads the TED talk transcripts dataset
    Please check dataset in https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate

    Arguments:
        dataset_name (string): name of dataset

    Returns:
        train_examples and val_examples (tuple): raw training and validation data 
    """
    examples, metadata = tfds.load(dataset_name, with_info=True, as_supervised=True)
    train_examples, val_examples = examples["train"], examples["validation"]
    clear_output()
    train_examples_sizes = [i for i, _ in enumerate(train_examples)][-1] + 1
    val_examples_sizes = [i for i, _ in enumerate(val_examples)][-1] + 1
    print(f"train_examples_sizes = {train_examples_sizes}")
    print(f"val_examples_sizes = {val_examples_sizes}")

    # origin data looks
    print(f"Show three raw training data ......")
    for src, tar in train_examples.take(3):
        print(src)
        print(tar)
        print("-" * 10)

    # make data looks better
    print(f"Decode the training data ......")
    for src_t, tar_t in train_examples.take(3):
        src = src_t.numpy().decode("utf-8")
        tar = tar_t.numpy().decode("utf-8")
        print(src)
        print(tar)
        print("-" * 10)

    return train_examples, val_examples


def data_preprocessing(examples):
    """
    Args:
        examples: raw data

    Returns:
        seq_src, seq_tar {tuple}: preprocessed source and targe setences
    """
    seq_ = []
    seq_src = []
    seq_tar = []

    for src_t, tar_t in examples:
        src = src_t.numpy().decode("utf-8")
        tar = tar_t.numpy().decode("utf-8")
        src = preprocess_sentence(src)
        tar = preprocess_sentence(tar)
        seq_.append((src, tar))
        seq_src.append(src)
        seq_tar.append(tar)

    # check
    print(f"Show three sentences after being preprocessed ......")
    for i in range(3):
        src = seq_src[i]
        tar = seq_tar[i]
        print(src)
        print(tar)
        print("-" * 10)

    return seq_src, seq_tar


def preprocess_sentence(s):
    """Preprocess the raw data before using tokenizer
    Recipe:
    1. converting all letters to lower or upper case
    2. removing punctuations(here, we keep them), accent marks and other diacritics
    3. standardization, but we don't use it. We keep the origin form
    4. Add BOS and EOS

    Args:
        s (string): a decoded raw sentence 

    Returns:
        s (string): a preprocessed sentence
    """
    s = s.lower().strip()

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    s = re.sub(r"[^a-zA-Zçáéíóúâêôãõàèìòù?.!,¿]+", " ", s)

    s = s.strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    s = "BOS " + s + " EOS"
    return s


def load_word_embedding(pretrained_word_embedding):
    """Load the pre-trained Word Embedding model from fastText.
    https://fasttext.cc/docs/en/crawl-vectors.html
    
    Args:
        pretrained_word_embedding (.vec): pre-trained Word Embedding from fastText

    Returns:
        embeddings_dict (dict): key: word (string), val: features (double)
    """
    embeddings_dict = dict()
    f = open(pretrained_word_embedding)
    lines_de = f.readlines()

    for line in lines_de[1:]:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype="float32")
        embeddings_dict[word] = vec

    f.close()
    return embeddings_dict


def build_word_embedding_matrix(embeddings_dict, tokenizer, size_of_vocabulary):
    """Create a weight matrix for words as the embedding layer in the model.
    Since pre-trained model can't cover all words in tokenizer, the function will show how many words are lost.

    Args:
        embeddings_dict (dict)
        tokenizer (keras tokenizer)
        size_of_vocabulary (interger)

    Returns:
        embedding_matrix (2d np array): row: vocabularies in tokenizer; column: features (double)
    """
    embedding_matrix = np.zeros((size_of_vocabulary, 300))
    lost = []
    j = 0
    for word, i in tokenizer.word_index.items():
        if j == size_of_vocabulary - 1:
            break

        if i == 2:
            embedding_vector = embeddings_dict.get("BOS")
            embedding_matrix[i] = embedding_vector
        elif i == 3:
            embedding_vector = embeddings_dict.get("EOS")
            embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                lost.append((word, i))
        j += 1

    num_lost = len(lost)
    print(f"num_lost = {len(lost)}")
    print(f"subword loss ratio = {num_lost/len(embedding_matrix)}")
    return embedding_matrix


def text_to_indices(tokenizer_src, tokenizer_tar, seq_src, seq_tar):
    """ Use the tokenizer to map text sequence to interger indices.
    Args:
        tokenizer_src (keras tokenizer)
        tokenizer_tar (keras tokenizer)
        seq_src (string)
        seq_tar (string)

    Returns:
        idices (tuple): idices_src(list(int)), idices_tar(list(int))
    """
    idices_src = tokenizer_src.texts_to_sequences(seq_src)
    idices_tar = tokenizer_tar.texts_to_sequences(seq_tar)
    print("-" * 25, "sequence to idices", "-" * 25)
    print(seq_src[5])
    print(idices_src[5])
    print(seq_tar[5])
    print(idices_tar[5])
    return idices_src, idices_tar


def filter_and_padding(idices_src, idices_tar, max_length):
    """ Make all indices sequences with the same length. 

    Args:
        idices_src (list(int))
        idices_tar (list(int))
        max_length (int): 

    Returns:
        idices_src_max(list(int)), idices_tar_max(list(int)): idices sequences with same length
    """
    tmp_idices_src = []
    tmp_idices_tar = []

    # filter out the sentences(idices) by max_length
    for idices_src_, idices_tar_ in zip(idices_src, idices_tar):
        if len(idices_src_) <= max_length and len(idices_tar_) <= max_length:
            tmp_idices_src.append(idices_src_)
            tmp_idices_tar.append(idices_tar_)

    print(tmp_idices_src[5])
    print(tmp_idices_tar[5])

    # Pad the sentences(idices) into same length(max_length)
    idices_src_max = pad_sequences(tmp_idices_src, maxlen=max_length, padding="post")
    idices_tar_max = pad_sequences(tmp_idices_tar, maxlen=max_length, padding="post")

    # Check num of data after filtering
    print("num of idices_src_max =", len(idices_src_max))
    print("num of idices_tar_max =", len(idices_tar_max))

    # pading check
    print("pading check......")
    for i in range(500, 502):
        print(idices_src_max[i])
        print(idices_tar_max[i])
        print("-" * 100)
    return idices_src_max, idices_tar_max

