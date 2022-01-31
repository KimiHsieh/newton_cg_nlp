import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from dataset import *


if __name__ == "__main__":
    # Load and Preprocess raw dataset
    dataset_name = "ted_hrlr_translate/pt_to_en"
    train_examples, val_examples = data_load(dataset_name)
    seq_tr_pt, seq_tr_en = data_preprocessing(train_examples)
    seq_val_pt, seq_val_en = data_preprocessing(val_examples)

    # Tokenizer
    # Load tokenizers if they exsit
    # Else Build the Tokenizers for both Portuguese to English dataset
    if os.path.exists("tokenizer_pt.pickle") and os.path.exists("tokenizer_en.pickle"):
        with open("tokenizer_pt.pickle", "rb") as handle:
            tokenizer_pt = pickle.load(handle)

        with open("tokenizer_en.pickle", "rb") as handle:
            tokenizer_en = pickle.load(handle)
    else:
        tokenizer_pt = Tokenizer(num_words=8000, oov_token="UNK")
        tokenizer_en = Tokenizer(num_words=8000, oov_token="UNK")

        tokenizer_pt.fit_on_texts(seq_tr_pt)
        tokenizer_en.fit_on_texts(seq_tr_en)

        size_of_vocabulary_pt_s = 7999 + 1  # +1 for padding
        size_of_vocabulary_en_s = 7999 + 1  # +1 for padding

        # Save the tokenizer
        with open("tokenizer_pt.pickle", "wb") as handle:
            pickle.dump(tokenizer_pt, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("tokenizer_en.pickle", "wb") as handle:
            pickle.dump(tokenizer_en, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Word Embedding
    # Skip this step if they already exsit
    # Else Build the embedding_matrix for both Portuguese to English dataset
    # The embedding_matrix is used as embedding layer in model.
    if os.path.exists("embedding_matrix_pt.npy") and os.path.exists(
        "embedding_matrix_en.npy"
    ):
        pass
    else:
        pt_file = "cc.pt.300.vec"
        en_file = "cc.en.300.vec"
        embeddings_dict_pt = build_word_embedding_matrix(pt_file)
        embeddings_dict_en = build_word_embedding_matrix(en_file)
        embedding_matrix_pt = build_word_embedding_matrix(embeddings_dict_pt)
        embedding_matrix_en = build_word_embedding_matrix(embeddings_dict_en)
        print(f"embedding_matrix_pt.shape = {embedding_matrix_pt.shape}")
        print(f"embedding_matrix_en.shape = {embedding_matrix_en.shape}")
        np.save("embedding_matrix_pt.npy", embedding_matrix_pt)
        np.save("embedding_matrix_en.npy", embedding_matrix_en)

    # Convert text sequence into index sequence
    print("Convert text sequence into index sequence......")
    print("training:")
    idices_tr_pt, idices_tr_en = text_to_indices(
        tokenizer_pt, tokenizer_en, seq_tr_pt, seq_tr_en
    )
    print("validation:")
    idices_val_pt, idices_val_en = text_to_indices(
        tokenizer_pt, tokenizer_en, seq_val_pt, seq_val_en
    )

    # Make all indices sequence with the same length.
    MAX_SIZE = 20
    print("Make all indices sequence with the same length.......")
    print("training:")
    idices_tr_pt_max, idices_tr_en_max = filter_and_padding(
        idices_tr_pt, idices_tr_en, 20
    )
    print("validation:")
    idices_val_pt_max, idices_val_en_max = filter_and_padding(
        idices_val_pt, idices_val_en, 20
    )

    # Save indices seqences as numpy array for model's input data
    # convert indices sequences into numpy and save
    idices_tr_pt_np = np.array(idices_tr_pt_max)
    idices_tr_en_np = np.array(idices_tr_en_max)
    idices_val_pt_np = np.array(idices_val_pt_max)
    idices_val_en_np = np.array(idices_val_en_max)

    # save
    np.save("idices_tr_pt_np.npy", idices_tr_pt_np)
    np.save("idices_tr_en_np.npy", idices_tr_en_np)
    np.save("idices_val_pt_np.npy", idices_val_pt_np)
    np.save("idices_val_en_np.npy", idices_val_en_np)

    print(f"Portuguese training data shape = {idices_tr_pt_np.shape}")
    print(f"English training data shape = {idices_tr_en_np.shape}")
    print(f"Portuguese validation data shape = {idices_val_pt_np.shape}")
    print(f"English validation data shape = {idices_val_en_np.shape}")
