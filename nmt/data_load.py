import numpy as np
import os
import pickle


class DataLoad:
    """ DataLoad object loads the preprocessing data for training. 
    """

    def __init__(self, cfg):
        """initialize the preprocessing data from config.json
        Args:
            cfg (.json): config.json
        """
        self.tokenizer_enc = os.path.join(cfg["preprocessing"], cfg["tokenizer_enc"])
        self.tokenizer_dec = os.path.join(cfg["preprocessing"], cfg["tokenizer_dec"])
        self.embedding_matrix_enc = os.path.join(
            cfg["preprocessing"], cfg["embedding_matrix_enc"]
        )
        self.embedding_matrix_dec = os.path.join(
            cfg["preprocessing"], cfg["embedding_matrix_dec"]
        )
        self.indices_tr_enc = os.path.join(cfg["preprocessing"], cfg["indices_tr_enc"])
        self.indices_tr_dec = os.path.join(cfg["preprocessing"], cfg["indices_tr_dec"])
        self.indices_val_enc = os.path.join(
            cfg["preprocessing"], cfg["indices_val_enc"]
        )
        self.indices_val_dec = os.path.join(
            cfg["preprocessing"], cfg["indices_val_dec"]
        )

    def load_tokenizer(self):
        """
        Args:
            tokenizer_en (str): tokenizer of encoder
            tokenizer_de (str): tokenizer of decoder
        Returns:
            A tuple containing two tf.keras.preprocessing.text.Tokenizer
                1st: tokenizer of encoder
                2nd: tokenizer of decoder
        """
        with open(self.tokenizer_enc, "rb") as handle:
            tokenizer_encoder = pickle.load(handle)
        with open(self.tokenizer_dec, "rb") as handle:
            tokenizer_decoder = pickle.load(handle)
        print("num of index in encoder =", len(tokenizer_encoder.word_index) + 1)
        print("num of index in decoder =", len(tokenizer_decoder.word_index) + 1)
        return tokenizer_encoder, tokenizer_decoder

    def load_weight_matrix(self):
        """Load pre-trained embedding_matrix for tf.keras.layers.Embedding
        Args:
            embedding_matrix_en (str): .npy file, Pre-trained embedding_matrix of encoder
            embedding_matrix_de (str): .npy file, Pre-trained embedding_matrix of decoder
        Returns:
            A tuple containing two np.array
                1st: embedding_matrix of encoder
                2nd: embedding_matrix of decoder

        """
        embedding_matrix_encoder = np.load(self.embedding_matrix_enc)
        embedding_matrix_decoder = np.load(self.embedding_matrix_dec)
        size_of_vocabulary_encoder = embedding_matrix_encoder.shape[0]
        size_of_vocabulary_decoder = embedding_matrix_decoder.shape[0]
        print("embedding_matrix_encoder.shape =", embedding_matrix_encoder.shape)
        print("embedding_matrix_decoder.shape =", embedding_matrix_decoder.shape)
        print("size_of_vocabulary_encoder =", size_of_vocabulary_encoder)
        print("size_of_vocabulary_decoder =", size_of_vocabulary_decoder)

        return (
            embedding_matrix_encoder,
            embedding_matrix_decoder,
            size_of_vocabulary_encoder,
            size_of_vocabulary_decoder,
        )

    def load_samples(self):
        """ load the preprocessing data 
        Args:
            indices_tr_en (str): .npy file, training data of encoder, sorce 
            indices_tr_de (str): .npy file, training data of decoder, target 
            indices_val_en (str): .npy file, validation data of encoder, sorce 
            indices_val_de (str): .npy file, validation data of encoder, target 

        Returns:
            A tuple containing four np.array

        """
        indices_tr_enc = np.load(self.indices_tr_enc)
        indices_tr_dec = np.load(self.indices_tr_dec)
        indices_val_enc = np.load(self.indices_val_enc)
        indices_val_dec = np.load(self.indices_val_dec)
        print("idices_tr_encoder.shape =", indices_tr_enc.shape)
        print("idices_tr_decoder.shape =", indices_tr_dec.shape)
        print("idices_val_encoder.shape =", indices_val_enc.shape)
        print("idices_val_decoder.shape =", indices_val_dec.shape)
        return indices_tr_enc, indices_tr_dec, indices_val_enc, indices_val_dec

    def input_generator(
        self, idices_enc, idices_dec, num_samples, data_type="training"
    ):
        """ this function organizes the input data for encoder and decoder.
        The returns will be the inputs of DataGenerator.

        Args:
            idices_enc (np.array): indices sequence of source for encoder
            idices_dec (np.array): indices sequence of target for decoder
            num_samples (int): Set the number of samples for traning. 
            data_type (str, optional): "training" or "validation". Defaults to "training".

        Returns:
            (tuple)
            inp_enc (np.array): encoder's input indices sequence (source)
            tar_inp_dec (np.array): decoder's input indices sequence (target)
            tar_real_dec (np.array) decoder's reference indices sequence (target)

        """

        num_inp = num_samples  # num of dataset used in training
        inp_enc = idices_enc[:num_inp]
        tar = idices_dec[:num_inp]  # tar source
        tar_inp_dec = tar[:, :-1]  # decoder's input of tar
        tar_real_dec = tar[:, 1:]  # decoder's reference of tar
        print(f'{"-" * 30} {data_type} data information {"-" * 30}')
        print("inp_enc.shape:", inp_enc.shape)
        print("-" * 20)
        print("tar.shape:", tar.shape)
        print("-" * 20)
        print("tar_inp_dec.shape:", tar_inp_dec.shape)
        print("-" * 20)
        print("tar_real_dec.shape:", tar_real_dec.shape)
        print("-" * 20)

        return inp_enc, tar, tar_inp_dec, tar_real_dec
