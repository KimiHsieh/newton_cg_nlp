import logging
import os
import sys
import numpy as np

# from IPython.display import clear_output

import tensorflow as tf

print("tensorflow version", tf.__version__)
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.text import Tokenizer

import transformer as tr
import data_load
import data_generator as dg
from utilities import *
import newton_cg as es

import json
from pprint import pprint

set_tf_loglevel(logging.FATAL)


def loss_function(tar_real_onehot, pred):
    """ TODO
  Args:
    tar_real_onehot: target indices sequence. (batch_size, len_seq)
    pred: output of transformer. (batch_size, len_seq, target_vocab_size)
  Returns:
    output: TODO
  """
    real = tf.argmax(tar_real_onehot, axis=-1)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(tar_real_onehot, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(tar_real_onehot, pred):
    # Exclude the <pad>
    real = tf.argmax(tar_real_onehot, axis=-1)  # reverse one hot
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    pred_argmax = tf.cast(tf.argmax(pred, axis=2), dtype=real.dtype)
    accuracies = tf.equal(real, pred_argmax)
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please add 1st arg: epochs, 2nd arg: batch_sizes")
        sys.exit()
    # Step1. Data Preparation
    with open("config.json") as j:
        cfg_data = json.load(j)["DATA_PATH"]
    data = data_load.DataLoad(cfg_data)
    print("-" * 30, "data information", "-" * 30)
    tokenizer_enc, tokenizer_dec = data.load_tokenizer()
    (
        embedding_matrix_enc,
        embedding_matrix_dec,
        size_of_vocabulary_enc,
        size_of_vocabulary_dec,
    ) = data.load_weight_matrix()
    idices_tr_enc, idices_tr_dec, idices_val_enc, idices_val_dec = data.load_samples()

    num_tr = idices_tr_enc.shape[0]  # num of dataset used in training
    num_val = idices_val_enc.shape[0]  # num of dataset used in validation
    tr_inp, tr_tar, tr_tar_inp, tr_tar_real = data.input_generator(
        idices_tr_enc, idices_tr_dec, num_tr, data_type="training"
    )
    val_inp, val_tar, val_tar_inp, val_tar_real = data.input_generator(
        idices_val_enc, idices_val_dec, num_val, data_type="validation"
    )

    # Step2. Build the Model
    with open("config.json") as j:
        cfg_model = json.load(j)["MODEL_HYPERPARAMETERS"]
    model_params = []
    for i, hyparams in enumerate(cfg_model[0:1]):
        # kill previous model
        tf.keras.backend.clear_session()
        tf.reset_default_graph()

        epochs = int(sys.argv[1])
        batch_size = int(sys.argv[2])
        print("epochs=", epochs)
        print("batch_size=", batch_size)

        num_layers = hyparams["num_layers"]
        num_heads = hyparams["num_heads"]
        rate = hyparams["dropout_rate"]
        dff = hyparams["dff"]
        pe_inp = hyparams["pe_inp"]
        pe_tar = hyparams["pe_tar"]
        d_model = embedding_matrix_enc.shape[1]

        transformer = tr.Transformer(
            num_layers,
            d_model,
            num_heads,
            dff,
            size_of_vocabulary_enc,
            size_of_vocabulary_dec,
            pe_inp,
            pe_tar,
            embedding_matrix_enc,
            tr_inp.shape[1],
            embedding_matrix_dec,
            tr_tar_inp.shape[1],
            rate=rate,
        )

        # Create a Encoder
        en_inps = keras.Input(shape=(None,), name="encoder_inps")
        # Create a Decoder
        de_inps = keras.Input(shape=(None,), name="decoder_inps")
        # noinspection PyCallingNonCallable
        out, _ = transformer(en_inps, de_inps, True)

        model = keras.Model(inputs=[en_inps, de_inps], outputs=out, name=f"model{i}")
        print("-" * 30, "Model Summary", "-" * 30)
        model.summary()

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=CustomSchedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        # Set checkpoint directory
        model_name = f"samples{num_tr}_{num_layers}layers_{num_heads}heads_{dff}dff"
        solver_name = f"Adam_CustomSchedule_{epochs}ep"
        checkpoint_dir = check_checkpts(model_name, solver_name)
        print("-" * 30, solver_name, "-" * 30)
        checkpoint_dir = os.path.join("checkpoints", model_name, solver_name)
        checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")
        print("checkpoint_path:", checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1
        )
        csv_logger = create_CSVLogger(checkpoint_dir)

        # Step5. Training
        training_generator = dg.DataGenerator(
            tr_inp,
            tr_tar_inp,
            tr_tar_real,
            batch_size=batch_size,
            n_classes=size_of_vocabulary_enc,
        )
        validation_generator = dg.DataGenerator(
            val_inp,
            val_tar_inp,
            val_tar_real,
            batch_size=batch_size,
            n_classes=size_of_vocabulary_enc,
        )

        loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        model.compile(
            optimizer=optimizer, loss=loss_function, metrics=[accuracy_function]
        )

        hist = model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[cp_callback, csv_logger],
        )

