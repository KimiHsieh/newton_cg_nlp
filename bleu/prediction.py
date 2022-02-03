# Tensorflow
# %tensorflow_version 1.
import tensorflow as tf
import newton_cg as es
import tensorflow.keras as keras

print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import sys
import json
import pickle
from translate_tools import *

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, "../nmt")
import transformer as tr
import data_load
from utilities import *

logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please add arg optimizer")
        print(f"Ex. python prediction.py Newton_CG_lr0.01_tau5.0")
        sys.exit()

    # Step1. Data Preparation
    with open("model_config.json") as j:
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

    # Step2. Load models' hyparams
    with open("model_config.json") as j:
        hyparams = json.load(j)["MODEL_HYPERPARAMETERS"]

    tf.keras.backend.clear_session()
    tf.reset_default_graph()

    num_layers = hyparams["num_layers"]
    num_heads = hyparams["num_heads"]
    rate = hyparams["dropout_rate"]
    dff = hyparams["dff"]
    pe_inp = hyparams["pe_inp"]  # positional encoding
    pe_tar = hyparams["pe_tar"]  # positional encoding
    d_model = embedding_matrix_enc.shape[1]

    # Step3. See if the checkpoint exists.
    model_name = f"samples{num_tr}_{num_layers}layers_{num_heads}heads_{dff}dff"
    solver_name = sys.argv[1]
    solver_dir = os.path.join("../nmt/checkpoints", model_name, solver_name)
    if not os.path.isfile(solver_dir + "/cp.ckpt.index"):
        print(f"No checkpoint found... ")
        sys.exit()

    checkpoint_path = os.path.join(solver_dir, "cp.ckpt")
    print(f"{model_name} {solver_name} checkpoint is found!!")

    # Step4. Build the model.
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

    model = keras.Model(inputs=[en_inps, de_inps], outputs=out, name=f"model")
    print("-" * 30, "Model Summary", "-" * 30)
    model.summary()

    # Step5. Load exsisting model checkpoint
    model.load_weights(checkpoint_path)
    print(f"{model_name} {solver_name} checkpoint is loaded!!")

    # Step6. Translate(Prediction) and save data.
    # Note: translate() takes 2 to 3 hours for 35,644 training sentences.
    data_save_path = os.path.join("data", model_name, solver_name)
    print(f" data_save_path = {data_save_path}")
    if not os.path.exists(data_save_path):
        print(data_save_path, " not found, build a directory......")
        os.makedirs(data_save_path)
        print(data_save_path, "is built.")
        pred_tr, src_tr, src_indices_tr = translate(idices_tr_enc, model)
        pred_val, src_val, src_indices_val = translate(idices_val_enc, model)
        ref_tr = create_ref(pred_tr, src_indices_tr, idices_tr_dec)
        ref_val = create_ref(pred_val, src_indices_val, idices_val_dec)
        save_pred(data_save_path, pred_tr, ref_tr, type_="tr")
        save_pred(data_save_path, pred_val, ref_val, type_="val")
    else:
        pred_tr, ref_tr, pred_val, ref_val = load_pred(data_save_path)

    # Step7. detokenize pred and ref from indices seq into string seq.
    tar_detokenizer = dict(map(reversed, tokenizer_dec.word_index.items()))
    pred_text_tr, ref_text_tr = detokenize(pred_tr, ref_tr, tar_detokenizer)
    pred_text_val, ref_text_val = detokenize(pred_val, ref_val, tar_detokenizer)
    show_results(pred_text_tr, ref_text_tr, 3, type_="Training data")
    show_results(pred_text_val, ref_text_val, 3, type_="Validation data")
    save_results(data_save_path, pred_text_tr, ref_text_tr, type_="tr")
    save_results(data_save_path, pred_text_val, ref_text_val, type_="val")

