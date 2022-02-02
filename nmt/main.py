import logging
import tensorflow as tf
import tensorflow.keras as keras
import transformer as tr
import data_load
import data_generator as dg
from utilities import *
import newton_cg as es
import json

print(tf.__version__)
set_tf_loglevel(logging.FATAL)


def loss_function(tar_real_onehot, pred):
    """calculate the loss between reference and decoder's output.

    Args:
        tar_real_onehot (np.array): decoder's reference indices sequence in one-hot form. 
                                    size = (batch_size, len_seq, target_vocab_size)
        pred (np.array): decoder's output indices sequence in one-hot form. 
                         size = (batch_size, len_seq, target_vocab_size)

    Returns:
        loss
    """

    real = tf.argmax(tar_real_onehot, axis=-1)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(tar_real_onehot, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(tar_real_onehot, pred):
    """calculate the accuracy of decoder's output w.r.t reference

    Args:
        tar_real_onehot (np.array): decoder's reference indices sequence in one-hot form. 
                                    size = (batch_size, len_seq, target_vocab_size)
        pred (np.array): decoder's output indices sequence in one-hot form. 
                         size = (batch_size, len_seq, target_vocab_size)

    Returns:
        accuracy
    """
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

    # Step2-0. Load models' hyparams
    with open("config.json") as j:
        cfg_model = json.load(j)["MODEL_HYPERPARAMETERS"]
    model_params = []

    # Step2-1. Loop over hyparams and create different models
    for i, hyparams in enumerate(cfg_model[0:1]):
        # kill previous model
        tf.keras.backend.clear_session()
        tf.reset_default_graph()

        num_layers = hyparams["num_layers"]
        num_heads = hyparams["num_heads"]
        rate = hyparams["dropout_rate"]
        dff = hyparams["dff"]
        pe_inp = hyparams["pe_inp"]  # positional encoding
        pe_tar = hyparams["pe_tar"]  # positional encoding
        d_model = embedding_matrix_enc.shape[1]

        # Step3-1. Load optimizers' information
        with open("config.json") as j:
            cfg_optimizers = json.load(j)["OPTIMIZERS"]
        optimizers = set_optimizers(cfg_optimizers, d_model)

        # Step3-2. Load checkpoints' information
        with open("config.json") as j:
            cfg_checkpts = json.load(j)["PRE_TRAINED"]

        # Step3-3. Loop over optimizers and create the model.
        for optimizer_info in optimizers:
            tf.keras.backend.clear_session()
            tf.reset_default_graph()

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

            model = keras.Model(
                inputs=[en_inps, de_inps], outputs=out, name=f"model{i}"
            )
            print("-" * 30, "Model Summary", "-" * 30)
            model.summary()

            print("-" * 30, "Start the new optimizer", "-" * 30)
            if optimizer_info[1] == "Newton_CG":
                optimizer, name, lr, tau = optimizer_info
                solver_name = name + "_lr" + str(lr) + "_tau" + str(tau)
                print(f"opt= {name}, lr={lr}, tau={tau}")
            else:
                optimizer, name, lr = optimizer_info
                solver_name = name + "_lr" + str(lr)
                print(f"opt= {name}, lr={lr}")

            # Step4. Set checkpoint directory
            model_name = f"samples{num_tr}_{num_layers}layers_{num_heads}heads_{dff}dff"
            checkpoint_dir = check_checkpts(model_name, solver_name)
            cp_callback = load_checkpts(cfg_checkpts, model, model_name, checkpoint_dir)
            csv_logger = create_CSVLogger(checkpoint_dir)

            # Step5. Training
            batch_size = cfg_optimizers["batch_size"]
            epochs = cfg_optimizers["epochs"]
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

