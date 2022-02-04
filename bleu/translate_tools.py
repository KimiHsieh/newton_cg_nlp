import numpy as np
import os
import pickle
import tensorflow as tf


def translate(source_indices_set, model):
    len_indices = len(source_indices_set[0])
    sorce_indices = []
    sorce = []
    predictions = []
    for i in range(len(source_indices_set)):
        translation = [2]
        for _ in range(len_indices):
            predict = model.predict(
                [source_indices_set[i : i + 1], np.asarray([translation])]
            )
            translation.append(np.argmax(predict[-1, -1]))
            if translation[-1] == 3:
                predictions.append(translation)
                sorce_indices.append(i)
                sorce.append(source_indices_set[i])

                break
    return predictions, sorce, sorce_indices


def detokenize(idices_predictions, idices_references, detokenizer):
    """Connects to the next available port.

    Args:
      idices_predictions: A port value greater or equal to 1024.
      idices_references: references indices form
      detokenizer: a dictionary turning indix to word

    Returns:
      pred, tuple: 0. a list containing split word. 1. sentece in string form  
      ref, tuple: 0. a list containing split word. 1. sentece in string form

    """
    pred = []
    ref = []
    for idices_pred, idices_ref in zip(idices_predictions, idices_references):
        trans_word_pred_lst = []
        trans_word_ref_lst = []

        for idx in idices_pred:
            if idx == 2 or idx == 3:
                continue
            if idx in detokenizer:
                trans_word_pred_lst.append(detokenizer[idx])
        trans_word_pred_str = " ".join(trans_word_pred_lst)
        pred.append((trans_word_pred_lst, trans_word_pred_str))

        for idx in idices_ref:
            if idx == 2 or idx == 3:
                continue
            if idx in detokenizer:
                trans_word_ref_lst.append(detokenizer[idx])
        trans_word_ref_str = " ".join(trans_word_ref_lst)
        ref.append((trans_word_ref_lst, trans_word_ref_str))

    return pred, ref


def create_ref(pred, src_indices, idices_tr_dec):
    ref = []
    for idx in src_indices:
        ref.append(idices_tr_dec[idx])
    assert len(pred) == len(ref), "numbers of pred and ref are different."
    return ref


def save_pred(data_save_path, pred, ref, type_="tr"):
    if type_ == "tr":
        pred_tr_data_save_path = os.path.join(data_save_path, "pred_tr.txt")
        ref_tr_data_save_path = os.path.join(data_save_path, "ref_tr.txt")
        with open(pred_tr_data_save_path, "wb") as fp:  # Pickling
            pickle.dump(pred, fp)
        with open(ref_tr_data_save_path, "wb") as fp:  # Pickling
            pickle.dump(ref, fp)
    else:
        pred_val_data_save_path = os.path.join(data_save_path, "pred_val.txt")
        ref_val_data_save_path = os.path.join(data_save_path, "ref_val.txt")
        with open(pred_val_data_save_path, "wb") as fp:  # Pickling
            pickle.dump(pred, fp)
        with open(ref_val_data_save_path, "wb") as fp:  # Pickling
            pickle.dump(ref, fp)


def load_pred(data_save_path):
    pred_tr_data_save_path = os.path.join(data_save_path, "pred_tr.txt")
    ref_tr_data_save_path = os.path.join(data_save_path, "ref_tr.txt")
    pred_val_data_save_path = os.path.join(data_save_path, "pred_val.txt")
    ref_val_data_save_path = os.path.join(data_save_path, "ref_val.txt")
    with open(pred_tr_data_save_path, "rb") as handle:
        pred_tr = pickle.load(handle)
    with open(ref_tr_data_save_path, "rb") as handle:
        ref_tr = pickle.load(handle)
    with open(pred_val_data_save_path, "rb") as handle:
        pred_val = pickle.load(handle)
    with open(ref_val_data_save_path, "rb") as handle:
        ref_val = pickle.load(handle)
    return pred_tr, ref_tr, pred_val, ref_val


def show_results(pred_detoken, ref_detoken, num, type_="tr"):
    if type_ == "tr":
        result_type = "Training"
    else:
        result_type = "Validation"
    for pred, ref in zip(pred_detoken[:num], ref_detoken[:num]):
        print(result_type)
        print(f'{f"pred":<10}{f"{pred[0]}":<10}')
        print(f'{f"ref":<10}{f"{ref[0]}":<10}')
        print("-" * 50)


def save_results(data_save_path, pred_text, ref_text, type_="tr"):
    pred_path = os.path.join(data_save_path, "pred_text_" + type_ + ".txt")
    ref_path = os.path.join(data_save_path, "ref_text_" + type_ + ".txt")
    with open(pred_path, "wb") as fp:  # Pickling
        pickle.dump(pred_text, fp)

    with open(ref_path, "wb") as fp:  # Pickling
        pickle.dump(ref_text, fp)

