import numpy as np
import logging
import os
import sys
import json
import pickle
from translate_tools import *

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from collections import defaultdict
from matplotlib import pyplot as plt


def bleu_score(predicted, reference):
    pred_4_bleu = []
    ref_4_bleu = []
    for pred, ref in zip(predicted, reference):
        pred_4_bleu.append(pred[0])
        ref_4_bleu.append([ref[0]])

    bleu_dic = {}
    bleu_dic["1-grams"] = 100 * corpus_bleu(
        ref_4_bleu, pred_4_bleu, weights=(1.0, 0, 0, 0)
    )
    bleu_dic["1-2-grams"] = 100 * corpus_bleu(
        ref_4_bleu, pred_4_bleu, weights=(0.5, 0.5, 0, 0)
    )
    bleu_dic["1-3-grams"] = 100 * corpus_bleu(
        ref_4_bleu, pred_4_bleu, weights=(0.3, 0.3, 0.3, 0)
    )
    bleu_dic["1-4-grams"] = 100 * corpus_bleu(
        ref_4_bleu, pred_4_bleu, weights=(0.25, 0.25, 0.25, 0.25)
    )
    print("1-grams:", bleu_dic["1-grams"])
    print("1-2-grams:", bleu_dic["1-2-grams"])
    print("1-3-grams:", bleu_dic["1-3-grams"])
    print("1-4-grams:", bleu_dic["1-4-grams"])

    return bleu_dic


def data_info(solver):
    data_dict = {}
    if "Adam" in solver:
        data_dict["name"] = "Adam"
        data_dict["lr"] = "schedule"
    elif "SGD" in solver:
        data_dict["name"] = "SGD"
        idx = solver.index("0.")
        lr = float(solver[idx:])
        data_dict["lr"] = lr
    else:
        data_dict["name"] = "Newton_CG"
        lr_idx_start = solver.index("lr") + 2
        lr_idx_end = solver.index("_tau")
        tau_idx_start = solver.index("au") + 2
        lr = float(solver[lr_idx_start:lr_idx_end])
        tau = float(solver[tau_idx_start:-1])
        data_dict["lr"] = lr
        data_dict["tau"] = tau

    return data_dict


def bleu_plot(bleu, data_dict, texts_data_path, solver, plot_type="tr"):
    if plot_type == "tr":
        type_ = "Training"
    else:
        type_ = "Validation"
    labels = list(bleu.keys())
    scores = list(bleu.values())

    # Plot the bar graph
    fig = plt.figure(figsize=(8, 6))
    plot = plt.bar(labels, scores)

    # Add the data value on head of the bar
    for value in plot:
        height = value.get_height()
        plt.text(
            value.get_x() + value.get_width() / 2.0,
            1.002 * height,
            "%0.1f" % height,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Add labels and title
    if data_dict["name"] == "Newton_CG":
        name = data_dict["name"]
        tau = data_dict["tau"]
        lr = data_dict["lr"]
        s1 = f"{type_} BLEU Scores with {name}, lr={lr}, "
        s2 = r"$\tau=$" + str(tau)
        plt.title(s1 + s2)

    else:
        name = data_dict["name"]
        lr = data_dict["lr"]
        plt.title(f"{type_} BLEU Scores with {name}, lr={lr}")

    plt.ylabel("BLEU", fontsize=12)
    plt.xticks(range(4), labels, fontsize=12)
    # Display the graph on the screen
    plt.ylim((0, 100))
    bar_name = "bleu_" + solver + "_" + plot_type + ".png"
    fig_name = os.path.join(texts_data_path, bar_name)
    fig.savefig(fig_name, bbox_inches="tight")


if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        solver_name = sys.argv[2]
        type_ = sys.argv[3]
    except IndexError:
        print("Please add three arguments: model_name, solver_name and type(tr or val)")
        print(
            f"Ex. python bleu.py samples35644_2layers_10heads_256dff Newton_CG_lr0.01_tau5.0 tr"
        )
        sys.exit()
    texts_data_path = os.path.join("data", model_name, solver_name)
    if type_ == "tr":
        ref_path = os.path.join(texts_data_path, "ref_text_tr.txt")
        pred_path = os.path.join(texts_data_path, "pred_text_tr.txt")
        if not (os.path.isfile(ref_path) and os.path.isfile(pred_path)):
            print(f"No ref_text_tr.txt or pred_text_tr.txt found... ")
            sys.exit()
    else:
        ref_path = os.path.join(texts_data_path, "ref_text_val.txt")
        pred_path = os.path.join(texts_data_path, "pred_text_val.txt")
        if not (os.path.isfile(ref_path) and os.path.isfile(pred_path)):
            print(f"No ref_text_val.txt or pred_text_val.txt found... ")
            sys.exit()

    with open(ref_path, "rb") as handle:
        ref = pickle.load(handle)

    with open(pred_path, "rb") as handle:
        pred = pickle.load(handle)

    assert len(pred) == len(ref), "numbers of pred and ref are different."
    print(f"{ref_path} is loaded")
    print(f"{pred_path} is loaded")
    bleu = bleu_score(pred, ref)
    data_dict = data_info(solver_name)
    bleu_plot(bleu, data_dict, texts_data_path, solver_name, type_)

