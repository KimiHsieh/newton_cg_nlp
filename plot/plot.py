import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_params(solver_list):
    solvers = []
    for solver in solver_list:
        solvers.append(solver)

    # Set data_dicts
    keys = [
        "grads_and_vars",
        "lr",
        "tau",
        "grad_iter",
        "epoch",
        "tr_acc",
        "tr_loss",
        "val_acc",
        "val_loss",
    ]
    data_dicts = {}

    for solver in solvers:
        data_dicts[solver] = {}

    for solver in solvers:
        for key in keys:
            data_dicts[solver][key] = []

    for solver in solvers:
        if "CustomSchedule" and "ep" in solver:
            data_dicts[solver]["lr"] = "schedule"
            print(solver, data_dicts[solver]["lr"])
            continue

        if "Newton" not in solver:
            idx = solver.index("0.")
            lr = float(solver[idx:])
            data_dicts[solver]["lr"] = lr
            print(solver, data_dicts[solver]["lr"])
        else:
            lr_idx_start = solver.index("lr") + 2
            lr_idx_end = solver.index("_tau")
            tau_idx_start = solver.index("au") + 2
            lr = float(solver[lr_idx_start:lr_idx_end])
            tau = float(solver[tau_idx_start:-1])
            data_dicts[solver]["lr"] = lr
            data_dicts[solver]["tau"] = tau
            print(solver, data_dicts[solver]["lr"], data_dicts[solver]["tau"])

    return data_dicts


def load_data(solver_list, data_dicts, checkpoint_dir):
    for solver in solver_list:
        solver_dir = os.path.join(checkpoint_dir, solver)

        log_lsts = sorted([_ for _ in os.listdir(solver_dir) if _.endswith(".log")])
        for logfile in log_lsts:
            logger_path = os.path.join(solver_dir, logfile)
            f = open(logger_path)
            lines = f.readlines()

            cp_keys = ["epoch", "tr_acc", "tr_loss", "val_acc", "val_loss"]
            for line in lines[1:]:
                values = line.split(",")
                values[0] = int(values[0])
                values[1:] = np.asarray(values[1:], dtype="float32")
                for k, val in zip(cp_keys, values):
                    data_dicts[solver][k].append(val)
            f.close()

        data_dicts[solver]["epoch"] = [
            i for i in range(len(data_dicts[solver]["epoch"]))
        ]

    for solver in solver_list:
        data_dicts[solver]["epoch"] = (
            np.asarray(data_dicts[solver]["epoch"], dtype="int32") + 1
        )
        # print(solver)
        # print(data_dicts[solver]["epoch"])


def loss_plot(data_dicts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    adam_idx = 0
    sgd_idx = 0
    end = -1
    pre_trained_start = 20
    top = 0
    for solver in data_dicts.keys():
        top += 1
        lr = data_dicts[solver]["lr"]
        tau = data_dicts[solver]["tau"]
        if "Adam" in solver:
            label = f"Adam(lr={lr})"
            adam_idx += 1
            y_tr = data_dicts[solver]["tr_loss"][pre_trained_start:end]
            y_val = data_dicts[solver]["val_loss"][pre_trained_start:end]
            x = data_dicts[solver]["epoch"][pre_trained_start:end]
            ax1.plot(x, y_tr, label=label)
            ax2.plot(x, y_val, label=label)
        elif "SGD" in solver:
            label = f"SGD(lr={lr})"
            sgd_idx += 1
            y_tr = data_dicts[solver]["tr_loss"][pre_trained_start:end]
            y_val = data_dicts[solver]["val_loss"][pre_trained_start:end]
            x = data_dicts[solver]["epoch"][pre_trained_start:end]
            ax1.plot(x, y_tr, label=label)
            ax2.plot(x, y_val, label=label)
        elif "Newton" in solver:
            label = f"Newton-CG(lr={lr}, " + r"$\tau$=" + str(tau) + ")"
            x = data_dicts[solver]["epoch"][:end] + 20

            ax1.plot(
                x, data_dicts[solver]["tr_loss"][:end], label=label,
            )
            ax2.plot(
                x, data_dicts[solver]["val_loss"][:end], label=label,
            )

    ax1.set_xticks(range(20, 41))
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_title("Training Loss", fontsize=20)
    ax1.set_xlabel("epoch", fontsize=18)
    ax1.set_ylabel("loss", fontsize=18)

    ax2.set_xticks(range(20, 41))
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_title("Validation Loss", fontsize=20)
    ax2.set_xlabel("epoch", fontsize=18)
    ax2.set_ylabel("loss", fontsize=18)
    ax2.legend(loc="upper center", bbox_to_anchor=(-0.1, -0.1), ncol=5, fontsize=16)
    fig.suptitle(f"Loss Comparison", fontsize=24)
    fig.savefig(f"loss_comparison.png", bbox_inches="tight", pad_inches=0)


def accuracy_plot(data_dicts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    adam_idx = 0
    sgd_idx = 0
    es_idx = 0
    end = -1
    pre_trained_start = 20
    top = 0
    for solver in data_dicts.keys():
        top += 1
        lr = data_dicts[solver]["lr"]
        tau = data_dicts[solver]["tau"]
        if "Adam" in solver:
            label = f"Adam(lr={lr})"
            adam_idx += 1

            y_tr = data_dicts[solver]["tr_acc"][pre_trained_start:end]
            y_val = data_dicts[solver]["val_acc"][pre_trained_start:end]
            x = data_dicts[solver]["epoch"][pre_trained_start:end]

            ax1.plot(x, y_tr, label=label)
            ax2.plot(x, y_val, label=label)
        elif "SGD" in solver:
            label = f"SGD(lr={lr})"
            sgd_idx += 1

            y_tr = data_dicts[solver]["tr_acc"][pre_trained_start:end]
            y_val = data_dicts[solver]["val_acc"][pre_trained_start:end]
            x = data_dicts[solver]["epoch"][pre_trained_start:end]

            ax1.plot(x, y_tr, label=label)
            ax2.plot(x, y_val, label=label)
        elif "Newton" in solver:
            label = f"Newton-CG(lr={lr}, " + r"$\tau$=" + str(tau) + ")"
            x = data_dicts[solver]["epoch"][:end] + 20

            ax1.plot(
                x, data_dicts[solver]["tr_acc"][:end], label=label,
            )
            ax2.plot(
                x, data_dicts[solver]["val_acc"][:end], label=label,
            )

    ax1.set_xticks(range(21, 41))
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_title("Training Accuracy", fontsize=20)
    ax1.set_xlabel("epoch", fontsize=18)
    ax1.set_ylabel("acc", fontsize=18)

    ax2.set_xticks(range(21, 41))
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_title("Validation Accuracy", fontsize=20)
    ax2.set_xlabel("epoch", fontsize=18)
    ax2.set_ylabel("acc", fontsize=18)
    ax2.legend(loc="upper center", bbox_to_anchor=(-0.1, -0.1), ncol=5, fontsize=16)
    fig.suptitle(f"Accuracy Comparison", fontsize=24)
    fig.savefig(f"acc_comparison.png", bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    try:
        model_name = str(sys.argv[1])
        checkpoint_dir = os.path.join("../nmt/checkpoints", model_name)
        print(checkpoint_dir)
        solver_lsts = []
        for solver in os.listdir(checkpoint_dir):
            if solver == "Adam_lrCustomSchedule":  # pass the pre-trained model
                continue
            else:
                solver_lsts.append(solver)

        print(solver_lsts)
    except IndexError:
        print("Please provide a model name......")
        sys.exit()
    except FileNotFoundError:
        print(f"There is no {model_name}......")
        sys.exit()

    data_dicts = load_params(solver_lsts)
    load_data(solver_lsts, data_dicts, checkpoint_dir)
    loss_plot(data_dicts)
    accuracy_plot(data_dicts)

