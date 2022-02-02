import os
import sys
import logging
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import newton_cg as es


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Pre-train the Adam optimizer with a custom learning rate scheduler.
    https://arxiv.org/abs/1706.03762

    """

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if level >= logging.ERROR:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if level >= logging.WARNING:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    logging.getLogger("tensorflow").setLevel(level)


def set_optimizers(cfg_optimizers, d_model):
    optimizers = []
    if cfg_optimizers["Adam"]:
        for adam_kwargs in cfg_optimizers["Adam"]:
            if adam_kwargs["learning_rate"] == "CustomSchedule":
                adam_kwargs.pop("learning_rate")
                opt = tf.keras.optimizers.Adam(
                    learning_rate=CustomSchedule(d_model), **adam_kwargs
                )
                optimizers.append((opt, "Adam", "CustomSchedule"))
            else:
                opt = tf.keras.optimizers.Adam(**adam_kwargs)
                optimizers.append((opt, "Adam", adam_kwargs["learning_rate"]))
    if cfg_optimizers["SGD"]:
        for sgd_kwargs in cfg_optimizers["SGD"]:
            opt = tf.keras.optimizers.SGD(**sgd_kwargs)
            optimizers.append((opt, "SGD", sgd_kwargs["learning_rate"]))
    if cfg_optimizers["Newton_CG"]:
        for cg_kwargs in cfg_optimizers["Newton_CG"]:
            opt = es.EHNewtonOptimizer(**cg_kwargs)
            optimizers.append(
                (opt, "Newton_CG", cg_kwargs["learning_rate"], cg_kwargs["tau"])
            )
    print("-" * 30, "Optimizers Information", "-" * 30)
    for optimizer in optimizers:
        if optimizer[1] == "Newton_CG":
            _, name, lr, tau = optimizer
            print(f"opt= {name}, lr={lr}, tau={tau}")
        else:
            _, name, lr = optimizer
            print(f"opt= {name}, lr={lr}")
    return optimizers


def check_checkpts(model_name, solver_name):
    checkpoint_dir = os.path.join("checkpoints", model_name, solver_name)

    print("-" * 30, "checkpoint file information", "-" * 30)
    if not os.path.exists(checkpoint_dir):
        print(checkpoint_dir, " not found, build a directory......")
        os.makedirs(checkpoint_dir)
        print(checkpoint_dir, "is built.")
    else:
        print(checkpoint_dir, " directory already exists!")

    return checkpoint_dir


def load_checkpts(cfg_checkpts, model, model_name, checkpoint_dir):
    if cfg_checkpts["use_pretrained"] and model_name in cfg_checkpts:
        print("Use Adam pretrained model")
        pre_trained_checkpoint_dir = cfg_checkpts[model_name][
            "pre_trained_checkpoint_dir"
        ]
        if os.path.isfile(pre_trained_checkpoint_dir + "/cp.ckpt.index"):
            pre_trained_checkpoint_path = os.path.join(
                pre_trained_checkpoint_dir, "cp.ckpt"
            )
            model.load_weights(pre_trained_checkpoint_path)
            print(pre_trained_checkpoint_path, "is found!")
            print("Latest 1st pretrained model checkpoint restored!!")
    elif cfg_checkpts["use_pretrained"] and model_name not in cfg_checkpts:
        print(
            "Use Use Adam pretrained model, but there's no pretrained model, please check it"
        )
        sys.exit()
    else:
        print("Without Adam pretrained model")
        checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")
        if os.path.isfile(checkpoint_dir + "/cp.ckpt.index"):
            model.load_weights(checkpoint_path)
            print(checkpoint_path, "is found!")
            print("Latest checkpoint is restored!!")
        else:
            last_epoch = 0
            print("checkpoint not found, training from scratch。")

    checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")
    print("checkpoint_path:", checkpoint_path)
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=1
    )
    return cp_callback


def create_CSVLogger(checkpoint_dir):
    # Create a CSVLogger that saves the logs
    print("-" * 30, ".log file information", "-" * 30)
    if os.path.isfile(os.path.join(checkpoint_dir, "training_0.log")):
        log_lsts = [_ for _ in os.listdir(checkpoint_dir) if _.endswith(".log")]
        print(".log files exist!")
        print(".log files:", log_lsts)
        # create new filename
        idx = str(len(log_lsts))
        filename = f"training_{idx}.log"
        print("Create next .log file:", filename)
    else:
        print(".log file not found, training from scratch。")
        filename = "training_0.log"

    logger_path = os.path.join(checkpoint_dir, filename)
    print(logger_path, "is created!")
    csv_logger = CSVLogger(logger_path)

    return csv_logger
