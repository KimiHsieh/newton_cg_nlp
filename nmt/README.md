`main.py` trains the NMT tasks in Transofromer with different models
and different optimizers by using nested `for loop` based on `config.json`.

- Within outer loop, different models are trained.
  - Within inner loop, the model is trained with different optimizers.
  ```python
      for i, hyparams in enumerate(cfg_model[0:1]):
        ...
        ...
        for optimizer_info in optimizers:
          ...
          ...
  ```

`config.json` contains all the settings of hyperparameters and optimizers.

```json
"OPTIMIZERS": {
        "batch_size":64,
        "epochs": 1,
        "Adam": [
            {"learning_rate": 0.0001, "beta_1": 0.9, "beta_2": 0.98, "epsilon": 1e-9}
        ],
        "SGD": [
            {"learning_rate": 0.001, "momentum": 0.9}
        ],
        "Newton_CG": [
            {"learning_rate": 0.001, "tau": 1.0E+1}
        ]
    },
    "MODEL_HYPERPARAMETERS": [
        {"num_layers": 2, "num_heads": 15, "dropout_rate": 0.1, "dff": 2, "pe_inp": 1000, "pe_tar": 1000},
        {"num_layers": 1, "num_heads": 15, "dropout_rate": 0.1, "dff": 2, "pe_inp": 1000, "pe_tar": 1000}
    ],
    ...
    ...
```

`nmt.sh` is the shell script that executes the whole process according
to my thesis in the cluster of Chair of Scientiﬁc Computing at TUM.

Essentially, it runs the process as follows:

1. Train the Adam model first as the pre-trained model.
2. Load the Adam pre-trained model and train other models.
3. Output the results into the `checkpoints` directory.

```sh
checkpoints
    └── samples35644_2layers_10heads_2dff
        ├── Adam_lrCustomSchedule
        ├── Adam_lrCustomSchedule_40ep
        ├── Newton_CG_lr0.001_tau10.0
        ├── Newton_CG_lr0.01_tau10.0
        └── SGD_lr0.001
```

In `nmt.sh`, we can set the epochs and batch size of Adam pre-trained model.
1st argument is the number of epochs. 2nd is the batch size.

```sh
python adam_schedule.py 1 64
```

Turning off the pre-trained model, the model will be trained from scratch.

```json
"PRE_TRAINED": {
        "use_pretrained": false,
        ...
        ...
    }
```
