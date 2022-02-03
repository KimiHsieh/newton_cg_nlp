`prediction.py` translates the source language to the target language.

```sh
Training data
pred      ['surgery', 'was', 'a', 'success']
ref       ['the', 'surgery', 'was', 'successful']
--------------------------------------------------
Validation data
pred      ['they', 'had', 'UNK', 'fish', 'with', 'UNK', 'UNK']
ref       ['did', 'they', 'eat', 'fish', 'and', 'chips']
```

`bleu.py` calculates the BLUE scores of the given model and optimizer.

<p align="center">
  <img src="data/samples35644_2layers_10heads_256dff/Newton_CG_lr0.01_tau10.0/bleu_Newton_CG_lr0.01_tau10.0_tr.png" alt="tr" width="600"/>
  <img src="data/samples35644_2layers_10heads_256dff/Newton_CG_lr0.01_tau10.0/bleu_Newton_CG_lr0.01_tau10.0_val.png" alt="val" width="600"/>
</p>
