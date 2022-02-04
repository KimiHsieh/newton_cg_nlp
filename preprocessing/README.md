`preprocessing.py` preprocesses the raw data as follows:

1. Load and preprocess raw dataset.
   1. Load the raw dataset from TED talk transcripts.
   2. Preprocess the raw dataset with proper format.
2. Tokenize the sentences.
3. Build the Word Embedding matrix with fastText.
4. Transform the text sentences into the indices sequences.
5. Make all indices sequence with the same length.
6. Save indices seqences into the numpy array as model's input data.

After preprocessing, this directory should contain:

- Portuguese and English tokenizers

  1. `tokenizer_pt.pickle`
  2. `tokenizer_en.pickle`

- Portuguese and English word embedding matrices used as word embedding layers in the model.

  1. `embedding_matrix_pt.npy`
  2. `embedding_matrix_en.npy`

- indices sequences as model's input data
  1. Portuguese training data `idices_tr_pt_np.npy`
  2. English training data `idices_tr_en_np.npy`
  3. Portuguese validation data `idices_val_en_np.npy`
  4. English validation data `idices_val_pt_np.npy`

The pre-trained Portuguese and English word embedding models, `cc.pt.300.vec` and `cc.en.300.vec` must be downloaded from [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) into this directory first.

### How to use

Run the `preprocessing.py`.

```sh
$ python preprocessing.py
```
