"""
DataGenerator takes care of memory consumption of large datasets by
generating batch dataset in real-time and feeding it right away to the deep learning model.
Since the newton_cg doesn't support SparseCategoricalCrossentropy,
every data(or sentence) is transformed to one-hot encoding. 
This issue results in memory exploding when the batch is fetched during training.
Therefore, we have to customize a data generators with Keras
"""

import numpy as np
import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
        self, inp, tar_inp, tar_real, batch_size=64, n_classes=8000, shuffle=True
    ):
        """Initialization"""
        self.batch_size = batch_size
        self.list_IDs = [i for i in range(inp.shape[0])]
        self.inp = inp
        self.tar_inp = tar_inp
        self.tar_real = tar_real
        self.inp_idices_len = inp.shape[1]
        self.tar_idices_len = tar_inp.shape[1]
        self.tar_real_idices_len = tar_real.shape[1]
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        inp_en = np.empty(
            (self.batch_size, self.inp_idices_len), dtype="int32"
        )  # faster than zeros
        tar_de = np.empty(
            (self.batch_size, self.tar_idices_len), dtype="int32"
        )  # faster than zeros
        tar_real_output = np.empty(
            (self.batch_size, self.tar_real_idices_len), dtype="int32"
        )  # faster than zeros
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            inp_en[i,] = self.inp[
                ID,
            ]

            # Store class
            tar_de[i] = self.tar_inp[
                ID,
            ]
            tar_real_output[i] = self.tar_real[
                ID,
            ]
        return (
            [inp_en, tar_de],
            keras.utils.to_categorical(tar_real_output, num_classes=self.n_classes),
        )

