import math
import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)

        self.reset()  # init iteration count

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.iteration = 0
            raise StopIteration("One epoch is finished. Iteration count is initialized.")

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size: (i+1) * batch_size]
        batch_x, batch_t = self.dataset[batch_index]

        self.iteration += 1
        return batch_x, batch_t

    # checking for example batch data
    def next(self):
        return self.__next__()
