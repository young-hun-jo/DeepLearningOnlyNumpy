import numpy as np


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x  # identity function
        if self.target_transform is None:
            self.target_transform = lambda x: x
        self.data = None
        self.label = None
        self.prepare()  # 자식 클래스에서 구현

    def __getitem__(self, index):
        # case when index is both integer and slicing
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), \
                   self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        raise NotImplementedError(f"This method should be run outside of Dataset class.")


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(train=self.train)


class BigData(Dataset):
    # 100만개 데이터가 존재한다고 가정하고 정의한 예시 클래스
    def __getitem__(self, index):
        x = np.load("data/{}.npy".format(index))
        t = np.load("data/{}.npy".format(index))
        return x, t

    def __len__(self):
        return int(1e6)


def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t