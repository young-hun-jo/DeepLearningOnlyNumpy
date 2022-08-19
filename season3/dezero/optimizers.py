import numpy as np


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        # preprocess parametrs
        for f in self.hooks:
            f(p)

        # update parametrs
        for param in params:
            self.update_one(param)

    def update_one(self):
        raise NotImplementedError("This method should be run outside of Optimizer class.")

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.lr = learning_rate

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.lr = learning_rate
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param)

        v = self.momentum * self.vs[v_key]
        v -= self.lr * param.grad.data
        param.data += v

