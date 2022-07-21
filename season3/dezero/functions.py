import numpy as np
from dezero.core import Function
from dezero.core import as_variable


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gy = gy * -sin(x)
        return gy


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # 캐싱된 출력은 현재 약한 참조 중!
        gx = 1 - y ** 2
        return gx


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.reshape(x, self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)  # gy를 x_shape 형상으로 바꾸는 것이 목적


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    """
    shape: 바꾸려는 형상
    """
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
