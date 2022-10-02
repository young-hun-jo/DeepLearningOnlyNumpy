import numpy as np
import dezero
from dezero.core import Function
from dezero.core import as_variable
from dezero.core import as_array
from dezero import utils



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
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gw = matmul(x.T, gy)
        return gx, gw


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = 2 * diff * (gy / len(diff))
        gx1 = -gx0
        return gx0, gx1


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y
        return gx


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        gx = f(gy)
        return gx


class GetItemGrad(Function):
    def __init__(self, slices, x_shape):
        self.slices = slices
        self.x_shape = x_shape

    def forward(self, gy):
        gx = np.zeros_like(self.x_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)  # 곱(*) 연산으로 and 논리 연산 수행
        gx = gy * mask   # clipping 된 값은은 기울기가 0이 되도록
        return gx


class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        # 수식으로 아직 이해 X
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)   # average CEE
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLASS_N = x.shape

        gy *= 1 / N
        y = softmax(x)
        t_one_hot = np.eye(CLASS_N, dtype=t.dtype)[t.data]  # 레이블 값을 인덱스로하는 원소값이 1인 행 벡터들만 슬라이싱
        y = (y - t_one_hot) * gy
        return y


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x, axes=None):
    return Transpose(axes)(x)


def broadcast_to(x, shape):
    return BroadcastTo(shape)(x)


def sum_to(x, shape):
    return SumTo(shape)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


def matmul(x, W):
    return MatMul()(x, W)


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def mean_squared_error_oom(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    return sum(diff ** 2) / len(diff)


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None  # t가 참조하는 데이터를 메모리에서 제거
    return y


def sigmoid(x):
    return Sigmoid()(x)


def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


def exp(x):
    return Exp()(x)


def get_item(x, slices):
    return GetItem(slices)(x)


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


def log(x):
    return Log()(x)


def softmax_simple(x, axis=1):
    if not isinstance(x, np.ndarray):
        max_x = x.data.max(axis=axis)   # batch도 고려
    else:
        max_x = x.max(axis=axis)
    x = as_variable(x - max_x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


def softmax(x, axis=1):
    return Softmax(axis)(x)


def softmax_cross_entropy_simple(x, t):
    # t는 레이블 인코딩 상태의 Dezero 인스턴스 객체
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]  # 로직 이해 -> 블로그/책 필기 참조
    y = -1 * sum(tlog_p) / N
    return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return as_variable(as_array(acc))


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if dezero.Config.train:
        scale = 1 - dropout_ratio
        mask = np.random.rand(*x.shape) > dropout_ratio
        y = x * mask / scale
        return y
    else:
        return x


from dezero.functions_conv import im2col
from dezero.functions_conv import conv2d_simple