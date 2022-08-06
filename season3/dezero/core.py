import numpy as np
import heapq
import weakref
import contextlib
import dezero


class Config:
    enable_backprop = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} dtype is not supported.")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, use_heap=False, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_sets = set()
        flag = -1e9

        def add_func(f):
            if f not in seen_sets:
                if not use_heap:
                    funcs.append(f)
                    seen_sets.add(f)
                    funcs.sort(key=lambda x: x.generation)
                else:
                    heapq.heappush(funcs, (-f.generation, flag, f))
                    seen_sets.add(f)
        add_func(self.creator)

        while funcs:
            if not use_heap:
                f = funcs.pop()
            else:
                f = heapq.heappop(funcs)[2]
            gys = [y().grad for y in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
                        flag += 1

                # 중간 출력값들의 기울기 삭제
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

    def reshape(self, shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype


class Parameter(Variable):
    pass


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]     # nd-array만 들어올 경우, Variable 클래스로 변환
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        # 역전파 활성모드
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError("This method should be run outside of Function class.")

    def backward(self, gy):
        raise NotImplementedError("This method should be run outside of Function class.")


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = x1 * gy
        gx1 = x0 * gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, power):
        self.power = power

    def forward(self, x):
        return x ** self.power

    def backward(self, gy):
        c = self.power
        x, = self.inputs
        gx = c * (x ** (c - 1)) * gy
        return gx


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def pow(x, power):
    return Pow(power)(x)


@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# 오버라이딩 적용 함수
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow