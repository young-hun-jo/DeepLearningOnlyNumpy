import os
import subprocess
import numpy as np


# 변수 계산 그래프
def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = "" if v.name is None else v.name

    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)


# 함수 계산 그래프
def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    # 변수와 함수 간 edge
    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


# 계산 그래프 dot 언어로 표현
def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen_sets = set()

    def add_func(f):
        if f not in seen_sets:
            funcs.append(f)
            seen_sets.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


# 계산 그래프 이미지 변환 및 대화 셀에 출력
def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose=True)

    tmp_dir = os.path.join(os.path.expanduser('~/Desktop/gitrepo/DeepLearningOnlyNumpy/season3'), 'dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as g:
        g.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    filename = os.path.join(tmp_dir, to_file)
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, filename)
    subprocess.run(cmd, shell=True)

    try:
        from IPython import display
        return display.Image(filename=filename)
    except:
        pass


def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.
    Args:
        x (ndarray): Input array.
        shape:
    Returns:
        ndarray: Output array of the shape.
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.
    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.
    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)  # 'out' argument for inplace
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m


def convert_dtype(*args):
    if len(args) == 1:
        return np.array(args[0], dtype=np.float64)
    return np.array(args[0], dtype=np.float64), np.array(args[1], dtype=np.float64)


def get_conv_outsize(input_size, kernel_size, pad_size, stride_size):
    output_size = (input_size + 2 * pad_size - kernel_size) // stride_size + 1
    return output_size


def pair(x):
    if isinstance(x, int):
        return x, x
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError