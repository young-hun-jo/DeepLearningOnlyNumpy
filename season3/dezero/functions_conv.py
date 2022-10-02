import numpy as np
from dezero import Function
from dezero import as_variable
from dezero.functions import linear
from dezero.utils import pair, get_conv_outsize


def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, PH, SH)
    OW = get_conv_outsize(W, KW, PW, SW)

    # convert image to 2d-array(row vector)
    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    # convert filter(all kernel) to 2d-array(column-vector)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, PH, SH)
    OW = get_conv_outsize(W, KW, PW, SW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


class Im2Col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super(Im2Col, self).__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx


def im2col(x ,kernel_size, stride=1, pad=0, to_matrix=True):
    y = Im2Col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2Im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super(Col2Im, self).__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    y = Col2Im(input_shape, kernel_size, stride, pad, to_matrix)(x)
    return y


def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, PH, SH)
    OW = get_conv_outsize(W, KW, PW, SW)

    img = np.pad(img,
                 ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                 mode='constant', constant_values=(0,))
    # np.ndarray는 인자로 넣어준 shape의 텐서를 만들어줌
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, PH, SH)
    OW = get_conv_outsize(W, KW, PW, SW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                   dtype=col.dtype)
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
    return img[:, :, PH: H+PH, PW: W+PW]