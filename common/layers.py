from common.functions import *
import numpy as np

#=============
# 활성화 함수 계층
#=============

class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x: np.array):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    

class Sigmoid:
    def __init__(self):
        self.out = None
        
    def forward(self, x: np.array):
        out = sigomid(x)
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx
    

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None
        
    def forward(self, x: np.array):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        # 행렬 곱
        out = np.matmul(self.x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.matmul(dout, self.W.T)
        self.dW = np.matmul(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)
        return dx
    

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x: np.array, t: np.array):
        """
        t: 2차원의 One-hot 형태
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.x.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
