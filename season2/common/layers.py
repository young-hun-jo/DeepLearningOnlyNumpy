import numpy as np
from common.functions import *


# Sigmoid 계층
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []  # 시그모이드 계층 자체에는 학습 파라미터 없음
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx
    
    
# 행렬 곱(편향은 더하지 않음) 계층
class Matmul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

    
# Affine(행렬 곱에 편향까지 더함) 계층
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        self.x = x
        out = np.matmul(x, W) + b
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

    
# Matmul 클래스를 활용한 Affine 계층 구현
class AffineMatmul:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        matmul = Matmul(W)
        out = matmul.forward(x) + b
        self.x = x
        return out
    
    def backward(self, dout):
        dx = matmul.backward(dout)
        db = np.sum(dout, axis=0)
        self.grads[1][...] = db
        return dx

    
# Softmax-with-Loss(CEE) 계층
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], [] # 파라미터 갱신값, 파라미터 별 기울기 담을 변수
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        
        # 정답 레이블이 원-핫 형태라면 레이블 형태로 변환(CEE로 손실 함수값 구하기 위해)
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
            
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1  # (y_i - t_i)를 의미!
        dx *= dout # 어차피 dout은 1임!
        dx /= batch_size
        
        return dx
