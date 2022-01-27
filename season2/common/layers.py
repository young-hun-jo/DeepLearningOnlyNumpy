import numpy as np
from common.functions import *

# Relu 계층
class Relu:
    def __init__(self):
        self.mask = None
        self.params = []
        self.grads = []
        
    def forward(self, x: np.array):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
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
    
# Softmax 계층
class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx
    
    
# 행렬 곱(편향은 더하지 않음) 계층
class Matmul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        W, = self.params       # -> list안에 np.array가 한 개 있을 때, ,(콤마)를 붙여주면 바깥에있는 list가 언패킹되어 반환됨!
        out = np.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        # skip-gram 모델일 경우, 맥락 데이터가 3차원으로 들어올 경우
        if dout.ndim == 3:
            dout = np.sum(dout, axis=1)
        if self.x.ndim == 3:
            self.x = np.sum(self.x, axis=1)
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

    
# Sigmoid-with-Loss 계층
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None   # Sigmoid 거쳐나온 후 출력값
        self.t = None   # 정답 레이블
        
    
    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        
        # np.c_[a, b] : 1차원 array a, b를 칼럼으로 붙이는 역할
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        return self.loss
    
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        dx = (self.y - self.t) * dout / batch_size
        return dx


# 계산 병목현상 개선하는 임베딩 계층
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
        
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        
        # 일반 for loop로 구현
        #for i, word_id in enumerate(self.idx):
            #dW[word_id] += dout[i]
        
        # dout에 지정된 self.idx(word_id)에 대응하는 dW 행렬의 인덱스 값에 각각 더해줌
        np.add.at(dW, self.idx, dout) 
        return None


# BatchNormalization clone
class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx