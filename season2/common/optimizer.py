import numpy as np


# SGD 클래스
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params: list, grads: list):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
            

# Momentum 클래스
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params: list, grads: list):
        # v 변수 초기화
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
        
        # 파라미터 업데이트
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]
            
            
# AdaGrad 클래스
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params: list, grads: list):
        # h 변수 초기화
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))
        
        # 파라미터 업데이트
        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= (self.lr * grads[i]) / (np.sqrt(self.h[i]) + 1e-7)


# RMSProp 클래스
class RMSProp:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params: list, grads: list):
        # h 변수 초기화
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))
        
        # 파라미터 업데이트
        for i in range(len(params)):
            self.h[i] *= self.decay_rate
            self.h[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= (self.lr * grads[i]) / (np.sqrt(self.h[i]) + 1e-7)
            
            
# Adam
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)