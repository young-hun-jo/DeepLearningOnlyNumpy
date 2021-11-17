# 파라미터 갱신하는 최적화 기법 종류별 클래스 구현
import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params: dict, grads: dict):
        """ 파라미터 갱신
        
        Parameters
        ----------
        params: 넘파이로 신경망 구현 시 얻을 수 있는 Weight, Bias 딕셔너리
        grads: 넘파이로 신경망 구현 시 얻을 수 있는 파라미터 별 기울기 값이 담겨 있는 딕셔너리
        
        => 나머지 최적화 기법의 params, grads 인자도 의미 동일
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]
            

class Momentum:
    """ 기울기(Gradient)를 보정하는 모멘텀 최적화 기법
    
    Parameters
    ----------
    lr: 초기 학습률
    momentum: 아무런 힘을 받지 않았을 때도록 하강시켜줄 수 있도록 하는 모멘텀 계수
    
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params: dict, grads: dict):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = (self.momentum * self.v[key]) - (self.lr * grads[key])
            params[key] += self.v[key]


class AdaGrad:
    """ 학습률을 개별 파라미터에 맞춤화되어 조정하는 최적화 기법
    
    Parameters
    ----------
    lr: 초기 학습률
    h: 기존 기울기를 계속 제곱해준 후 파라미터 갱신 시 역수로 취하는 값(즉, 과거에 크게 학습한 파라미터는 이제는 작게 학습하도록 해줌)
    
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params: dict, grads: dict):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= (self.lr * grads[key]) / (np.sqrt(self.h[key]) + 1e-7)

            
class RMSProp:
    """ AdaGrad에서 더 나아가 기울기를 계속 곱해줄 때, 과거의 기울기는 조금 반영하고 최신 기울기는 크게 반영하는 지수이동평균 적용한 기법
    
    Parameters
    ----------
    lr: 초기 학습률
    decay_rate: 기울기에 지수이동평균 적용할 계수
    
    """
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params: dict, grads: dict):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * (grads[key] * grads[key])
            params[key] -= (self.lr * grads[key]) / (np.sqrt(self.h[key]) + 1e-7)
            

class Adam:
    """ AdaGrad와 Momentum 기법을 혼합하고 편향 보정 항을 추가한 최적화 기법
    
    """
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params: dict, grads: dict):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        lr_r = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_r * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
