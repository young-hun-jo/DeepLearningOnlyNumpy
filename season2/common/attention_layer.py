import numpy as np
from common.layers import softmax


# (1) 선택 작업 계층
class WeightedSum:
    """ Encoder의 모든 은닉 상태 hs 와 LSTM 계층에서 흘러나온 가중치 a 간의 Weighted Sum
    
    """
    
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        
    def forward(self, hs, a):
        """ Weighted Sum 순전파 수행
        
        Args:
            hs: Encoder의 모든 은닉 상태 hs
            a: RNN 계층에서 출력한 은닉상태(현재 shape: (batch_size, 은닉상태 차원 수)
        
        """
        N, T, H = hs.shape   # (bacth_size, 입력시퀀스 길이, 은닉상태 차원 수)
        
        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)
        
        self.cache = (hs, ar)
        return c
    
    def backward(self, dc):
        """ Weighted Sum 역전파 수행
        
        Args:
            dc: 순전파 시, Affine 계층으로 전달한 맥락 벡터 c의 기울기 값
        
        """
        hs, ar = self.cache
        N, T, H = hs.shape
        
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)  # sum의 역전파
        dar = dt * hs
        dhs = dt * ar
        
        da = np.sum(dar, axis=2)  # repeat의 역전파
        
        return dhs, da
    
    
# (2) 가중치(a) 계산 계층
class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = softmax()
        self.cache = None
        
        
    def forward(self, hs, h):
        """ 가중치 계산 계층에서의 순전파 수행
        
        Args:
            hs: Encoder의 모든 은닉 상태 hs
            h: RNN 계층에서 출력한 은닉상태(현재 shape: (batch_size, 은닉상태 차원 수)
        """
        N, T, H = hs.shape
        
        hr = h.reshape(N, 1, H).repeat(T, axis=1)  # RNN 계층에서 나온 은닉상태 repeat
        
        # 내적 수행하여 Encoder의 각 은닉 상태와 RNN 계층에서 나온 은닉상태 간의 유사도 각각 계산
        t = hs * hr
        s = np.sum(t, axis=2)
        
        a = self.softmax.forward(s)
        self.cache = (hs, hr)
        return a
    
    def backward(self, da):
        """ 가중치 계산 계층에서의 역전파 수행
        
        Args:
            da: (1) 선택 작업 계층으로 보낸 가중치(a)의 기울기
        
        """
        hs, hr = self.cache
        N, T, H = hs.shape
        
        ds = self.softmax.backward(da)   
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)   # sum의 역전파
        dhr = dt * hs
        dhs = dt * hr
        dh = np.sum(dhr, axis=1)
        
        return dhs, dh