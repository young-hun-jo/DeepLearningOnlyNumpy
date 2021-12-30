import numpy as np


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
            a: LSTM 계층에서 흘러나온 가중치(hs의 행 개수 즉, 입력 시퀀스의 길이와 동일)
        
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