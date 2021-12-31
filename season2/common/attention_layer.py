import numpy as np
from common.layers import Softmax


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
        self.softmax = Softmax()
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
    
    
    
# (3) 결합계층
class Attention:
    """ (1) 은닉상태, 가중치 간 Weighted sum 계층, (2) 가중치 계산 계층을 결합하는 클래스
    
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()  # (2) 가중치 계산 계층
        self.weight_sum_layer = WeightedSum()             # (1) 은닉상태, 가중치 간 Weigted sum 계층
        self.attention_weight = None
    
    
    def forward(self, hs, h):
        """ (3) 결합 계층의 순전파
        
        Args:
            hs: Encoder의 모든 은닉 상태 hs
            h: RNN 계층에서 출력한 은닉상태(현재 shape: (batch_size, 은닉상태 차원 수)
        
        """
        # (2) 가중치 계산
        a = self.attention_weight_layer.forward(hs, h)
        # (1) Weighted sum 계층
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    
    def backward(self, dout):
        """(3) 결합 계층의 역전파
        
        Args:
            dout: Affine 계층으로부터 흘러들어오고 있는 국소적인 미분값
        
        """
        # 순전파 시, hs가 분기(repeat)되어 (1),(2) 계층으로 흘러들어갔으므로 역전파 시 sum!
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        
        return dhs, dh
    
    
    
class TimeAttention:
    """입력 시퀀스 길이(T) 전체를 처리하는 Time Attention 계층
    
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None
        
    
    def forward(self, hs_enc, hs_dec):
        """ Time Attention 계층의 순전파(학습 시)
        
        Args:
            hs_enc: Encoder의 모든 은닉 상태 hs
            hs_dec: 출력 시퀀스 길이 만큼의 RNN 계층에서 나오는 은닉상태 벡터 값 (batch_size, 출력시퀀스 길이, 은닉상태 차원 수)
        
        """
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)   # T개의 Attention 계층에서 나오는 출력값 담을 빈 껍데기 생성
        self.layers = []
        self.attention_weights = []
        
        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
        
        return out
    
    
    def backward(self, dout):
        """ Time Attention 계층의 역전파(학습 시)
        
        Args:
            dout: Affine 계층으로부터 흘러들어오는 기울기 값
        
        """
        N, T, H = dout.shape
        dhs_enc = 0                    # Encoder의 hs가 T개의 Attention 계층들로 분기되었기 때문에 이를 역전파하면 sum 하므로 이를 위한 값 초기화
        dhs_dec = np.empty_like(dout)  # Decoder의 LSTM 계층 방향으로 역전파될 때 저장할 기울기
        
        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh
            
        return dhs_enc, dhs_dec