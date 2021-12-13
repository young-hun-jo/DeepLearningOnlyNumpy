# 하나의 입력 벡터를 넣는 단일 RNN 계층 클래스
class RNN:
    """ 하나의 입력 벡터를 넣는 단일 RNN 계층 클래스
    
    Args:
        Wx: 입력 벡터 x에 곱해지는 가중치
        Wh: 이전 RNN 계층에서 흘러넘어온 은닉 상태 벡터 h에 곱해지는 가중치
        b: 편향 파라미터
    
    """
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None   # 역전파 시 사용할 데이터 저장

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        tanh = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b  # bias는 브로드캐스팅되어 알아서 확장 계산됨
        h_next = np.tanh(tanh)
        
        self.cache = x, h_prev, h_next
        return h_next
    
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        
        dtanh = dh_next * (1 - h_next**2)
        db = np.sum(dtanh, axis=0) # Repeat 노드 역전파이기 때문에 sum!
        dWh = np.matmul(h_prev.T, dtanh)
        dh_prev = np.matmul(dtanh, Wh.T)
        dWx = np.matmul(x.T, dtanh)
        dx = np.matmul(dtanh, Wx.T)
        
        # 깊은 복사 수행
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev

    
# 단일 RNN 계층이 T개 있는 TimeRNN 계층 구현(이 때, T란, 시계열 데이터 길이의 T를 의미)
class TimeRNN:
    """ 단일 RNN 계층이 T개 있는 TimeRNN 계층 구현
    
    Args:
        Wx: 입력 벡터 x에 곱해지는 파라미터
        Wh: 이전 RNN 계층으로부터 흘러들어오는 은닉 상태 벡터 h에 곱해지는 파라미터
        b: 편향 파라미터
        stateful: 은닉 상태를 유지할지 여부. 유지하면 Truncated BPTT 수행하면서 순전파를 끊지 않고 전달
    
    """
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = []  # 여러개의 단일 RNN 계층을 저장할 리스트
        
        self.h = None   # 다음 신경망 블록에 넘길 이전 신경망의 마지막 은닉 상태 벡터
        self.dh = None  # 이전 블록의 은닉 상태 기울기 값(이는 Truncated BPTT에서는 필요없지만 seq2seq에서 필요하기 때문이라고 함)
        self.stateful = stateful
        
    # 은닉 상태를 설정하는 함수
    def set_state(self, h):
        self.h = h
        
    # 은닉 상태를 초기화하여 은닉 상태 유지를 끊어버리는 함수
    def reset_state(self):
        self.h = None
    
    
    def forward(self, xs):
        """ xs라는 T 길이의 시계열 전체 입력 벡터를 순전파 수행
        
        Args:
            xs: T 길이의 시계열 전체 입력 벡터
        
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape     # (batch_size, time length, 입력벡터 차원 수)
        D, H = Wx.shape        # (입력벡터 차원 수, 은닉 상태 벡터 차원 수)
        
        self.layers = []
        # T길이의 RNN 계층 전체의 은닉 상태 벡터를 담을 배열 초기화
        hs = np.empty((N, T, H), dtype='f')
        
        # 순전파 유지 끊을 경우
        if not self.stateful or self.h is None: 
            self.h = np.zeros((N, H), dtype='f')
            
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)  # loop 다 돌고 마지막에 self.h에는 해당 블록의 최종 은닉 상태 벡터 들어있음
            hs[:, t, :] = self.h
            self.layers.append(layer)
            
        return hs
    
    
    def backward(self, dhs):
        """ T길이의 시계열 전체를 한 번에 역전파 수행
        
        Args:
            dhs: T길이 시계열 내의 모든 은닉 상태 벡터의 기울기
        
        """
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        
        dxs = np.empty((N, T, D), dtype='f') # T길이 시계열 내의 모든 입력 벡터의 기울기
        dh = 0
        grads = [0, 0, 0] # 하나의 RNN 계층에서의 Wx, Wh, b 파라미터의 기울기 담을 리스트
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 각 RNN 계층 당, 순전파 시 은닉상태 h 벡터가 두 갈래로 분기되었으므로 역전파 시에는 기울기를 sum!
            dx, dh = layer.backward(dhs[:, t, :] + dh) 
            dxs[:, t, :] = dx
            
            # 하나의 RNN 계층에서의 Wx, Wh, b 파라미터의 기울기 담기
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        # T길이 시계열 전체의 RNN 계층에 걸쳐서 만들어진 Wx, Wh, b 파라미터의 기울기 담기
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
            
        self.dh = dh  # 역전파 방향으로 했을 때, 마지막 은닉 상태 벡터의 기울기 값 저장 for seq2seq
        
        return dxs