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
