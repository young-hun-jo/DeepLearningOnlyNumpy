from common.layers import *
from common.functions import sigmoid


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
    
    
# 타임 임베딩 계층
class TimeEmbedding:
    """ 입력되는 단어를 분산 표현 벡터로 바꾸어주는 임베딩 계층
    
    Args:
        W: 원본 입력 단어 즉, One-hot 형태로 되어있는 Sparse한 단어 벡터에 곱해주는 파라미터
           즉, 이 W를 곱해줌으로써 Embedding 계층 내에 Dense한 벡터로 변환됨!
    
    """
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
        
        
    def forward(self, xs):
        """ 원본 입력인 Sparse 벡터를 임베딩 벡터로 변환
        
        Args:
            xs: 원본 입력 단어 즉, One-hot 형태로 되어있는 Sparse한 벡터 단어들이 T개의 시계열 길이만큼 존재!
        
        """
        N, T = xs.shape
        V, D = self.W.shape
        
        out = np.empty((N, T, D), dtype='f')  # D는 임베딩 벡터 차원 수, N은 배치 사이즈
        self.layers = []
        
        # T길이의 모든 입력 Sparse 벡터를 모두 Dense 벡터로 변환하는 임베딩 계층 순전파 수행
        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        
        return out
    
    
    def backward(self, dout):
        N, T, D = dout.shape
        
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
            
        self.grads[0][...] = grad
        return None
    
    
# RNN 계층에서 계산되어 나온 은닉 상태 벡터에 행렬 곱을 수행해 최종 출력값(Softmax)이전 Score 계산 계층
class TimeAffine:
    """ RNN 계층에서 나온 은닉 상태 벡터에 행렬 곱 수행 -> 단, Numpy reshape을 활용해 효율적으로 계산
    
    Args:
        W: 은닉 상태 벡터에 곱해줄 파라미터
        b: 은닉 상태 벡터에 더해줄 편향 파라미터
    
    """
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
        
    def forward(self, x):
        """ 순전파 수행
        
        Args:
            x: 변수명은 x지만 실질적으로는 RNN 계층에서 흘러나온 은닉 상태 벡터
        
        """
        N, T, D = x.shape
        W, b = self.params
        
        rx = x.reshape(N*T, -1)
        out = np.matmul(rx, W) + b
        self.x = x
        
        return out.reshape(N, T, -1)
    
    
    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params
        
        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)
        
        db = np.sum(dout, axis=0)
        dW = np.matmul(rx.T, dout)
        drx = np.matmul(dout, W.T)
        dx = drx.reshape(N, T, -1)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx

    
    
# T길이 시계열의 모든 Softmax-with-Loss 계층 
class TimeSoftmaxWithLoss:
    """ T길이 시계열의 모든 Softmax-with-Loss 계층 순전파, 역전파를 수행
    
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1
        
    
    def forward(self, xs, ts):
        """ 순전파 수행
        
        Args:
            xs: TimeAffine 계층을 거쳐나온 Score 값
            ts: 정답 레이블
        
        """
        N, T, V = xs.shape # 여기서 V는 원본 입력 Sparse 벡터의 차원 수랑 동일함(즉, Vocabulary size)
        
        if ts.ndim == 3: # (배치사이즈, 시계열 길이, Vocabulary_size)
            ts = ts.argmax(axis=2) # 레이블 인코딩 형태로 변환
        
        mask = (ts != self.ignore_label)  # mask의 역할..?
        
        xs = xs.reshape(N*T, V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)
        
        # 소프트맥스 변환
        ys = softmax(xs)
        ls = np.log(ys[np.arange(N*T), ts])
        ls *= mask
        loss = -np.sum(ls)
        loss /= mask.sum()
        
        self.cache = (ts, ys, mask, (N, T, V))
        return loss
    
    
    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache
        
        dx = ys
        dx[np.arange(N*T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]
        
        dx = dx.reshape((N, T, V))
        
        return dx