from common.functions import sigmoid
from common.layers import *


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
    
    
    
# 총 4개의 게이트의 각 Wx, Wh, b 파라미터를 결합해 Affine 변환으로 한 번에 계산!
class LSTM:
    """ 단일 LSTM 계층 클래스 구현
    
    Args:
        Wx: 4개의 게이트에서 각각 입력 벡터 Xt 에 곱해지는 파라미터 Wx 4개
        Wh: 4개의 게이트에서 각각 이전 은닉 상태 벡터 h_t-1 에 곱해지는 파라미터 Wh 4개
        b: 4개의 게이트에서 각각 더해지는 편향 b 파라미터 4개
    
    """
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
        
    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape  # 은닉 상태 벡터 차원 수 (batch_size, 노드 수)
        
        # 총 4개의 게이트에서의 아핀 변환을 한 번에 계산
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        
        # slicing 해서 각 게이트에 보내기
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        
        c_next = f * c_prev + g * i
        h_next = np.tanh(c_next) * o
        
        self.cache = (x, h_prev, c_prev, f, g, i, o, c_next) # 역전파 시 사용할 데이터들 캐싱해두기
        
        return h_next, c_next
    
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, f, g, i, o, c_next = self.cache
        #===============
        # 게이트 역전파 수행
        #===============
        tanh_c_next = np.tanh(c_next)
        
        ds = dh_next * o * (1 - tanh_c_next**2) + dc_next
        
        dc_prev = ds * f  # 이전 기억 셀의 기울기
        
        # output 게이트
        do = dh_next * tanh_c_next
        do *= o * (1 - o)
        # input 게이트
        di = ds * g
        di *= i * (1 - i)
        # 새로운 기억 셀(main 게이트)
        dg = ds * i
        dg *= (1 - g**2)
        # forget 게이트
        df = ds * c_prev
        df *= f * (1 - f)
        
        # 4개 게이트 기울기 가로로 결합, horizontal stack
        dA = np.hstack((df, dg, di, do))
        
        #=================================
        # Affine 변환(행렬 곱)에 대한 역전파 수행
        #=================================
        # 파라미터 기울기 계산
        dWx = np.matmul(x.T, dA)
        dWh = np.matmul(h_prev.T, dA)
        db = dA.sum(axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        # 입력, 은닉상태 벡터 기울기 계싼
        dx = np.matmul(dA, Wx.T)
        dh_prev = np.matmul(dA, Wh.T)
        
        return dx, dh_prev, dc_prev
    
    
    
# T 길이의 시계열 데이터를 한번에 처리하는 LSTM 계층
class TimeLSTM:
    """ 단일 LSTM 계층이 T개 있는 TimeLSTM 계층 구현
    
    Args:
        Wx: 입력 벡터 x에 곱해지는 파라미터
        Wh: 이전 LSTM 계층으로부터 흘러들어오는 은닉 상태 벡터 h에 곱해지는 파라미터
        b: 편향 파라미터
        stateful: 은닉 상태를 유지할지 여부. 유지하면 Truncated BPTT 수행 시 순전파를 끊지 않고 전달
        
    """
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = []
        
        self.h, self.c = None, None   # 은닉상태벡터, 기억셀
        self.dh = None                # 이전 블록의 은닉 상태 기울기 값(이는 Truncated BPTT에서는 필요없지만 seq2seq에서 필요하기 때문이라고 함)
        self.stateful = stateful
    
    def forward(self, xs):
        """ xs라는 T 길이의 시계열 전체 입력 벡터를 순전파 수행
        
        Args:
            xs: T 길이의 시계열 전체 입력 벡터
            
        """
        Wx, Wh, b = self.params
        N, T, D = xs.shape       # (batch_size, time length, 입력벡터 차원 수)
        H = Wh.shape[0]          # 은닉상태 벡터 차원 수
        
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')  # T 길이의 LSTM 계층 전체의 은닉 상태 벡터를 담을 배열 초기화
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
            
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)  # loop다 돌면 마지막 은닉상태,기억셀이 들어있음!
            hs[:, t, :] = self.h
            
            self.layers.append(layer)
            
        return hs
    
    
    def backward(self, dhs):
        """ T길이의 시계열 전체를 한 번에 Truncated BPTT 수행
        
        Args:
            dhs: T길이 시계열 내의 모든 은닉 상태 벡터의 기울기
        
        """
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]  # 입력 벡터 차원 수
        
        dxs = np.empty((N, T, D), dtype='f') # T길이 시계열 내의 모든 입력 벡터의 기울기
        dh, dc = 0, 0
        
        grads = [0, 0, 0] # 하나의 LSTM 계층에서 Wx, Wh, b 파라미터의 기울기 담을 리스트
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)  # dh는 분기의 역전파므로 sum
            dxs[:, t, :] = dx
            
            # 하나의 LSTM 계층에서 Wx, Wh, b 파라미터의 기울기 담기
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        
        # T길이 시계열 전체의 LSTM 계층에 걸쳐서 만들어진 Wx, Wh, b 파라미터 기울기 담기
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
            
        self.dh = dh  # 역전파 방향으로 보았을 때의 마지막 은닉 상태 벡터 기울기 값 캐싱 for seq2seq
        
        return dxs
    
    # 은닉상태, 기억셀 설정 함수
    def set_state(self, h, c=None):
        self.h, self.c = h, c
    
    # 은닉상태, 기억셀 초기화하여 순전파 시에도 유지 끊어버리기
    def reset_state(self):
        self.h, self.c = None, None