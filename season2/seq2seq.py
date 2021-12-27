# Encoder 클래스 구현하기 -> LSTM 모델 사용
import numpy as np
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss

class Encoder:
    """ LSTM 모델 기반으로 하는 Encoder 클래스
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: LSTM 계층 내의 은닉 상태 벡터 차원 수(노드 수)
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False) # Encoder는 중간에 출력되는 은닉상태 벡터 생략
        
        # 모든 계층의 파라미터 취합
        self.params, self.grads = [], []
        self.layers = [self.embed, self.lstm]
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.hs = None  # Encoder에서 내뱉을 T길이의 은닉상태 벡터값들. 그런데 이 중 마지막 값만 필요함!
        
    
    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        hs = xs
        self.hs = hs
        return hs[:, -1, :]  # 마지막 은닉상태 벡터만 추출
    
    
    def backward(self, dh):
        dout = np.zeros_like(self.hs)   # T길이의 은닉상태 벡터 값들 빈 껍데기 형상 만들어놓기 for dh 넣을 위치 마련
        dout[:, -1, :] = dh             # 이 때, dh는 Decoder에서 역전파로 흘러들어오는 기울기 값
        
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    
# Decoder 클래스 구현하기 -> LSTM 모델 사용
class Decoder:
    """ LSTM 모델 기반으로 하는 Decoder 클래스(단, Softmax-with-Loss 계층은 포함 X)
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: LSTM 계층 내의 은닉 상태 벡터 차원 수(노드 수)
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # Embedding -> LSTM -> Affine 계층
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)
        
        self.layers = [self.embed, self.lstm, self.affine]
        
        # 모든 계층 파라미터 취합
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
        
    def forward(self, xs, h):
        """ Decoder 학습 시 순전파 수행
        
        Args:
            xs: T길이 시계열 데이터(학습이기 떄문에 이미 정답을 알고 있음!)
            h: Encoder에서 전달해준 은닉 상태 벡터
        
        """
        self.lstm.set_state(h)  # Decoder의 최초 은닉 상태 입력으로서 Encoder에서 전달해준 h 설정!
        for layer in self.layers:
            xs = layer.forward(xs)
        score = xs
        return score
    
    
    def backward(self, dscore):
        """ Decoder 역전파 수행
        
        Args:
            dscore: Softmax 계층으로부터 흘러들어온 국소적인 미분값
        """
        for layer in reversed(self.layers):
            dscore = layer.backward(dscore)
            
        # Encoder로 역전파를 수행할 때 전달해줄 h의 기울기 값!
        dh = self.lstm.dh
        return dh
    
    
    def generate(self, h, start_id, sample_size):
        """ Decoder에서 테스트 시 문장 생성하는 메소드
        
        Args:
            h: Encoder에서 전달해준 은닉 상태(Decoder에서 최초로 입력되는 은닉상태)
            start_id: h와 같이 Decoder에서 최초로 입력되는 데이터. 보통 <eos>와 같은 기호임
            sample_size: 생성하는 문자 개수
        
        """
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)  # Encoder가 전달해준 은닉 상태 Decoder에서 최초의 은닉상태로 입력 설정!
        
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))  # Mini-batch로 구현되므로 2d-array롤 변경
            for layer in self.layers:
                x = layer.forward(x)
            score = x
            
            # score이 가장 높은 index(문자ID) 추출
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
        
        return sampled
    
    
# Encoder-Decoder 연결 및 TimeSoftmax-with-Loss 계층 추가
class Seq2Seq:
    """ Encoder-Decoder 연결 및 TimeSoftmax-with-Loss 계층 추가한 전체 Seq2Seq 모델 클래스
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: LSTM 계층 내의 은닉 상태 벡터 차원 수(노드 수)
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.loss_layer = TimeSoftmaxWithLoss()
        
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
        
        
    def forward(self, xs, ts):
        # 정답 레이블로부터 Decoder의 입력 / 출력(정답) 데이터로 분할 -> 학습 시이기 때문!
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        
        h = self.encoder.forward(xs)                         # Encoder
        score = self.decoder.forward(decoder_xs, h)          # Decoder
        loss = self.loss_layer.forward(score, decoder_ts)    # Softmax-with-Loss
        return loss
    
    
    def backward(self, dout=1):
        dscore = self.loss_layer.backward(dout)
        dh = self.decoder.backward(dscore)
        dx = self.encoder.backward(dh)
        return dx
    
    
    def generate(self, xs, start_id, sample_size):
        # 테스트 시, Decoder에서 문장 생성!
        h = self.encoder.forward(xs)     # Encoder에서 은닉상태 반환
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled