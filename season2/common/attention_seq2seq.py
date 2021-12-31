import numpy as np
import sys
sys.path.append('..')
from common.time_layers import *
from common.seq2seq_layer import Encoder, Seq2Seq
from common.attention_layer import TimeAttention


class AttentionEncoder(Encoder):
    """ Attention 메커니즘 적용한 Seq2Seq의 Encoder 클래스
    
    """
    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        hs = xs
        return hs # 모든 은닉상태 벡터 추출
    
    def backward(self, dhs):
        for layer in reversed(self.layers):
            dhs = layer.backward(dhs)
            
        return dhs
    
    
    
class AttentionDecoder:
    """ Attention 메커니즘 적용한 Seq2Seq의 Decoder 클래스
    
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
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')  # Affine 계층은 맥락 벡터, LSTM의 은닉 상태 2개를 연결한 입력으로 받음
        affine_b = np.zeros(V).astype('f')
        
        # Embedding -> LSTM -> Attention -> Affine
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]
        
        # 모든 계층 파라미터 취합
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
    
    def forward(self, xs, enc_hs):
        """ Decoder의 순전파 수행(학습 시)
        
        Args:
            xs: Decoder에 입력되는 시퀀스(최초는 <eos>일 것이고, 두번째 부터는 직전 입력의 예측값이 입력으로 들어감)
            enc_hs: Encoder에서 내뱉은 모든 은닉 상태를 결합
        
        """
        h = enc_hs[:, -1]       # enc_hs의 마지막 행 벡터
        self.lstm.set_state(h)  # Decoder의 최초 입력 중 하나로 설정!
        
        out = self.embed.forward(xs)     
        dec_hs = self.lstm.forward(out)            # h_LSTM
        c = self.attention.forward(enc_hs, dec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)
        
        return score
    
    
    def backward(self, dscore):
        """ Decoder의 역전파 수행(학습 시)
        
        Args:
            dscore: TimeSoftmax-with-Loss 계층으로부터 흘러들어오는 국소적인 미분값
        
        """
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2
        
        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]   # Attention 계층으로의 보낼 기울기 / LSTM 계층으로 보낼 기울기(1)
        denc_hs, ddec_hs1 = self.attention.backward(dc) # Encoder가 보낸 hs로 보낼 기울기 / LSTM 계층으로 보낼 기울기(2)
        ddec_hs = ddec_hs0 + ddec_hs1                   # LSTM 계층 기울기 취합
        
        dout = self.lstm.backward(ddec_hs)              # LSTM -> Embedding 계층으로 보낼 기울기
        dh = self.lstm.dh                               # Encoder가 보낸 hs중 마지막 행벡터로 보낼 기울기
        
        denc_hs[:, -1] += dh                            # Attention 계층에서 보낸 hs 기울기 중에 마지막 행벡터에만 기울기 추가!
        
        self.embed.backward(dout)                       # Embedding -> input으로 역전파
        
        return denc_hs
    
    
    def generate(self, enc_hs, start_id, sample_size):
        """ Decoder에서 테스트 시 문장 생성하는 메소드
        
        Args:
            enc_hs: Encoder에서 전달해준 모든 은닉 상태를 결합
            start_id: h와 같이 Decoder에서 최초로 입력되는 데이터. 보통 <eos>와 같은 기호임
            sample_size: 생성하는 문자 개수
        
        """
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]    # Decoder의 최초 입력 중 하나로 넣을 Encoder가 보낸 hs의 마지막 행 벡터
        self.lstm.set_state(h)
        
        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))
            
            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)
            
            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)
        
        return sampled
    
    
class AttentionSeq2Seq(Seq2Seq):
    """ Attention 메커니즘 적용한 Seq2Seq 클래스
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: LSTM 계층 내의 은닉 상태 벡터 차원 수(노드 수)
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        
        # Attention 적용한 Encoder - Decoder 
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        
        # Loss
        self.loss_layer = TimeSoftmaxWithLoss()
        
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads