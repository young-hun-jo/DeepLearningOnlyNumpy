# Peeky Decoder 클래스 구현
import numpy as np
from common.time_layers import *
from common.seq2seq import Seq2Seq, Encoder


class PeekyDecoder:
    """ 기존 Deecoder 부분을 Peeky Decoder로 변경한 클래스
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: LSTM 계층 내의 은닉 상태 벡터 차원 수(노드 수)
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # Embedding -> LSTM -> Affine
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H + D, 4*H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)  # Decoder이기 때문에 은닉 상태 유지!
        self.affine = TimeAffine(affine_W, affine_b)
        
        # 모든 계층 파라미터 취합
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None
        
        
    def forward(self, xs, h):
        """ Peeky Decoder 순전파 수행(학습 시))
        
        Args:
            xs: (batch_size, 전체 시계열 길이(T)) Decoder가 내뱉는 도착어. 학습 시이기 때문에 정답 시퀀스나 마찬가지
            h: (batch_size, 은닉상태 벡터 차원 수) Encoder가 전달하는 최종 은닉상태 벡터
        """
        N, T = xs.shape
        N, H = h.shape
        
        self.lstm.set_state(h) # Encoder가 전달하는 은닉상태 벡터 입력!
        
        # Embedding 순전파 후 concatenate
        out = self.embed.forward(xs)
        # np.repeat(array, cnt): array를 cnt번 복제
        hs = np.repeat(h, T, axis=0).reshape(N, T, H) # Encoder가 전달한 은닉상태(h)를 T길이 만큼 복제(다른 계층에도 전달하기 위해)
        out = np.concatenate((hs, out), axis=2)  # 마지막 차원(axis=0,axis=1,axis=2) 방향으로 concat
        
        # LSTM 순전파 후 concatenate
        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)
        
        # Affine 계층 순전파
        score = self.affine.forward(out)
        self.cache = H
        return score
    
    
    def backward(self, dscore):
        H = self.cache
        
        # Affine -> concatenate 역전파 수행
        dout = self.affine.backward(dscore)
        dhs0, dout = dout[:, :, :H], dout[:, :, H:]  # dhs0: Encoder에서 전달받은 h에 대한 기울기, dout: LSTM 계층으로 흘려보낼 기울기
        # LSTM -> concatenate 역전파 수행
        dout = self.lstm.backward(dout)
        dhs1, dembed = dout[:, :, :H], dout[:, :, H:]
        self.embed.backward(dembed)
        
        # 분기된 Encoder h값들에 대한 기울기 역전파하니까 sum!
        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)  # self.lstm.dh: TimeLSTM 계층에서 역전파 시의 마지막 dh가 캐싱되어 있도로 구현했었음!
        
        return dh
    
    
    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        
        self.lstm.set_state(h)         # Encoder에서 전달한 은닉상태 입력!
        
        H = h.shape[1]                 # 은닉상태 벡터 차원 수
        peeky_h = h.reshape(1, 1, H)  # 이것으로 문장 생성(Test) 시, 다른 LSTM, Affine 계층들이 Encoder가 전달하는 정보 엿봄!
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            
            # Embediing & concatenate
            out = self.embed.forward(x)
            out = np.concatenate((peeky_h, out), axis=2)
            # LSTM & concatenate
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            # Affine
            score = self.affine.forward(out)
            
            # argmax
            char_id = np.argmax(score.flatten())
            sampled.append(char_id)
        
        return sampled
    
    

# Peeky Decoder 기반으로 하는 PeekySeq2Seq 클래스 구현
class PeekySeq2Seq(Seq2Seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        
        # Encoder - Decoder - Softmax 계층 설정
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.loss_layer = TimeSoftmaxWithLoss()
        
        # 모든 계층 파라미터 취합
        self.params, self.grads = [], []
        for layer in (self.encoder, self.decoder):  # loss 계층에는 파라미터 X
            self.params += layer.params
            self.grads += layer.grads
