import numpy as np
from common.time_layers import *


class BetterRnnlm:
    """ RNN 언어 모델 성능 개선하는 기법 3가지 적용(계층 다층화/ Dropout / 가중치 공유)
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: LSTM 계층 내의 은닉 상태 벡터 차원 수(노드 수)
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # 1. 임베딩 계층
        embed_W = (rn(V, D).astype('f'))
        # 2. LSTM 다층화 계층
        lstm_Wx1 = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4*H).astype('f')
        lstm_Wx2 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4*H).astype('f')
        # 3. Affine 계층
        affine_b = np.zeros(V).astype('f')
        
        # 계층 정의
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)  # 임베딩 계층 파라미터와 가중치 공유(임베딩 계층 파라미터의 Transpose)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.dropout_layers = [self.layers[1], self.layers[3], self.layers[5]]
        
        # 모든 계층 파라미터 취합
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    
    def predict(self, xs, train_flg=False):
        # 학습 시에만 Dropout 적용하도록 설정
        for layer in self.dropout_layers:
            layer.train_flg = train_flg
        # 순전파 수행
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    
    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)        # 순전파 수행
        loss = self.loss_layer.forward(score, ts)  # loss 계산
        return loss
    
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)   # Loss 계층 역전파
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    # 학습한 파라미터 저장 및 로드 메소드
    def save_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
            
    def load_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)