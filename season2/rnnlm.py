import numpy as np
from common.time_layers import *
import pickle


class Rnnlm:
    """ LSTM 계층 사용한 언어 생성 모델
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: LSTM 계층 내의 은닉 상태 벡터 차원 수(노드 수)
    
    """
    def __init__(self, vocab_size=10000, word_vec=100, hidden_size=100):
        V, D, H = vocab_size, word_vec, hidden_size
        rn = np.random.randn
        
        #===========
        # 가중치 초기화
        #===========
        # 1. 임베딩 계층
        embed_W = (rn(V, D) / 100).astype('f')
        # 2. LSTM 계층
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        # 3. Affine 계층
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        
        # 계층 생성
        self.layers = [TimeEmbedding(embed_W),
                       TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
                       TimeAffine(affine_W, affine_b)]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]  # 추후에 은닉상태, 기억 셀 유지 끊는 함수 구현 위해 캐싱해두기
        
        
        # 모든 계층의 파라미터 취합
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
            
    # 순전파 수행 후 예측값 반환
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    
    # 순전파 수행(predict 메소드) 및 Loss 계산
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    
    # 역전파 수행
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    
    def reset_state(self):
        self.lstm_layer.reset_state()
    
    
    # 학습한 파라미터 저장 및 로드 메소드
    def save_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
            
    def load_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)