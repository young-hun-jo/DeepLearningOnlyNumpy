import numpy as np
from common.time_layers import *


class SimpleRnnlm:
    """ 간단한 RNN을 사용한 언어 생성 모델
    
    Args:
        vocab_size: 주어진 말뭉치 내 unique한 단어 개수(Vocabulary size)
        wordvec_size: One-hot으로 되어있는 Sparse 입력 단어를 몇 차원 임베딩 Dense 벡터로 줄일 것인지
        hidden_size: RNN 계층 내의 벡터 차원 수(노드 수)
    
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        np_randn = np.random.randn
        
        #===========
        # 가중치 초기화
        #===========
        # 1. 임베딩 계층
        embed_W = (np_randn(V, D) / 100).astype('f')
        # 2. RNN 계층(은닉상태 벡터는 (N, H)어야 함(N은 배치 사이즈) -> 입력층 노드 개수를 활용하는 Xaiver 초기화 기법 사용
        rnn_Wx = (np_randn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (np_randn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        # 3. Affine 계층 -> 입력층 노드 개수를 활용하는 Xaiver 초기화 기법 사용
        affine_W = (np_randn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        
        # 계층 생성
        self.layers = [TimeEmbedding(embed_W),
                       TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
                       TimeAffine(affine_W, affine_b)]
        # Softmax-with-Loss 계층
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]  # 추후에 은닉상태 끊는 함수 기능 구현 위해 캐싱해두기
        
        
        # 모든 계층의 파라미터 취합
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    
    def forward(self, xs, ts):
        """ 순전파 수행
        
        Args:
            xs: One-hot 형태로 되어 있는 입력 단어 벡터
            ts: 정답 레이블
        
        """
        # 임베딩, RNN, Affine 계층 순전파 수행
        for layer in self.layers:
            xs = layer.forward(xs)
        
        # Softmax-with-Loss 계층 순전파 수행
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    
    # RNN 계층의 은닉 상태 전달하는 것 끊는 함수
    def reset_state(self):
        self.rnn_layer.reset_state()