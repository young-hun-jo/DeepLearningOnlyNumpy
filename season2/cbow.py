import numpy as np
from common.layers import Embedding
from common.negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    """ 병목현상을 Embedding 계층과 Negative Sampling으로 해결한 CBOW
    
    Args:
        vocab_size: 말뭉치 내 unique한 단어 개수(Vocabulary size)
        hidden_size: 은닉층 노드 개수
        window_size: 입력시킬 맥락 윈도우 사이즈(만약 1이라면, 좌우 맥락 1개씩 총 2개를 고려)
        corpus: 말뭉치 내 단어별로 고유 ID가 붙은 리스트
        sample_size: 네거티브 샘플링 수행 시 샘플링할 부정적인 경우 개수
    
    """
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        
        # 파라미터 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # [입력 ~ 은닉층] 계층 생성
        self.in_layers = []
        for i in range(window_size * 2):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        # [은닉층 ~ 손실함수] 계층 생성
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        
        # 모든 계층의 파라미터, 기울기 담는 변수 취합
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 단어의 벡터를 저장
        self.word_vecs = W_in
        
    
    def forward(self, contexts, target):
        """ 순전파 수행
        
        Args:
            contexts: 맥락 데이터
            target: 타깃 데이터
        """
        # [입력 ~ 은닉층] 계층 순전파 수행: 여러개의 입력층(맥락)에서 하나의 은닉층으로!
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)      # 여러개의 입력층에서 왔기 때문에 은닉층 노드값 입력층 개수만큼 스케일링(평균 집계)
        
        # [은닉층 ~ 손실함수] 계층 순전파 수행
        loss = self.ns_loss.forward(h, target)
        
        return loss
    
    
    def backward(self, dout=1):
        # [은닉층 ~ 손실함수] 역전파 수행
        dh = self.ns_loss.backward(dout)
        dh *= 1 / len(self.in_layers)     # 역전파 시, 은닉층에서 다시 여러개의 입력층으로 나누어지기 때문에!
        
        # [입력 ~ 은닉층] 역전파 수행
        for layer in self.in_layers:
            layer.backward(dh)
            
        return None