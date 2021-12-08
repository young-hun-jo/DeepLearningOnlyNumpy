# Skip-gram 
from common.layers import *
from common.negative_sampling_layer import NegativeSamplingLoss


class SkipGram:
    """ 병목현상을 Embedding 계층과 Negative Sampling으로 해결한 Skip-gram
    
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
        W_out = 0.01 * np.random.randn(V, H).astype('f')
        
        # 계층 생성 -> 입력층은 1개(타깃), 출력층은 여러개(맥락)
        self.in_layer = Embedding(W_in)
        self.loss_layers = []
        for i in range(2 * window_size):
            layer = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
            self.loss_layers.append(layer)
            
        # 모든 계층 파라미터, 기울기 담는 변수 취합
        layers = [self.in_layer] + self.loss_layers
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 단어 분산 표현 저장
        self.word_vecs = W_in
        
        
    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        
        loss = 0
        for i, layer in enumerate(self.loss_layers):
            loss += layer.forward(h, contexts[:, i])
        return loss
    
    
    def backward(self, dout=1):
        dh = 0
        # 순전파 시 출력층(맥락) 여러개로 분기되었으니 역전파 시 sum
        for i, layer in self.loss_layers:
            dh += layer.backward(dout)
        self.in_layer.backward(dh)
        return None