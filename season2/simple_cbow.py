import numpy as np
from common.layers import Matmul, SoftmaxWithLoss


class SimpleCBOW:
    """ 간단한 1층 CBOW 신경망 모델
    
    Args:
        vocab_size: 말뭉치 속 unique한 단어 개수
        hidden_size: 1층 은닉층 속 뉴런 개수
    
    """
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        
        # 파라미터 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')  # 'f'로 하면 32비트 부동소수점으로 변환
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # 계층 생성 -> 현재 고려하는 맥락이 2개이기 때문에 W_in 파라미터를 사용하는 MatMul 계층도 2개!
        self.in_layer1 = Matmul(W_in)
        self.in_layer2 = Matmul(W_in)
        self.out_layer = Matmul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        
        # 모든 파라미터 모으기
        self.params, self.grads = [], []
        layers = [self.in_layer1, self.in_layer2, self.out_layer, self.loss_layer]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 우리가 얻고자 하는 단어의 분산 표현
        self.word_vecs = W_in
        
    
    def forward(self, contexts, target):
        # 맥락 데이터(contexts) 전체 중에서 모든 0번째 맥락, 1번째 맥락 각각 가져오기
        h0 = self.in_layer1.forward(contexts[:, 0, :])  # contexts[:, 0] 으로 해도 무방
        h1 = self.in_layer2.forward(contexts[:, 1, :])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        
        return loss
    
    
    def backward(self, dout=1):
        d_score = self.loss_layer.backward(dout)
        d_h = self.out_layer.backward(d_score)
        d_h *= 0.5
        self.in_layer1.backward(d_h)
        self.in_layer2.backward(d_h)
        
        return None