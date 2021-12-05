# Skip-gram
import numpy as np
from common.layers import Matmul, SoftmaxWithLoss


class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        
        # 파라미터 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # 계층 생성
        self.in_layer = Matmul(W_in)
        self.out_layer = Matmul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()
        
        # 모든 계층 파라미터 모으기
        self.params, self.grads = [], []
        layers = [self.in_layer, self.out_layer]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        self.word_vecs = W_in
        
        
    def forward(self, contexts, target):
        # target으로 contexts를 예측
        h = self.in_layer.forward(target)
        score = self.out_layer.forward(h)
        loss_1 = self.loss_layer1.forward(score, contexts[:, 0]) # contexts[:, 0] 으로 해도 무방
        loss_2 = self.loss_layer2.forward(score, contexts[:, 1])
        loss = loss_1 + loss_2
        return loss
    
    
    def backward(self, dout=1):
        dloss_1 = self.loss_layer1.backward(dout)
        dloss_2 = self.loss_layer2.backward(dout)
        dscore = dloss_1 + dloss_2  # 덧셈 노드 역전파 시 그대로 흘러보냄
        dh = self.out_layer.backward(dscore)
        self.in_layer.backward(dh)
        return None