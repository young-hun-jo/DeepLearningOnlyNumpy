# Embedding Dot 계층 구현
import sys
sys.path.append('..')
import numpy as np
from common.layers import Embedding, SigmoidWithLoss


class EmbeddingDot:
    def __init__(self, W):
        # 임베딩 계층 생성
        self.embed = Embedding(W)
        # 임베딩 계층의 파라미터 가져오기
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
        
    def forward(self, h, idx):
        # W_out 파라미터에서 특정 idx에 해당하는 행 벡터만 추출
        target_W = self.embed.forward(idx)
        
        # dot을 안하고 곱(*)하는 이유: 어차피 하나의 행 벡터와 열 벡터를 내적하는 것이기 때문에 곱하는 것과 동일. 
        # 곱(*) 사용하면 행 벡터, 열벡터 shape 맞춰줄 필요도 없이 편함
        out = np.sum(target_W * h, axis=1) 
        
        self.cache = (h, target_W)  # 역전파 시 해당 값들을 사용해야 하므로 캐싱해두기
        
        return out
    
    
    def backward(self, dout):
        h, target_W = self.cache
        
        # 1개의 값만 있는 dout들이 배치 사이즈만큼 있음 -> 행렬 곱 수행하기 위해
        dout = dout.reshape(dout.shape[0], 1)
        
        # 1. dot 계층 역전파 - 곱셈 노드의 역전파 = 순전파 시 반대편 입력을 곱하면 됨!
        dtarget_W = dout * h
        dh = dout * target_W
        
        # 2. 임베딩 계층 역전파
        self.embed.backward(dtarget_W)
        
        return dh