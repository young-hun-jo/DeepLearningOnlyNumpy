# Embedding Dot 계층 구현
import sys
sys.path.append('..')
import numpy as np
from collections import Counter
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
    
    
    
# Unigram Negatvie Sampling
class UnigramSampler:
    """ Unigram 기반 네거티브 샘플링 
    
    Args:
        corpus: 말뭉치 내 각 단어의 고유 ID가 담긴 리스트
        power: 단어별 확률분포 계산 시 제곱할 값
        sample_size: 네거티브한 경우를 몇 개 샘플링할지
    
    """
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None    # 말뭉치 내 각 단어별 발생 확률분포
        
        # 말뭉치 내 각 단어별 출현 빈도 구하기
        counts = Counter()
        for word_id in corpus:
            counts[word_id] += 1
        
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        
        # 단어별 출현 빈도 기반으로 단어별 확률 분포 계산
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_[i] = counts[i]
            
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(word_p)
        
        
    def get_negative_sample(self, target: np.array):
        """ 긍정적인 경우인 타겟 단어를 제외한 부정적인 경우 샘플링 수행
        
        Args:
            target: 특정 배치 사이즈만큼의 긍정적인 타겟 단어들이 담긴 1차원 array
        
        """
        batch_size = target.shape[0]
        
        negative_sample = np.zeros((batch_size, self.vocab_size), dtype=np.int32)
        
        # 각 긍정적인 경우(타겟 단어) 마다 부정적인 경우 확률분포에서 추출
        for i in range(batch_size):
            prob = self.word_p.copy()
            target_idx = target[i]
            prob[target_idx] = 0  # 타겟 단어의 발생확률을 0으로 만들어서 샘플링 안되도록!
            prob /= np.sum(prob)  # 확률 총합 다시 1로 만들기
            
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, 
                                                     replace=False, p=prob)
            
        return negative_sample