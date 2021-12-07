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
        
        # 1. 곱셈 노드의 역전파(원래는 dot 계층 역전파지만 여기선 편의를 위해 곱셈 노드로 구현) = 순전파 시 반대편 입력을 곱하면 됨!
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
            self.word_p[i] = counts[i]
            
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
        
        
    def get_negative_sample(self, target: np.array):
        """ 긍정적인 경우인 타겟 단어를 제외한 부정적인 경우 샘플링 수행
        
        Args:
            target: 특정 배치 사이즈만큼의 긍정적인 타겟 단어들이 담긴 1차원 array
        
        """
        batch_size = target.shape[0]
        
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        
        # 각 긍정적인 경우(타겟 단어) 마다 부정적인 경우 확률분포에서 추출
        for i in range(batch_size):
            prob = self.word_p.copy()
            target_idx = target[i]
            prob[target_idx] = 0  # 타겟 단어의 발생확률을 0으로 만들어서 샘플링 안되도록!
            prob /= np.sum(prob)  # 확률 총합 다시 1로 만들기
            
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, 
                                                     replace=False, p=prob)
            
        return negative_sample
    
    
    
class NegativeSamplingLoss:
    """ Negative Sampling Loss 계층 = Negative Sampling + Embedding dot + Sigmoid-with-Loss
    
    Args:
        W: 은닉층 -> 출력층 사이의 가중치
        corpus: 말뭉치 내 단어별로 고유한 ID가 붙은 리스트
        power: 단어 별 확률분포 계산 시 제곱할 수
        sample_size: 네거티브한 경우를 몇 개 만들 것인지(네거티브 샘플링 횟수)
        
    """
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)  # 네거티브 샘플링 클래스
        # 긍정적인 경우 1개 + 부정적인 경우 sample_size개수 만큼의 계층 만들기
        self.loss_layers = [SigmoidWithLoss() for _ in range(self.sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(self.sample_size+1)]
        
        self.params, self.grads = [], []
        # 갱신된 파라미터, 기울기가 담길 EmbeddingDot계층의 변수들 모으기 
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
            
            
    def forward(self, h, target: np.array):
        """ 긍정, 부정인 경우에 대해 각각 순전파 수행
        
        Args:
            h: 은닉층의 노드
            target: 긍정적인 경우에 해당하는 정답 즉, '진짜' 정답
        
        """
        batch_size = target.shape[0]
        # 부정적인 경우 샘플링 -> 부정적인 경우에 해당하는 단어ID가 들어있음
        negative_sample = self.sampler.get_negative_sample(target)
        
        # 1.긍정적인 경우 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        positive_label = np.ones(batch_size, dtype=np.int32) # 모두 긍정이므로 1(정답)을 원소로 하는 레이블 생성
        loss = self.loss_layers[0].forward(score, positive_label)
        
        # 2.부정적인 경우 각각 순전파 수행
        negative_label = np.zeros(batch_size, dtype=np.int32) # 부정적인 경우의 단어ID는 레이블을 0(정답X)으로 생성
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]  # 각 부정적인 경우마다 샘플링된 타겟 하나씩 모두 가져오기
            score = self.embed_dot_layers[i+1].forward(h, negative_target)
            loss += self.loss_layers[i+1].forward(score, negative_label)  # 긍정, 부정 경우 loss 총 합산!
            
        return loss
    
    
    def backward(self, dout=1):
        # 순전파 시 긍정, 부정 경우마다 분기(Repeat)되어서 EmbeddingDot & Sigmoid-with-loss 계층으로 퍼져나갔음
        # 분기 노드 역전파 시에는 합쳐지기 때문에 흘러들어오는 국소 미분값들 모두 sum 하면 됨!
        
        dh = 0
        for layer_1, layer_2 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = layer_1.backward(dout)
            dh += layer_2.backward(dscore)
        
        return dh