from common.time_layers import TimeEmbedding, TimeAffine
from common.functions import softmax
from common.layers import Affine, Relu
import numpy as np


class TimeKeyLayer:
    """ input -> Embedding -> Key Layer 순전파/역전파 클래스(T길이)
    
    """
    def __init__(self, V, D, d):
        nr = np.random.rand
        W_embed = nr(V, D)
        W_key = nr(D, d)
        b_key = np.zeros(d)
        
        self.embed_layer = TimeEmbedding(W_embed)
        self.key_layer = TimeAffine(W_key, b_key)
        self.layers = [self.embed_layer, self.key_layer]
        # 파라미터 취합
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.embed_out = None
            
    def forward(self, xs):
        # Embedding layer
        xs = self.embed_layer.forward(xs)
        self.embed_out = xs
        # Key layer
        xs = self.key_layer.forward(xs)
        return xs
    
    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


class TimeQueryLayer:
    """ input -> query layer 순전파/역전파 클래스
    
    """
    def __init__(self, V, d):
        nr = np.random.rand
        W_query = nr(V, d)
        b_query = np.zeros(d)
        
        self.layers = TimeAffine(W_query, b_query)
        self.params, self.grads = self.layers.params, self.layers.grads
        
        self.cache = V
        
    def forward(self, xs):
        V = self.cache
        if xs.ndim == 2:
            xs = np.eye(V)[xs]
        xs = self.layers.forward(xs)
        return xs
    
    def backward(self, dout):
        dout = self.layers.backward(dout)
        return dout

    
class TimeValueLayer:
    def __init__(self, V, d):
        nr = np.random.rand
        W_value = nr(V, d)
        b_value = np.zeros(d)
        
        self.layers = TimeAffine(W_value, b_value)
        self.params, self.grads = self.layers.params, self.layers.grads
        
        self.V = V
        
    def forward(self, xs):
        if xs.ndim == 2:
            xs = np.eye(self.V)[xs]
        xs = self.layers.forward(xs)
        return xs
    
    def backward(self, dout):
        dout = self.layers.backward(dout)
        return dout
    

# Key, Query를 입력으로 받는 Compatibility Function
class CompatibilityFunction:
    def __init__(self, V, D, d):
        # T길이의 Key, Query layer 정의
        self.key_layer = TimeKeyLayer(V, D, d)
        self.query_layer = TimeQueryLayer(V, d)
        
        # 파라미터 취합(Key 파라미터, Query 파라미터 순으로 저장됨)
        self.params, self.grads = [], []
        self.params += self.key_layer.params + self.query_layer.params
        self.grads += self.key_layer.grads + self.query_layer.grads
        
        self.d = d
        self.cache = None
        self.embed_out = None
        
    def forward(self, xs):
        out_query = self.query_layer.forward(xs)
        out_query_T = np.transpose(out_query, (0, 2, 1))
        
        out_key = self.key_layer.forward(xs)
        out_key_T = np.transpose(out_key, (0, 2, 1))
        
        out = np.matmul(out_query, out_key_T) / np.sqrt(self.d)
        score = softmax(out)
        
        self.cache = out_key, out_query_T, out, score
        self.embed_out = self.key_layer.embed_out
        
        return score
    
    def backward(self, dscore):
        out_key, out_query_T, out, score = self.cache
        # Softmax 역전파
        dx = score * dscore
        sumdx = np.sum(dx, axis=2, keepdims=True)
        dx -= score * sumdx
        # Scaling factor 역전파
        dx *= -(out ** 2)
        # 행렬 곱 역전파
        dkey = np.matmul(out_query_T, dx)
        dquery = np.matmul(dx, out_key)
        
        # Key, Query layer 역전파
        self.key_layer.backward(dkey)  # 마지막 Embedding 레이어 역전파는 None을 반환
        dquery = self.query_layer.backward(dquery)
        return dquery
    

# Compatibility Function이 구한 Weight와 Value를 Weighted Sum하는 계층
class WeightedValue:
    def __init__(self, V, d):
        self.value_layer = TimeValueLayer(V, d)
        self.params, self.grads = self.value_layer.params, self.value_layer.grads
        
    def forward(self, xs, score):
        out_value = self.value_layer.forward(xs)
        
        B, T, d = out_value.shape
        Z = np.zeros((B, T, d))
        for b in range(B):
            for t1 in range(T):
                z = np.zeros((1, d))
                for t2 in range(T):
                    v = out_value[b, t2, :]
                    w = score[b, t1, t2]
                    z += (v * w)
                Z[b, t1] = z
        
        out_value_T = np.transpose(out_value, (0, 2, 1))
        self.cache = score, out_value, out_value_T
        return Z
    
    def backward(self, dZ):
        """
        Args:
            dZ: Z에 대한 기울기 - shape:(10, 29, 64) 
        """
        score, out_value, out_value_T = self.cache
        # Compatibility Function으로 들어갈 역전파 기울기
        dscore = np.matmul(dZ, out_value_T)
        # value layer로 들어갈 역전파 기울기
        dvalue = np.zeros_like(out_value)
        
        B, T, d = out_value.shape
        for b in range(B):
            for t1 in range(T):
                v = dvalue[b, t1, :]
                for t2 in range(T):
                    w = score[b, t2, t1]
                    v += (v * w)
                dvalue[b, t1, :] = v
        # value layer에 파라미터 하나 있음 ^^ 빼먹지 말기!
        dvalue = self.value_layer.backward(dvalue)
        return dscore, dvalue
    

# 위 구현한 클래스들 기반으로 Self-Attention 계층 구현
class SelfAttention:
    def __init__(self, V, D, d):
        # Compatibility Function
        self.comp_func = CompatibilityFunction(V, D, d)
        self.weighted_value = WeightedValue(V, d)
        self.layers = [self.comp_func, self.weighted_value]
        
        # 파라미터 취합
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
        self.embed_out = None
            
    def forward(self, xs):
        score = self.comp_func.forward(xs)
        z = self.weighted_value.forward(xs, score)
        
        self.embed_out = self.comp_func.embed_out
        return z
    
    def backward(self, dZ):
        dscore, dvalue = self.layers[1].backward(dZ)
        dquery = self.layers[0].backward(dscore)
        
        return dvalue, dquery
    
    
# 3개의 SelfAttention 계층 -> BatchNormalization 계층은 구현은 못함..
class MultiSelfAttention:
    def __init__(self, V, D, d, h, T):
        """
        Args:
            h: 몇개의 Self-Attention 계층을 만들 것인지
            T: 입력 시퀀스 길이 for z값들 concat 후, 입력 시퀀스 길이(T)만큼의 Linear layer 추가해주어야 하기 떄문
        """
        # 여러개의 Self-Attention 계층 생성
        self.layers = [SelfAttention(V, D, d) for _ in range(h)]
        for t in range(T):
            W_linear = np.random.rand(d, D)   # D(512)는 논문에서 제안한 차원 수임
            b_linear = np.zeros(D)
            linear_layer = Affine(W_linear, b_linear)
            self.layers.append(linear_layer)
        
        # 여러개의 Self-Attention 파라미터 취합
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
        self.cache = V, D, d, h, T
        self.embed_out = None
        
    def forward(self, xs):
        V, D, d, h, T = self.cache
        
        B, _ = xs.shape
        z = np.zeros((B, T, d))
        # Multi Self-Attention 계층 순전파 후 concat
        for layer in self.layers[:h]:
            z += layer.forward(xs) # concat을 sum으로!

        # 각각 Linear 연산(배치부터 loop)
        out = np.zeros((B, T, D))
        for b in range(B):
            for t, layer in enumerate(self.layers[h:]):
                single_z = z[b, t, :].reshape(1, d)
                o = layer.forward(single_z)
                out[b, t, :] = o
        
        # Embedding output을 Skip-connection
        # 하나의 Self-Attention 마다 1개의 Embedding Layer가 있기 때문에 Multi 개수만큼 평균값을 취한 후 Skip Connection!
        self.embed_out = np.zeros((B, T, D))
        for layer in self.layers[:h]:
            self.embed_out += layer.embed_out
        out += (self.embed_out / h)

        return out
    
    def backward(self, dout):
        """
        Args:
            dout: out의 기울기(shape: (10, 29, 512))
        """
        _, _, d, h, _ = self.cache
        B, T, D = dout.shape
        
        # 각각 Linear 연산에 대한 역전파(배치부터 거꾸로 loop)
        dZ = np.zeros((B, T, d))
        for b in range(B-1, -1, -1): # 역전파 시, 배치 순서도 순전파때와는 거꾸로 진행해야 함!
            for t, layer in enumerate(reversed(self.layers[h:])):
                single_dout = dout[b, T-1-t, :].reshape(1, D)
                dz = layer.backward(single_dout)
                dZ[b, t, :] = dz
        
        # Self-Attention 계층 역전파 -> dvalue, dquery, dkey 까지(초기 입력 단계까지) 역전파수행됨!
        grads_query_value = []
        for layer in reversed(self.layers[:h]):
            dvalue, dquery = layer.backward(dZ)
            grads_query_value += [dvalue, dquery]
            
        return grads_query_value  
    
    
# FFN(Feed Forward Neural Network) 계층 구현
class FeedForward:
    def __init__(self, T, D, F1=2048, F2=512):
        """
        Args:
            T: 입력 시퀀스 길이
            D: Multi-head Self-Attention 계층에서 나온 출력 벡터의 차원 수
            F1: FFN에서 첫번째 선형 회귀식 때의 차원 수
            F2: FFN에서 두번째 선형 회귀식 때의 차원 수
        
        """
        self.out = None
        nr = np.random.rand
        # 각 입력 시퀀스 마다 FFN 파라미터 달리해주어야 함!(단, 각 배치의 위치는 동일)
        self.layers = []
        for t in range(T):
            W_one = nr(D, F1)
            b_one = np.zeros(F1)
            W_two = nr(F1, F2)
            b_two = np.zeros(F2)
            
            linear1 = Affine(W_one, b_one)
            relu = Relu()
            linear2 = Affine(W_two, b_two)
            self.layers.append([linear1, relu, linear2])
        
        # 파라미터 취합
        self.params, self.grads = [], []
        for t in range(T):
            for i in range(len(self.layers[0])):
                self.params += self.layers[t][i].params
                self.grads += self.layers[t][i].grads
                
        self.cache = T, D, F1, F2
        
    def forward(self, out):
        """
        Args:
            out: Multi-head Self-Attention 계층에서 내뱉은 출력 벡터
        
        """
        B, T, D = out.shape
        self.out = out
        
        ffn_out = np.zeros_like(out)
        for b in range(B):
            for t in range(T):
                single_out = out[b, t, :].reshape(1, D)
                for layer in self.layers[t]:
                    single_out = layer.forward(single_out)
                ffn_out[b, t, :] = single_out
        
        # Skip Connection
        ffn_out += out
        return ffn_out
        
    def backward(self, ffn_dout):
        """
        Args:
            ffn_dout: ffn_out의 기울기(shape: (B, T, F2))
        
        """
        T, D, F1, F2 = self.cache
        B, _, _ = self.out.shape
        dout = np.zeros_like(self.out)
        
        for b in range(B-1, -1, -1):
            for t in range(T-1, -1, -1):
                for layers in reversed(self.layers):
                    single_ffn_dout = ffn_dout[b, t, :].reshape(1, F2)
                    for layer in layers[::-1]:
                        single_ffn_dout = layer.backward(single_ffn_dout)
                    dout[b, t, :] = single_ffn_dout
        return dout