from common.transformer_layer import MultiSelfAttention
from common.transformer_layer import FeedForward
import numpy as np

class TransformerEncoder:
    def __init__(self, V, D, d, T, h, F1=2048, F2=512):
        
        # Multi-head Self-Attention
        self.multi_head_self_attention = MultiSelfAttention(V, D, d, h, T)
        # FFN
        self.ffn_layer = FeedForward(T, D, F1, F2)
        self.layers = [self.multi_head_self_attention, self.ffn_layer]
        
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def backward(self, enc_dhs):
        """
        Args:
            enc_dhs: Encoder의 순전파 최종 출력 기울기
        """
        for layer in reversed(self.layers):
            enc_dhs = layer.backward(enc_dhs)
        grads_query_value = enc_dhs
        return grads_query_value