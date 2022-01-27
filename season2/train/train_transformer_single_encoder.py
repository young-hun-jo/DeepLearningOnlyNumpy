import sys
sys.path.append('..')
import numpy as np
from dataset import sequence
from common.transformer import TransformerEncoder

(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 3개의 배치 데이터만 테스트
xs = x_train[:3]
_, T = xs.shape
V = len(char_to_id)
D = 512
d = 64
h = 3  # Self-Attention 개수(Multi 개수)
F1 = 2048
F2 = 512

encoder = TransformerEncoder(V, D, d, T, h, F1, F2)
enc_hs = encoder.forward(xs)
print('enc_hs shape:', enc_hs.shape)

enc_dhs = np.ones_like(enc_hs)
grads_query_value = encoder.backward(enc_dhs)
print(len(grads_query_value))