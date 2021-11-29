# PTB 말뭉치 단어들 간 관련셩 평가하기
from dataset import ptb
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi
import numpy as np

window_size = 2
wordvec_size = 100  # SVD로 감소시킬 차원 수 

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)  # 말뭉치 내 unique한 단어 개수

# 동시발생 행렬 계산
C = create_co_matrix(corpus, vocab_size, window_size)

# PPMI 행렬 계산
M = ppmi(C, verbose=False)

# SVD 계산
try:
    # Sklearn의 무작위 Truncated SVD 수행(속도가 빠름)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(M, n_components=wordvec_size, n_iter=5, random_state=42)
except ImportError:
    # linalg.svd -> 속도 느림
    U, S, V = np.linalg.svd(M)
    
word_vecs = U[:, :wordvec_size]

# 쿼리로 넣는 단어들과 관련있는 단어들 테스트
sample_size = 10
querys = np.random.choice(len(word_to_id), sample_size)
for query in querys:
    most_similar(id_to_word[query], word_to_id, id_to_word, word_vecs, top=5)
    print()
