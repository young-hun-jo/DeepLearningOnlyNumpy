import numpy as np


# Gradient Clipping
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


# 텍스트 단어 단위로 분할 후, 텍스트 말뭉치, Word-ID 딕셔너리 반환
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


# 단어의 동시발생 행렬
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    # 동시발생 행렬 초기화 -> 사이즈는 말뭉치의 unique한 단어들 개수로!
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        for size in range(1, window_size+1):
            left_idx = idx - size
            right_idx = idx + size
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix


# 코사인 유사도 계산
def cos_similarity(x: np.array, y: np.array, eps=1e-8):
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    cos = np.matmul(nx, ny)
    
    return cos
