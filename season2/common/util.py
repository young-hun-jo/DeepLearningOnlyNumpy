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


# 단어 벡터들 기반으로 코사인 유사도를 활용해 비슷한 단어들 쿼리 및 출력
def most_similar(query, word_to_id: dict, id_to_word: dict, co_matrix, top=5):
    """ 쿼리한 단어와 가장 유사도가 큰 상위 단어들 출력하는 함수
    
    Args:
        query: 비교 대상 단어
        word_to_id: 단어를 key, 단어의 ID를 value로 하는 딕셔너리
        id_to_word: 단어의 ID를 key, 단어를 value로 하는 딕셔너리
        co_matrix: 단어의 동시발생 행렬
        top: 상위 몇 개 단어 출력할 것인지
    
    """
    if query not in word_to_id:
        print(f'{query} 라는 단어는 Vocabulary에 존재하지 않습니다')
        return
    
    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = co_matrix[query_id]
    
    # 동시 발생 행렬의 벡터를 활용해 단어들 간 코사인 유사도 계산
    vocab_size = len(word_to_id)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(query_vec, co_matrix[i])
    
    # 코사인 유사도 기준으로 내림차순 정렬
    count = 0
    for j in (-1 * similarity).argsort():
        if id_to_word[j] == query:    # 쿼리 자기 자신 단어와 비교는 pass
            continue
        print('%s 와의 유사도 : %s' % (id_to_word[j], similarity[j]))
        
        count += 1
        if count >= top:
            return


# 동시발생 행렬 기반으로 하는 PPMI 계산
def ppmi(C: np.array, verbose=False, eps=1e-8):
    """ 동시발생 행렬 기반 양의 상호정보량(PPMI) 계산
    
    Args:
        C: 동시발생 행렬
    
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)            # 말뭉치 내의 전체 단어 개수
    S = np.sum(C, axis=0)    # 말뭉치의 unique한 단어들 발생 횟수(이 때, 단어란, 동시발생 행렬 만들 때 window_size에 따라 어절일 수도 있음)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    # 말뭉치 단어 하나씩 PPMI 계산
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)
            M[i, j] = max(0, pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print(f'{100*cnt/total :.1f}% 완료')
    return M
