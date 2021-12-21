import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import Adam
from simple_rnnlm import SimpleRnnlm
from dataset import ptb

# 하이퍼파라미터 설정
batch_size = 10
wordvec_size = 100  # 임베딩 계층 내 벡터 차원 수
hidden_size = 100   # RNN 계층 내 벡터 차원 수
time_size = 5       # Truncated BPTT 수행 시 잘라내는 작은 신경망 길이
lr = 0.01
max_epoch = 100


# 말뭉치 데이터 준비
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]  # 말뭉치 전체 중 1000개의 단어만 사용
vocab_size = int(max(corpus) + 1)  # unique한 단어 ID의 최댓값(단어 ID가 0부터 시작하므로 +1)

xs = corpus[:-1] 
ts = corpus[1:]
data_size = len(xs)
print('말뭉치 크기: %d, 어휘 수: %d' % (corpus_size, vocab_size))


# 학습 시 사용하는 파라미터
steps = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []


# 모델, Optimizer 생성
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = Adam(lr)

# 배치 데이터 만들기 위해 데이터 시작점 Offset 인덱스 생성
jump = data_size // batch_size
offsets = [i * jump for i in range(batch_size)]


# 전체 Epoch 학습 수행
for epoch in range(max_epoch):
    # 한 Epoch 동안 도는 step 수행
    for step in range(steps):
        # 하나의 배치 데이터 담을 빈그릇 생성
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        for t in range(time_size): # time_size 길이만큼 데이터 할당 loop
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        # 하나의 배치가 모두 담긴 후 학습 시작
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
        
    # Perplexity 평가 척도 측정
    ppl = np.exp(total_loss / loss_count)
    print('| Epoch: %d | Perplexity: %.2f' % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0