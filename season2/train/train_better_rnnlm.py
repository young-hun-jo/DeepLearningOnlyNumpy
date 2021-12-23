import sys
sys.path.append('..')
from common.optimizer import Adam
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from better_rnnlm import BetterRnnlm
import numpy as np


# 하이퍼파라미터
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35       # Truncated BPTT 시, 잘라내는 작은 신경망 단위
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

# 데이터 로드
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_val, _, _ = ptb.load_data('val')
corpus_test, _, _ = ptb.load_data('test')

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# model, optimizer
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = Adam(lr)
trainer = RnnlmTrainer(model, optimizer)

# 학습
best_ppl = float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size,
               time_size=time_size, max_grad=max_grad)
    
    # 검증 데이터로 평가하기 위해 학습 데이터로 얻은 은닉 상태벡터, 기억 셀 삭제
    model.reset_state()
    # 모델을 검증 데이터로 평가. eval_perplexity 함수 내부에 train_flg가 False로 되어있음. 이는 곧 Valid 또는 Test로 평가한다는 셈!
    ppl = eval_perplexity(model, corpus_val)  
    print('Validation Perplexity:', ppl)
    
    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    # learning rate scheduler 적용
    else:
        lr /= 4.0
        optimizer.lr = lr
    
    model.reset_state()
    print('-'*50)

    
# 테스트 데이터로 평가
model.reset_state()
test_ppl = eval_perplexity(model, corpus_test)
print('Test Perplexity:', test_ppl)