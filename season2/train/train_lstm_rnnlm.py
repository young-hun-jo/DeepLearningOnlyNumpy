import sys
sys.path.append('..')
from common.optimizer import Adam
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from rnnlm import Rnnlm
from dataset import ptb

# 하이퍼파라미터
batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 학습 데이터 로드 및 전처리
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)

xs = corpus[:-1]
ts = corpus[1:]

# 모델 생성
model = Rnnlm(vocab_size=vocab_size, word_vec=wordvec_size, hidden_size=hidden_size)
optimizer = Adam()
trainer = RnnlmTrainer(model, optimizer)

# 모델 학습
trainer.fit(x=xs, t=ts, max_epoch=max_epoch, batch_size=batch_size,
            time_size=time_size, max_grad=max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# 모델 테스트 데이터로 평가
model.reset_state()  # 학습 데이터로 학습한 뒤 얻어진 은닉상태벡터, 기억셀 삭제해야 함! 왜냐하면 학습 데이터, 테스트 데이터는 서로 다른 시퀀스!
ppl_test = eval_perplexity(model, corpus_test)
print('테스트 perplexity:', ppl_test)