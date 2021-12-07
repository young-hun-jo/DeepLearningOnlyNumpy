import sys
sys.path.append('..')
# CBOW 모델 최종 PTB 데이터 학습 코드
import numpy as np
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.util import create_contexts_target
from dataset import ptb

# 하이퍼파라미터
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 데이터 로드
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

# 맥락, 타깃 데이터로 전처리
contexts, target = create_contexts_target(corpus, window_size)

# CBOW 모델 생성
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 학습 시작
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()


# 학습 종료 후 단어분산표현 저장
word_vecs = model.word_vecs

params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'

with open('Users/younghun/Desktop'+pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)