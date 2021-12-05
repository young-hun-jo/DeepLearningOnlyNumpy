# Skip-gram으로 학습시켜보기
import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_contexts_target, convert_one_hot
from common.optimizer import Adam
from common.trainer import Trainer
from simple_skipgram import SimpleSkipGram


# 1. 말뭉치 전처리
window_size = 1

text = 'You say goodbye and I say Hello.'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size)

vocab_size = len(word_to_id)
contexts_ohe = convert_one_hot(contexts, vocab_size)
target_ohe = convert_one_hot(target, vocab_size)

# 2. 하이퍼파라미터 설정
hidden_size = 5
batch_size = 3
epochs = 1000

# 3. Skip-gram 모델
model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 4. 학습
trainer.fit(x=target_ohe, 
            t=contexts_ohe, max_epochs=epochs, batch_size=batch_size)
