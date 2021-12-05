import sys
sys.path.append('..')
from common.util import preprocess, create_contexts_target, convert_one_hot
from common.optimizer import Adam
from common.trainer import Trainer
from simple_cbow import SimpleCBOW

# 1. 하이퍼파라미터 설정
window_size = 1   # 맥락 고려 개수(1이면 2개의 맥락을 고려, 2이면 4개의 맥락을 고려)
hidden_size = 3
batch_size = 4
epochs = 1000

# 2. 말뭉치 전처리
text = 'You say goodbye and I say Hello.'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size)
                                          
vocab_size = len(word_to_id)
contexts_ohe = convert_one_hot(contexts, vocab_size)
target_ohe = convert_one_hot(target, vocab_size)

# 3. 신경망 모델(Trainer 클래스)
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 4. 학습
trainer.fit(x=contexts_ohe, t=target_ohe, max_epochs=epochs, batch_size=batch_size)
