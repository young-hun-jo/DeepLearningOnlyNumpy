import sys
sys.path.append('..')
import numpy as np
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from common.attention_seq2seq import AttentionSeq2Seq

# 데이터 로드
(x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 입력 시퀀스 reverse!
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 모델 하이퍼파라미터
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

# 모델
model = AttentionSeq2Seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)


acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epochs=1,
               batch_size=batch_size, max_grad=max_grad)
    
    # test 데이터로 학습시키고 있는 모델 성능 검증
    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]  # 1d-array로 변환
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                   id_to_char, verbose, is_reverse=True)
    
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

    
model.save_params()