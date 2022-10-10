# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.functions as F
from dezero.datasets import SinCurve
from dezero.dataloaders import SeqDataLoader
from dezero.models import BetterRNN
from dezero.optimizers import Adam

max_epoch = 100
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = SinCurve(train=True)
train_loader = SeqDataLoader(train_set, batch_size)
seqlen = len(train_set)

model = BetterRNN(hidden_size, out_size=1)
optimizer = Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()   # reset state per each epoch
    loss, bptt_cnt = 0, 0

    for x, t in train_loader:
        # predict and get loss
        y = model(x)
        loss += F.mean_squared_error(y, t)
        bptt_cnt += 1

        # Truncated BPTT
        if bptt_cnt % bptt_length == 0 or bptt_cnt == seqlen:
            # 1. 현재 상태 기울기 초기화
            model.clear_grads()
            # 2. 역전파 수행
            loss.backward(use_heap=True)
            # 3. 다음 BPTT 대비 위해 앞단 Variable의 창조자 삭제
            loss.unchain_backward()
            # 4. 기울기 갱신
            optimizer.update()
    avg_loss = float(loss.data) / bptt_cnt
    print('Epoch:', epoch, '-> Loss:', avg_loss)
