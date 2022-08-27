# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import sys
import logging
import dezero
import dezero.functions as F
from dezero import DataLoader
from dezero.models import MLP
from dezero.optimizers import AdaGrad
from dezero.datasets import Spiral

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s] %(levelname)s : %(message)s')

# hyper-parameter
max_epoch = 300
batch_size = 30
hidden_size = (10, 5)
lr = 1.0

# dataset
train_set = Spiral(train=True)
test_set = Spiral(train=False)

# dataloader
train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# build and compile model
model = MLP((*hidden_size, 3))
optimizer = AdaGrad(lr=lr).setup(model)

# train and test(validation) in each Epoch
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    # mini-batch for train
    for x, t in train_loader:
        # predict and get loss, accuracy
        y_pred = model(x)
        loss = F.softmax_cross_entropy(y_pred, t)
        acc = F.accuracy(y_pred, t)

        # clear gradients
        model.clear_grads()
        # backward
        loss.backward(use_heap=True)
        # update params
        optimizer.update()

        # verbose metric
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    if (epoch+1) % 20 == 0:
        logging.info(f"(Epoch{epoch+1})Train Loss: {sum_loss / len(train_set)}, Accuracy: {sum_acc / len(train_set)}")

    # mini-batch for test
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():   # deactivate backward
        for x, t in test_loader:
            y_pred = model(x)
            loss = F.softmax_cross_entropy(y_pred, t)
            acc = F.accuracy(y_pred, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
    if (epoch+1) % 20 == 0:
        logging.info(f"(Epoch{epoch+1})Test Loss: {sum_loss / len(test_set)}, Accuracy: {sum_acc / len(test_set)}")
        print()








