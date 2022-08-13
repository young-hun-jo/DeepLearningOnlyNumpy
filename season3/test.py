# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.layers as L
from dezero import functions as F
from dezero import Layer


# dataset
np.random.seed(42)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)   # label

# build model
model = Layer()
model.l1 = L.Linear(10)
model.l2 = L.Linear(1)


def predict(x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    # predict and get loss
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    # clear grads
    model.clear_grads()

    # backward
    loss.backward(use_heap=True)

    # update parameters
    for param in model.params():
        param.data -= lr * param.grad.data

    if i == 0 or (i+1) % 1000 == 0:
        print(f"# Epoch:{i+1} -> Loss: {loss}")








