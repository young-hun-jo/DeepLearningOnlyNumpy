# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import layers as L
from dezero import functions as F


# dataset
np.random.seed(42)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)   # label

layer1 = L.Linear(out_size=10)
layer2 = L.Linear(out_size=1)


def predict(x):
    y = layer1(x)
    z = F.sigmoid(y)
    k = layer2(z)
    return k


lr = 0.2
iters = 10000

for i in range(iters):
    # predict
    y_pred = predict(x)
    # loss
    loss = F.mean_squared_error(y, y_pred)

    # init gradients in parameters in middle layers for avoiding accumulated gradients(wrong gradients)
    layer1.clear_grads()
    layer2.clear_grads()

    # backward based on Loss function
    loss.backward(use_heap=True)

    # update parameters
    for layer in [layer1, layer2]:
        for param in layer.params():
            param.data -= lr * param.grad.data

    if (i+1) % 1000 == 0:
        print(f'Epoch:{i+1} -> Loss:', loss)




