# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import dezero.layers as L
from dezero import functions as F
from dezero.models import MLP
from dezero.optimizers import SGD, MomentumSGD, AdaDelta, AdaGrad, Adam


# dataset
np.random.seed(42)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)   # label

# build model
model = MLP((45, 30, 25, 1))
optimizer = Adam().setup(model)

iters = 5000

for i in range(iters):
    # predict and loss
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    # clear gradients and backpropagation
    model.clear_grads()
    loss.backward(use_heap=True)

    # update parameters
    optimizer.update()

    # verbose
    if (0 <= i <= 30) or (i % 500) == 0:
        print(f"Epoch:{i+1} => Loss:{loss.data}")








