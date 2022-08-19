# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.models import MLP
from dezero import functions as F
from dezero.optimizers import SGD  # SGD를 import


# dataset
np.random.seed(42)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# build and compile model
model = MLP((25, 40, 3, 1))
lr = 0.01
iters = 10000
optimizer = SGD(lr).setup(model)  # 달라진 부분!

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.clear_grads()
    loss.backward()

    # 달라진 부분!
    optimizer.update()

    if (0 <= i <= 30) or (i % 1000) == 0:
        print(f"Epoch:{i+1} -> Loss:{loss.data}")











