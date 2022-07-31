# # 현재 파일이 전역변수에 있는지 확인하고 있다면, 부모 디렉토리를 모듈검색 경로에 추가!
# if '__file__' in globals():  # 딕셔너리 형태로 반환됨 {'변수명': 변수값, ... , }
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero import functions as F

# make dataset
np.random.seed(42)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)  # noise는 broadcasting 됨
print(y.shape)

# init weight
I, H, O = 1, 50, 1
std = 0.01
W1 = Variable(np.random.randn(I, H) * std)
b1 = Variable(np.zeros(H))
W2 = Variable(np.random.randn(H, O) * std)
b2 = Variable(np.zeros(O))

# predictor
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 1000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    # clear gradient in parameters
    W1.clear_grad()
    b1.clear_grad()
    W2.clear_grad()
    b2.clear_grad()

    # backpropagation
    loss.backward(use_heap=True, retain_grad=False)

    # update parameters using GD
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if (i+1) % 100 == 0:
        print(f"{i+1}th Epoch -> Loss:", loss)

