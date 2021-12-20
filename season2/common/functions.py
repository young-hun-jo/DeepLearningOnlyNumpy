import numpy as np

def sigmoid(x: np.array):
    return 1 / (1 + np.exp(-x))


def relu(x: np.array):
    return np.maximum(0, x)


def softmax(x: np.array):
    """
    x: 2차원 데이터로 입력될 시, shape: (batch_size, feature)
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y: np.array, t: np.array):
    """
    y: Softmax 확률 값
    t: 클래스 정답
    """
    # 데이터가 1개 들어왔을 경우
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    # 레이블을 One-hot -> label 형태로 변경
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
