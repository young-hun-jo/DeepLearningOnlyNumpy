import numpy as np

def numerical_gradient(f, x: np.array):
    """
    f: 손실함수
    x: 입력 값
    """
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        # f(x+h)
        x[idx] = tmp_val + h
        fx1 = f(x)
        # f(x-h)
        x[idx] = tmp_val - h
        fx2 = f(x)
        
        grad[idx] = (fx1 - fx2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()
        
    return grad
