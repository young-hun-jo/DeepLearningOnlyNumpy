import numpy as np
import heapq


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} dtype is not supported.")
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def cleargrad(self):
        self.grad = None
        
    def backward(self, use_heapq=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_sets = set()
        flag = 0 
        
        def add_func(f, use_heapq=False):
            if f not in seen_sets:
                # Python sort 사용
                if not use_heapq:
                    funcs.append(f)
                    seen_sets.add(f)
                    funcs.sort(key=lambda x: x.generation)
                # 우선순위 큐 사용
                else:
                    heapq.heappush(funcs, (-f.generation, flag, f))
                    seen_sets.add(f)
                    
        add_func(self.creator, use_heapq)
        
        while funcs:
            if not heapq:
                f = funcs.pop()
            else:
                f = heapq.heappop(funcs)[2]
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator, use_heapq)
                    flag += 1
            flag = 0


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        # 함수의 세대값 설정
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError("This method should be called in other function class")
        
    def backward(self, gy):
        raise NotImplementedError("This method should be called in other function class")
        
        
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

    
def square(x):
    return Square()(x)

def add(x0, x1):
    return Add()(x0, x1)



# Test case 1
x = Variable(np.array(2.0))

y = add(x, x)
print(y.data)

y.backward(use_heapq=True)
print(y.grad)
print(x.grad)

# Test case 2
x = Variable(np.array(2.0))
a = square(x)
b = square(a)
c = square(a)
y = add(b, c)
print(y.data)

y.backward(use_heapq=True)
print(y.grad)
print(c.grad, b.grad)
print(a.grad)
print(x.grad)
