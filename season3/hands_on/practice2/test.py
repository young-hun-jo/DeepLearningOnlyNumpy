from dezero import Variable
import numpy as np

a = Variable(np.array([1,2,3]))
b = np.array([4,5,6])
y = b + a
print(y)
y.backward()
print(a.grad)

