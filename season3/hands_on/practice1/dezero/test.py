if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

a = np.array(100)
b = Variable(np.array(5))
y = b * a
print(y)
y.backward()
print(b.grad)