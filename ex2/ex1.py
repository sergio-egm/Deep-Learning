import numpy as np

a=np.array([[0.5, -1],[-1, 2]],dtype=np.float32)
print(a.shape)
b=a.flatten().copy()
print(b)
b[0::2]=0
print(b)
print(a)