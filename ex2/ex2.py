import numpy as np 
from numpy.random import rand
import time
import numba as nb

def dot_prod(A,v):
    w=np.zeros(v.shape[0])
    for i in range(v.shape[0]):
        w[i]=sum(A[i]*v)
    return w

@nb.njit("float64[:](float64[:,:],float64[:])")
def dot_prod1(A,v):
    w=np.zeros(v.shape[0])
    for i in range(v.shape[0]):
        w[i]=sum(A[i]*v)
    return w

N=5000

v=np.random.rand(N).astype(np.float64)
#print(v)
A=np.array(rand(N,N),dtype=np.float64)
#print(A)

print("Self-made")
t0=time.time()
w0=dot_prod(A,v)
print("Execution time: {:e} s".format(time.time()-t0))

print("Numpy")
t0=time.time()
w1=np.dot(A,v)
print("Execution time: {:e} s".format(time.time()-t0))

if not np.allclose(w0,w1):
    raise ValueError("Values doesn't meatch with numpy!")

print("Numba")
t0=time.time()
w2=dot_prod1(A,v)
print("Execution time: {:e} s".format(time.time()-t0))

if not np.allclose(w0,w2):
    raise ValueError("Values doesn't meatchwith numba!")