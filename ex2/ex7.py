import numpy as np 
import matplotlib.pyplot as plt 

def true_function(x):
    return np.cos(1.5*np.pi*x)

def f(x):
    return true_function(x)+np.random.randn()*0.1
'''
def polynomial(x,N,p):
    y=0
    for i in range(N+1):
        y+=p[i]*x**(N-i)
'''
x=np.linspace(0,1,num=100,endpoint=True)

x0=np.sort(np.random.rand(30))
y=f(x0)
plt.scatter(x0,y)

for i in [1,4,15]:
    p=np.polyfit(x0,y,i)
    z=np.poly1d(p)
    plt.plot(x,z(x),label=f'{i}')

plt.plot(x,true_function(x),label='True')
plt.legend()
plt.show()