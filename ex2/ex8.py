import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from scipy.optimize import minimize 

def true_fun(x):
    return np.cos(1.5*np.pi*x)

n_samples=30 
x=np.sort(np.random.rand(n_samples))
y=true_fun(x)+np.random.randn(n_samples)*0.1
x_test=np.linspace(0,1,100,endpoint=True)

deg=[1,4,15]

def loss(p,func):
    ypred=func(list(p),x)
    return tf.reduce_mean(tf.square(ypred-y)).numpy()

for d in deg:
    res=minimize(loss,np.zeros(d+1),args=(tf.math.polyval),method='BFGS')
    plt.plot(x_test,np.poly1d(res.x)(x_test),label=f'{d}')
plt.plot(x_test,true_fun(x_test),label='true')
plt.scatter(x,y,label='samples')
plt.legend()
plt.show()