import numpy as np
import matplotlib.pyplot as plt 

x=np.linspace(0,5,num=100,endpoint=True)
y=np.exp(-x)*np.cos(2*np.pi*x)

plt.plot(x,y)
#plt.plot(x,np.exp(-x))
#plt.plot(x,-np.exp(-x))
#plt.plot(x,np.cos(2*np.pi*x))
plt.show()