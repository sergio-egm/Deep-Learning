import numpy as np
import matplotlib.pyplot as plt 


def f(x):
    return -np.sin(x*x)/x+0.01*x*x

x=np.linspace(-3,3,num=100,endpoint=True)
np.savetxt("output.dat",np.vstack([x,f(x)]).T)
plt.plot(x,f(x),marker='o',label=r'$f(x)=-\frac{sin(x^2)}{x}+0.01\,x^2$')
plt.xlim(-3,3)
plt.title('f(x)')
plt.legend()
plt.xlabel("x")
plt.ylabel('f(x)')
plt.savefig('output5.png')
plt.show()