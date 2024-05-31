import numpy as np 
import matplotlib.pyplot as plt 
import hyperopt
from hyperopt import hp

class f:
    def __init__(self):
        self.coeff=np.array([100,-26,12,28,-28,-2,1])
        self.K=0.05
    
    def __call__(self,x):
        return self.K*np.sum(self.coeff*np.array([x**i for i in range(7)]))
    
    def plot(self,xmin,xmax):
        plt.figure()
        x = np.linspace(xmin , xmax , num = 1000 , endpoint = True)
        y = np.array([self(X) for X in x])

        plt.scatter(x , y)

        plt.grid()
        plt.xlabel('x')
        plt.ylabel('f(x)')

        plt.title("$f(x)= 0.05 (x^6 - 2 x^5 - 28 x^4 + 28 x^3 + 12 x^2 -26 x + 100)$")


def uniform_search(func):
    space = hp.uniform('x' , -5 ,6)

    plt.figure()

    plt.hist([ hyperopt.pyll.stochastic.sample(space) for _ in range(1000)])

    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.title('Domain Space')

    plt.grid()


def main():
    my_fun = f()

    my_fun.plot(-5 , 6)

    uniform_search(my_fun)
    plt.show()


if __name__=='__main__':
    main()