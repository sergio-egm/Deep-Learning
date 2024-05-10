import numpy as np 
import matplotlib.pyplot as plt 
import hyperopt
from tqdm import tqdm

class f:
    def __init__(self):
        self.coeff=np.array([100,-26,12,28,-28,-2,1])
        self.K=0.5
    
    def __call__(self,x):
        return self.K*np.sum(self.coeff*np.array([x**i for i in range(7)]))

def main():
    my_fun=f()
    x=np.linspace(-5,6,endpoint=True,num=1000)
    y=np.array([my_fun(xi) for xi in tqdm(x)])

    plt.scatter(x,y)
    plt.show()


if __name__=='__main__':
    main()