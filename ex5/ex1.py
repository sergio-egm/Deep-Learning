import numpy as np 
import matplotlib.pyplot as plt 
import hyperopt
from hyperopt import hp , tpe , Trials , STATUS_OK , fmin , rand
from time import time

class f:
    def __init__(self):
        self.coeff=np.array([100,-26,12,28,-28,-2,1])
        self.K=0.05
    
    def __call__(self,x):
        return self.K*np.sum(self.coeff*np.array([x**i for i in range(7)]))
    
    def plot(self,xmin,xmax,best = None):
        plt.figure()
        x = np.linspace(xmin , xmax , num = 1000 , endpoint = True)
        y = np.array([self(X) for X in x])

        plt.scatter(x , y)

        if best is not None:
            plt.axvline(best[0]['x'] , color = 'red'    , label = 'TPE')
            plt.axvline(best[1]['x'] , color = 'orange' , label = 'RANDOM')

        plt.grid()
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('f(x)')

        plt.title("$f(x)= 0.05 (x^6 - 2 x^5 - 28 x^4 + 28 x^3 + 12 x^2 -26 x + 100)$")


def uniform_search(func):
    space = hp.uniform('x' , -5 ,6)

    plt.figure()

    plt.hist([ hyperopt.pyll.stochastic.sample(space) for _ in range(1000)] , density = True , bins = 100)

    plt.xlabel('x')
    plt.ylabel('Frequency')
    plt.title('Domain Space')

    plt.grid()

    return space


def optimize(space , f , algorithm):
    trials    = Trials()

    best      = fmin(fn        = f,
                     space     = space ,
                     algo      = algorithm ,
                     max_evals = 2000 ,
                     trials    = trials)
    
    return trials , best


def plot_trials(trials , best , title):
    plt.figure(figsize = (10 , 5))
    plt.subplot(1 , 2 , 1)
    plt.grid()

    plt.scatter(trials.idxs_vals[0]['x'] , trials.idxs_vals[1]['x'])

    plt.axhline(best['x'] , color = 'red')

    plt.xlabel('Iterrations')
    plt.ylabel('x')
    plt.title(title)

    plt.subplot(1 , 2 , 2)
    plt.grid()

    plt.hist(trials.idxs_vals[1]['x'] , bins = 100 , density = True)
    plt.xlabel('x')
    plt.ylabel('Frequency')

    plt.axvline(best['x'] , color = 'red')


def main():
    my_fun = f()

    space = uniform_search(my_fun)

    trials , best = optimize(space , my_fun , tpe.suggest)
    print(best)
    appo = [best]

    plot_trials(trials , best , 'TPE')

    trials , best = optimize(space , my_fun , rand.suggest)
    print(best)
    appo.append(best)

    plot_trials(trials , best , 'RANDOM')

    my_fun.plot(-5 , 6 , appo)
    plt.show()


if __name__=='__main__':
    main()