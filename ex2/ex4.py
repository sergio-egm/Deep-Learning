import numpy as np 
import matplotlib.pyplot as plt 

data=np.loadtxt("data4.dat",dtype=np.float64)
plt.scatter(data[:,0],data[:,1],color='red')
plt.title("Charged particles")
plt.xlabel("x-coordinates")
plt.ylabel("y-coordinates")
plt.savefig("output.png")
plt.show()