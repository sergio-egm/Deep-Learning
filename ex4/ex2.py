import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data=tf.keras.datasets.fashion_mnist.load_data()
print(f"( {len(data)} , {len(data[1])} , {np.shape(data[0][1])} )")
print(data[0][1])
plt.imshow(data[0][0][55],cmap=plt.cm.binary)
plt.title(data[0][1][55])
plt.xticks([])
plt.yticks([])
plt.show()