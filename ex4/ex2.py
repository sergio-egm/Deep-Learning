import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Loading data
data_target=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_images(data , target , index):
    lenght = index.shape[0] * index.shape[1]
    appo = np.reshape(index , lenght)
    #print(index.shape)

    plt.figure(figsize = (10 , 10))

    for i in range(lenght):
        plt.subplot(index.shape[0] , index.shape[1] , i + 1)
        plt.imshow(data[appo[i]],cmap = 'Greys')
        plt.xticks([])
        plt.yticks([])
        plt.title(data_target[target[appo[i]]])

    

def main():
    (train_data , train_values) , (test_data , test_values) = tf.keras.datasets.fashion_mnist.load_data()

    print(f"Train set shape: {len(train_data)} x {np.shape(train_data[0])}")
    print(f"Test  set shape: {len(test_data) } x {np.shape(test_data[0]) }\n")

    #Normalization datasets
    pixel_max  = np.max(train_data)
    print(f"Max Value: {pixel_max}")

    train_data = train_data / pixel_max
    test_data  = test_data  / pixel_max

    plot_images(train_data   ,
                train_values ,
                np.random.randint(low  = 0 ,
                                  high = len(train_data) ,
                                  size = (6,6)))

    plt.show()

if __name__=='__main__':
    main()