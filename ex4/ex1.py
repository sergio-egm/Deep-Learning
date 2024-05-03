import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    load_data=np.loadtxt("data.dat")

    train_data=np.copy(load_data[:,:2])
    validation_data=np.copy(load_data[:,2:])


    #Model
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1,input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01),loss='mean_squared_error')
    history=model.fit(train_data[:,0],train_data[:,1],batch_size=train_data.shape[1],epochs=500,validation_data=(validation_data[:,0],validation_data[:,1]))

    plt.scatter(train_data[:,0],train_data[:,1],label='Train')
    plt.scatter(validation_data[:,0],validation_data[:,1],label='Validation')

    plt.legend()
    plt.grid()
    plt.show()



if __name__=='__main__':
    main()