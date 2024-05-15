import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_prediction(model , xmin , xmax , name , color , num=100):
    x=np.linspace(xmin , xmax ,
                  num=num , endpoint=True)

    plt.plot(x , model.predict(x) ,
             label = name + ' model' ,
             color = color)
    

def plot_history(history,name):
    plt.figure()
    plt.title(name + ' model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()

    plt.plot(history.epoch , history.history['loss'] ,
             label = 'Training loss')
    plt.plot(history.epoch , history.history['val_loss'] ,
             label = 'Validation loss')
    
    plt.legend()

def main():
    load_data = np.loadtxt("data.dat")

    train_data = np.copy(load_data[:,:2])
    validation_data = np.copy(load_data[:,2:])


    #Linear fit
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1 , input_shape = (1 , )))

    model.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.01) ,
                  loss = 'mean_squared_error')
    
    history = model.fit(train_data[:,0] , train_data[:,1] ,
                        batch_size = train_data.shape[1] ,
                        epochs = 500 ,
                        validation_data = (validation_data[:,0] , validation_data[:,1]))

    #NN model
    model_nn = tf.keras.models.Sequential()
    model_nn.add(tf.keras.layers.Dense(10 , input_shape = (1 , ) , activation = 'relu'))
    model_nn.add(tf.keras.layers.Dense(10 , activation = 'relu'))
    model_nn.add(tf.keras.layers.Dense(10 , activation = 'relu'))
    model_nn.add(tf.keras.layers.Dense(1))

    model_nn.compile(optimizer = tf.keras.optimizers.SGD(lr = 0.01) ,
                  loss = 'mean_squared_error')

    history_nn = model_nn.fit(train_data[:,0] , train_data[:,1] ,
                        batch_size = train_data.shape[1] ,
                        epochs = 500 ,
                        validation_data = (validation_data[:,0] , validation_data[:,1]))

    plt.scatter(train_data[:,0] , train_data[:,1] ,
                label = 'Train')
    plt.scatter(validation_data[:,0] , validation_data[:,1] ,
                label = 'Validation')
    plot_prediction(model ,
                    np.min(train_data[:,0]) , np.max(train_data[:,0]),
                    name = 'Linear' , color = 'red')
    plot_prediction(model_nn ,
                    np.min(train_data[:,0]) , np.max(train_data[:,0]),
                    name = 'NN' , color = 'lime')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plot_history(history,'Linear')
    plot_history(history_nn,'NN')
    plt.show()



if __name__=='__main__':
    main()