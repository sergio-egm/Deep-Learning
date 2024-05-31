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
    
    plt.subplots_adjust(hspace = .2 , wspace = .5)



def create_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape = (28 , 28)))
    model.add(tf.keras.layers.Dense(128 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(10  , activation = 'softmax'))

    return model

def plot_history(history):
    plt.figure(figsize = (10 , 8))

    plt.subplot(1 , 2 ,1)

    plt.plot(history.history['loss']     , label = 'Train')
    plt.plot(history.history['val_loss'] , label = 'Validation')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1 , 2 ,2)

    plt.plot(history.history['accuracy']     , label = 'Train')
    plt.plot(history.history['val_accuracy'] , label = 'Validation')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace = .2 , wspace = .5)



def plot_errors(ds_img , ds_lab , model):
    plt.figure(figsize = (10 , 10))
    evaluations = model(ds_img)

    count = 0
    n     = 0

    while count < 36 :
        eva = np.argmax(evaluations[n])

        if eva != ds_lab[n] :
            plt.subplot(6 , 6 , count + 1)

            plt.title(f"{data_target[eva]}\n{data_target[ds_lab[n]]}")
            plt.imshow(ds_img[n] , cmap = 'binary')

            plt.xticks([])
            plt.yticks([])

            count += 1
        
        n +=1
        if n == len(ds_img):
            print(f"{count} images")
            break
    
    plt.subplots_adjust(hspace = .5 , wspace = .5)



    

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
    
    model = create_model()

    model.summary()

    model.compile(
        optimizer = 'adam' ,
        loss      = 'sparse_categorical_crossentropy' ,
        metrics   = ['accuracy']
    )

    history = model.fit(train_data , train_values ,
                        validation_data=(test_data , test_values) ,
                        epochs = 10)
    
    plot_history(history)
    plot_errors(test_data , test_values , model)

    plt.show()

if __name__=='__main__':
    main()