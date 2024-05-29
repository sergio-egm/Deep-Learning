import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Rescaling , Conv2D , MaxPooling2D , Flatten , Dense , Dropout


#Load dataset by url
def load_datasets(img_height , img_width , batch_size , val_split , seed):
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    data_dir    = tf.keras.utils.get_file('flower_photos' , origin = dataset_url , untar = True)
    data_dir    = pathlib.Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir ,
        validation_split = val_split ,
        subset     = 'training' ,
        image_size = (img_height , img_width) ,
        batch_size = batch_size ,
        seed       = seed
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir ,
        validation_split = val_split ,
        subset     = 'validation' ,
        image_size = (img_height , img_width) ,
        batch_size = batch_size ,
        seed       = 0
    )

    return train_ds , validation_ds

def create_model(filters , model = None):
    if model is None:
        model = tf.keras.models.Sequential()

    model.add(Rescaling(1./255 , input_shape = (180 , 180 , 3)))
    for f in filters:
        model.add(Conv2D(f , (3 , 3) , activation = 'relu'))
        model.add(MaxPooling2D((2 , 2)))
    model.add(Dropout(rate = 0.3))
    model.add(Flatten())
    model.add(Dense(128 , activation = 'relu'))


    return model


def plot_loss(history):
    plt.grid()
    plt.plot(history.history['loss']     , label = 'Train Loss')
    plt.plot(history.history['val_loss'] , label = 'Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

def plot_accuracy(history):
    plt.grid()
    plt.plot(history.history['accuracy']     , label = 'Train Accuracy')
    plt.plot(history.history['val_accuracy'] , label = 'Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()




def create_model_aum(filters):
    model = tf.keras.models.Sequential()

    model.add(Rescaling(1./255 , input_shape = (180 , 180 , 3)))
    for f in filters:
        model.add(Conv2D(f , (3 , 3) , activation = 'relu'))
        model.add(MaxPooling2D((2 , 2)))
    model.add(Flatten())
    model.add(Dense(128 , activation = 'relu'))


    return model



def plot_samples(model , data):
    plt.figure(figsize = (10 , 10))
    for image in data.take(1):
        augmented_image = model(image)
        for i in range(8):
            plt.subplot(4 , 4 ,2*i+1)
            plt.imshow(augmented_image[i].numpy().astype('uint8'))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(4 , 4 ,2*(i+1))
            plt.imshow(image[0][i].numpy().astype('uint8'))
            plt.xticks([])
            plt.yticks([])

    
    plt.subplots_adjust(hspace = .5 , wspace = .2)




def main():
    print('Load and split datasets')
    train_ds , validation_ds = load_datasets(img_height = 180 , img_width = 180 ,
                                             batch_size = 32 ,
                                             val_split  = .2 ,
                                             seed       = 0)
    
    print('\nCreate model')
    model = create_model([15 , 32 , 64])
    model.compile(optimizer = 'adam' ,
                  loss      = 'sparse_categorical_crossentropy' ,
                  metrics   = ['accuracy'])

    # model.summary()

    data_aumentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal',
                                       input_shape = (180 , 180 , 3)) ,
            tf.keras.layers.RandomRotation(.1) ,
            tf.keras.layers.RandomZoom(.1)
        ]
    )

    # plot_samples(data_aumentation , train_ds)

    data_aumentation = create_model([15 , 32 , 64] , data_aumentation)

    data_aumentation.summary()

    data_aumentation.compile(optimizer = 'adam' ,
                            loss      = 'sparse_categorical_crossentropy' ,
                            metrics   = ['accuracy'])

    train_ds      = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    # history = model.fit(train_ds ,
    #                     epochs = 10 ,
    #                     validation_data = validation_ds)

    history = data_aumentation.fit(train_ds ,
                                    epochs = 2 ,
                                    validation_data = validation_ds)
    

    plt.figure()

    plt.subplot(1 , 2 , 1)
    plot_loss(history)
    plt.subplot(1 , 2 , 2)
    plot_accuracy(history)

    plt.subplots_adjust(hspace = 0.1 , wspace = 0.5)

    plt.show()












if __name__ == '__main__':
    main()
