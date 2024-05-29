import pathlib
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def get_dataset(dataset_url):
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=dataset_url, extract=True)
    path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = pathlib.Path(path) / 'train'
    validation_dir = pathlib.Path(path) / 'validation'

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir ,
        validation_split = .2 ,
        subset     = 'training' ,
        image_size = (256 , 256) ,
        batch_size = 32 ,
        seed       = 0 
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        validation_split = .2 ,
        subset     = 'validation' ,
        image_size = (256 , 256) ,
        batch_size = 32 ,
        seed       = 0 
    )

    return train_ds , validation_ds



def plot_samples(data , model):
    names = ['Cat' , 'Dog']
    plt.figure(figsize = (10 , 10))

    for image in data.take(1):
        appo_img = model(image)
        for i in range(18):
            plt.subplot(6 , 6 , 2*i + 1)
            plt.imshow(appo_img[i].numpy().astype('uint8'))
            plt.xticks([])
            plt.yticks([])

            plt.subplot(6 , 6 , 2* (i + 1))
            plt.imshow(image[0][i].numpy().astype('uint8'))
            plt.xticks([])
            plt.yticks([])
        
    plt.subplots_adjust(hspace = .3 , wspace = .5)



def create_model(model):
    base_model = tf.keras.applications.MobileNetV2(input_shape = (256 , 256 , 3) ,
                                                   include_top = False ,
                                                   weights     = 'imagenet')
    
    base_model.trainable = False

    model.add(tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input))
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Dense(1))

    return model


def plot_history(history):
    plt.figure(figsize = (10 , 10))

    plt.subplot(1 , 2 , 1)
    plt.plot(history.history['loss'] , label = 'Training')
    plt.plot(history.history['val_loss'] , label = 'Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1 , 2 , 2)
    plt.plot(history.history['accuracy'] , label = 'Training')
    plt.plot(history.history['val_accuracy'] , label = 'Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplots_adjust(hspace = .2 , wspace = .5)








def main():
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    train_ds , validation_ds = get_dataset(dataset_url)
    # print(train_ds)

    data_aumentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal' ,
                                       input_shape = (256 , 256 , 3)) ,
            tf.keras.layers.RandomRotation(.2)
        ]
    )

    plot_samples(train_ds , data_aumentation)
    plot_samples(validation_ds , data_aumentation)

    model = create_model(data_aumentation)
    model.compile(
        optimizer     = tf.keras.optimizers.Adam(learning_rate=1e-4) ,
        loss          = tf.keras.losses.BinaryCrossentropy(from_logits=True) ,
        metrics       = ['accuracy']
    )

    model.summary()

    train_ds      = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    history = model.fit(train_ds ,
                        epochs = 10 ,
                        validation_data = validation_ds)
    
    plot_history(history)

    plt.show()






if __name__ == '__main__':
    main()