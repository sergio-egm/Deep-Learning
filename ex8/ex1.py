import pathlib
import tensorflow as tf
from tensorflow.keras.layers import Rescaling , Conv2D , MaxPooling2D , Flatten , Dense


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

def create_model(filters):
    model = tf.keras.models.Sequential()

    model.add(Rescaling(1./255 , input_shape = (180 , 180 , 3)))
    for f in filters:
        model.add(Conv2D(f , (3 , 3) , activation = 'relu'))
        model.add(MaxPooling2D((2 , 2)))
    model.add(Flatten())
    model.add(Dense(128 , activation = 'relu'))


    return model







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

    model.summary()

    train_ds      = train_ds.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    history = model.fit(train_ds ,
                        epochs = 2 ,
                        validation_data = validation_ds)












if __name__ == '__main__':
    main()
