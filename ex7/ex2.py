import numpy as np
import tensorflow as tf
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def get_datasets():
    training_images = np.load('data/training_images.npy')
    training_boxes  = np.load('data/training_boxes.npy')
    training_labels = np.load('data/training_labels.npy')

    print(f'Training images {training_images.shape}')
    print(f'Training boxes  {training_boxes.shape}')
    print(f'Training labels {training_labels.shape}')

    training_ds={
        'img': training_images ,
        'box': training_boxes  ,
        'lab': training_labels
    }

    validation_images = np.load('data/validation_images.npy')
    validation_boxes  = np.load('data/validation_boxes.npy')
    validation_labels = np.load('data/validation_labels.npy')

    print(f'Validation images {validation_images.shape}')
    print(f'Validation boxes  {validation_boxes.shape}')
    print(f'Validation labels {validation_labels.shape}')

    validation_ds={
        'img': validation_images ,
        'box': validation_boxes  ,
        'lab': validation_labels
    }

    return training_ds , validation_ds



def plot_samples(ds , box_sample = None):
    plt.figure(figsize = (10 , 10))

    for i in range(25):
        plt.subplot(5 , 5 , i+1)

        plt.imshow(ds['img'][i] , cmap = 'binary')
        
        box    = ds['box'][i] * len(ds['img'][i])
        xy     = (box[0] , box[1])
        height = box[2] - box[0]
        width  = box[3] - box[1]

        # print(xy,height,width)

        plt.gca().add_patch(Rectangle(
            xy ,
            height = height ,
            width  = width ,
            edgecolor = 'blue' ,
            facecolor = 'none' ,
            lw = 1
        ))
        if box_sample is not None:
            box    = box_sample[i] * 75
            xy     = (box[0] , box[1])
            height = box[2] - box[0]
            width  = box[3] - box[1]

            plt.gca().add_patch(Rectangle(
                xy ,
                height = height ,
                width  = width ,
                edgecolor = 'red' ,
                facecolor = 'none' ,
                lw = 1
            ))

        plt.xticks([])
        plt.yticks([])
    
    plt.subplots_adjust(hspace = .3 , wspace = .5)



def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16 , 3 , activation = 'relu')(inputs)
    x = tf.keras.layers.AveragePooling2D((2 , 2))(x)
    x = tf.keras.layers.Conv2D(32 , 3 , activation = 'relu')(inputs)
    x = tf.keras.layers.AveragePooling2D((2 , 2))(x)
    x = tf.keras.layers.Conv2D(64 , 3 , activation = 'relu')(inputs)
    x = tf.keras.layers.AveragePooling2D((2 , 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128 , activation = 'relu')(x)

    return x

def regresson(inputs):
    return tf.keras.layers.Dense(4 , name = 'bounding_box')(inputs)

def classifier(inputs):
    return tf.keras.layers.Dense(10 , activation = 'softmax' , name = 'classifier')(inputs)


def create_model():
    inputs       = tf.keras.layers.Input(shape = (75 , 75 , 1))
    dense_output = feature_extractor(inputs)
    bounding_box = regresson(dense_output)
    class_layer  = classifier(dense_output)

    model  = tf.keras.Model(inputs = inputs , outputs = [bounding_box , class_layer])
    return model

def plot_history(history):
    plt.figure(figsize = (10 , 10))

    plt.subplot(1 , 2 , 1)

    plt.plot(history.history['bounding_box_mse'])
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Bounding Boxes')

    plt.subplot(1 , 2 , 2)

    plt.plot(history.history['classifier_acc'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Classifier')

    plt.subplots_adjust(hspace = .2 , wspace= .5)







def main():
    training_ds , validation_ds = get_datasets()
    plot_samples(training_ds)

    model = create_model()

    model.summary()

    model.compile(
        optimizer = 'adam' ,
        loss = {
            'classifier'   : 'categorical_crossentropy' ,
            'bounding_box' : 'mse'
        } ,
        metrics = {
            'classifier'   : 'acc' ,
            'bounding_box' : 'mse'
        }
    )

    history = model.fit(
        training_ds['img'] , (training_ds['box'] , training_ds['lab']) ,
        #validation_data = (
        #    validation_ds['img'] , (validation_ds['box'] , validation_ds['lab'])
        #) ,
        epochs = 10
    )

    plot_samples(training_ds , model.evaluate(training_ds['img']))
    plot_history(history)
    plt.show()


if __name__ == '__main__':
    main()
