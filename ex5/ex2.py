import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def create_model(m_shape):
    model=tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Flatten(input_shape=m_shape))
    model.add(tf.keras.layers.Dense(2,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    return model


def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    print(f"Train shape : {np.shape(train_images)}")
    print(f"Test  shape : {np.shape(test_images)}")

    norm = np.max(train_images)

    train_images = train_images / norm
    test_images  = test_images / norm

    model=create_model(np.shape(train_images[0]))
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=5)
    acc=model.evaluate(test_images,test_labels)
    print(acc)



if __name__=='__main__':
    main()