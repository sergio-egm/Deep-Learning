import tensorflow as tf
import matplotlib.pyplot as plt

(train_images , train_labels) , (test_images , test_labels) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane' , 'automobile' , 'bird' , 'cat' , 'deer' , 'dog' , 'frog' , 'horse' , 'ship' , 'truck']

# plt.figure(figsize = (10 , 10))
# for i in range(25):
#     plt.subplot(5 , 5 , i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i][0]])
# 
# plt.show()
 
model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.Rescaling(1./255 , input_shape = (32 , 32 , 3)))
# model1.add(tf.keras.layers.Flatten())
# model1.add(tf.keras.layers.Dense(64 , activation = 'relu'))
model1.add(tf.keras.layers.Conv2D(32 , (3 , 3) , activation = 'relu'))
model1.add(tf.keras.layers.MaxPooling2D((2 , 2)))
model1.add(tf.keras.layers.Conv2D(32 , (3 , 3) , activation = 'relu'))
model1.add(tf.keras.layers.MaxPooling2D((2 , 2)))
model1.add(tf.keras.layers.Conv2D(32 , (3 , 3) , activation = 'relu'))
model1.add(tf.keras.layers.Flatten())
model1.add(tf.keras.layers.Dense(64 , activation = 'relu'))
model1.add(tf.keras.layers.Dense(10 , activation = 'softmax'))

model1.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])
model1.summary()

history = model1.fit(train_images , train_labels , epochs = 10 , validation_data = (test_images , test_labels))
test_loss ,test_acc = model1.evaluate(test_images , test_labels , verbose = 2)

print(f"Test loss {test_loss} - Test accurecy {test_acc}")

plt.figure()
plt.plot(history.history['accuracy']     , label = 'Accuracy')
plt.plot(history.history['val_accuracy'] , label = 'Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0 , 1])
plt.legend()
plt.show()