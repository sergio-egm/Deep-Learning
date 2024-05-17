import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

iris = load_iris()

df = pd.DataFrame(iris.data , columns = iris.feature_names)
df['label'] = iris.target_names[iris.target]
print(df)

#sns.pairplot(df , hue = 'label')
#plt.show()

label = pd.get_dummies(df['label'] , prefix = 'label')
print(label)
df = pd.concat([df , label] , axis = 1)
df.drop(['label'] , axis = 1 , inplace = True)

print(df)

train_dataset = df.sample(frac = 0.8 , random_state = 1)
test_dataset = df.drop(train_dataset.index)

X_train = train_dataset[['sepal length (cm)' , 'sepal width (cm)' , 'petal length (cm)' , 'petal width (cm)']]
y_train = train_dataset[['label_setosa' , 'label_versicolor' , 'label_virginica']]

#Build model

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64  , activation = 'relu' , input_shape = (4 , )),
    tf.keras.layers.Dense(128 , activation = 'relu'),
    tf.keras.layers.Dense(128 , activation = 'relu'),
    tf.keras.layers.Dense(128 , activation = 'relu'),
    tf.keras.layers.Dense(64  , activation = 'relu'),
    tf.keras.layers.Dense(64  , activation = 'relu'),
    tf.keras.layers.Dense(64  , activation = 'relu'),
    tf.keras.layers.Dense(3   , activation = 'softmax')
])

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

history = model.fit(
    X_train ,
    y_train ,
    epochs = 200 ,
    validation_split = 0.4 ,
    batch_size = 32 ,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = 'val_loss' , patience = 10) ,
        tf.keras.callbacks.TensorBoard(log_dir = './log') ,
    ] ,
)

train_metrics = history.history['loss']
val_metrics   = history.history['val_loss']
epochs        = range(1 , len(train_metrics) + 1)

plt.plot(epochs , train_metrics , label = 'trainig loss')
plt.plot(epochs , val_metrics   , label = 'validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()