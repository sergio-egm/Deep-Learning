import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

train_data  = np.load('training_data.npy')
train_label = np.load('training_label.npy')
test_data   = np.load('test_data.npy')
test_label  = np.load('test_label.npy')

print(f"Trainig data shape: {train_data.shape[0]} x {train_data[0].shape}")
print(f"Test data shape   : {test_data.shape[0]}  x {test_data[0].shape}\n")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(30 , activation = 'relu' , input_shape = (train_data.shape[1] , 1)))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer = 'adam' , loss = 'mse')
history = model.fit(train_data , train_label , epochs = 25 , batch_size = 32)  #Not working merged two comands

train_predict = model.predict(test_data)

plt.figure(figsize = (10 , 8))

plt.subplot(3 , 1 , 1)
plt.plot(train_label , label = 'train data')
plt.plot(range(len(train_label) , len(train_label) + len(test_label)) , test_label , 'k' , label = 'test data')
plt.legend()
plt.xlabel('day')
plt.ylabel('temperature')
plt.title('All day')

plt.subplot(3 , 2 , 3)
plt.plot(test_label , color = 'k',label = 'True value')
plt.plot(train_predict , color = 'red' , label = 'predicted')
plt.legend()
plt.xlabel('day')
plt.ylabel('temperature')
plt.title('Predicted data (full data set)')

plt.subplot(3 , 2 , 4)
plt.plot(test_label[:100] , color = 'k',label = 'True value')
plt.plot(train_predict[:100] , color = 'red' , label = 'predicted')
plt.legend()
plt.xlabel('day')
plt.ylabel('temperature')
plt.title('Predicted data (first 100 days)')

plt.show()