import tensorflow as tf 
import numpy as np 


tf.random.set_seed(0)

n_input=1
n_hiden_1=5
n_hiden_2=2
n_output=1


#Ex1

w={
    'w1': tf.Variable(tf.random.normal((n_input, n_hiden_1))),
    'w2': tf.Variable(tf.random.normal((n_hiden_1, n_hiden_2))),
    'w3': tf.Variable(tf.random.normal((n_hiden_2, n_output)))
}

b={
    'b1': tf.Variable(tf.random.normal([n_hiden_1]), name='b1'),
    'b2': tf.Variable(tf.random.normal([n_hiden_2]), name='b2'),
    'b3': tf.Variable(tf.random.normal([n_output]), name='b3')
}

def mlp(x):
    l1=tf.sigmoid(tf.add(tf.matmul(x,w['w1']),b['b1']))
    l2=tf.sigmoid(tf.add(tf.matmul(l1,w['w2']),b['b2']))
    l3=tf.sigmoid(tf.add(tf.matmul(l2,w['w3']),b['b3']))
    return l3

x=np.linspace(-1,1,10,dtype=np.float32).reshape(-1,1)

y1=mlp(x)


#Keras

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(n_hiden_1,activation='sigmoid',input_dim=n_input))
model.add(tf.keras.layers.Dense(n_hiden_2,activation='sigmoid'))
model.add(tf.keras.layers.Dense(n_output,activation='linear'))

model.summary()

model.set_weights(
    [w['w1'],b['b1'],
    w['w2'],b['b2'],
    w['w3'],b['b3']]
)

y=model.predict(x)

print(np.allclose(y,y1))