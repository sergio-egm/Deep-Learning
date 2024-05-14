import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

n_input   = 1
n_hiden_1 = 5
n_hiden_2 = 2
n_output  = 1

#Set random weights and bias
w={
    'w1' : tf.Variable(tf.random.normal((n_input, n_hiden_1))),
    'w2' : tf.Variable(tf.random.normal((n_hiden_1, n_hiden_2))),
    'w3' : tf.Variable(tf.random.normal((n_hiden_2, n_output)))
}

b={
    'b1' : tf.Variable(tf.random.normal([n_hiden_1]) , name='b1'),
    'b2' : tf.Variable(tf.random.normal([n_hiden_2]) , name='b2'),
    'b3' : tf.Variable(tf.random.normal([n_output])  , name='b3')
}

#Define a tensor w/ sigmoid activation function
def mlp(x):
    l1 = tf.sigmoid(tf.add(tf.matmul(x,w['w1'])  , b['b1']))
    l2 = tf.sigmoid(tf.add(tf.matmul(l1,w['w2']) , b['b2']))
    l3 = tf.sigmoid(tf.add(tf.matmul(l2,w['w3']) , b['b3']))
    return l3

#Testing the model
x = np.linspace(-1,1,10,dtype=np.float32).reshape(-1,1)

print(mlp(x))