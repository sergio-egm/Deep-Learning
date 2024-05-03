import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 


tf.random.set_seed(0)

def f(x):
    return x*3.0+2.0

def generate_data():
    x=tf.linspace(-2,2,200)
    x=tf.cast(x,tf.float32)
    noise=tf.random.normal(shape=x.shape)
    y=f(x)+noise 
    return x,y 

class MyModel(tf.Module):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
    
    def __call__(self,x):
        return self.w*x+self.b 


def loss(target_y,predicted_y):
    return tf.reduce_mean(tf.square(target_y-predicted_y))


def report(model,loss):
    return f"W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, loss = {loss:2.5f}"

def train(model,x,y,learning_rate):
    with tf.GradientTape() as t:
        current_loss=loss(y,model(x))
    
    dw,db=t.gradient(current_loss,[model.w,model.b])
    model.w.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)

def trainig_loop(model,x,y,epochs):
    weights=[]
    baias=[]

    for epoch in epochs:
        train(model,x,y,learning_rate=0.1)
        weights.append(model.w.numpy())
        baias.append(model.b.numpy())
        current_loss=loss(y,model(x))
        print(f'Epoch {epoch:2d}:')
        print("\t",report(model,current_loss))
    
    return weights,baias



x,y=generate_data()

model=MyModel()
ypred=model(x)

current_loss=loss(y,model(x))

print("Untrained model loss: %1.6f" % current_loss.numpy())
print("Starting...")
print('\t',report(model,current_loss))
epochs=range(10)
weights,baias=trainig_loop(model,x,y,epochs)
print("Trained loss %1.6f" % loss(model(x),y).numpy())