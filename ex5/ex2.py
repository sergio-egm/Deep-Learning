import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import STATUS_OK , Trials , fmin , tpe , space_eval , hp

def plot_samples(images , labels):
    plt.figure(figsize=(10 , 10))

    for i in range(36):
        plt.subplot(6 , 6 , i+1)

        plt.imshow(images[i] , cmap = 'binary')
        plt.title(labels[i])

        plt.xticks([])
        plt.yticks([])

        
    plt.subplots_adjust(hspace = .3 , wspace = .5)

def create_model(m_shape):
    model=tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Flatten(input_shape=m_shape))
    model.add(tf.keras.layers.Dense(2,activation='relu'))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    return model



def train(featurs , labels ,parameters):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape = (28 , 28)))
    model.add(tf.keras.layers.Dense(parameters['layer_size'] ,
                                    activation = 'relu'))
    model.add(tf.keras.layers.Dense(10 ,
                                    activation = 'softmax'))
    

    optimizer = tf.keras.optimizers.Adam(learning_rate = parameters['learning_rate'])

    model.compile(optimizer = optimizer ,
                  loss      = 'sparse_categorical_crossentropy' ,
                  metrics   = ['accuracy'])
    
    model.fit(featurs , labels , epochs = 5)

    return model


def test(model , features , labels):
    acc = model.evaluate(features , labels)
    return acc













def main():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    print(f"Train shape : {np.shape(train_images)}")
    print(f"Test  shape : {np.shape(test_images)}")

    norm = np.max(train_images)

    train_images = train_images / norm
    test_images  = test_images / norm

    plot_samples(train_images , train_labels)

    model=create_model(np.shape(train_images[0]))
    #model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    #model.fit(train_images,train_labels,epochs=5)

    #acc=model.evaluate(test_images,test_labels)
    #print(acc)



    #Hyperopt DNN
    train_setup = {
        'layer_size'    : 2 ,
        'learning_rate' : 1.
    }

    model = train(train_images , train_labels , train_setup)

    print(f'Accuracy : {test(model , test_images , test_labels)[1] * 100:.1f} %')


    def hyper_func(params):
        model    = train(train_images , train_labels , params)
        test_acc = test(model , test_images , test_labels)

        return {'loss'   : - test_acc[1] ,
                'status' : STATUS_OK}
    
    search_space = {
        'layer_size'    : hp.choice('layer_size' , np.arange(10 , 100 , 20)) ,
        'learning_rate' : hp.loguniform('learning_rate' , -10 , 0)
    }

    trials = Trials()
    best   = fmin(hyper_func ,
                  space     = search_space ,
                  algo      = tpe.suggest ,
                  max_evals = 5 ,
                  trials    = trials)
    
    print(space_eval(search_space , best))

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    xs = [t['tid'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]
    ax1.set_xlim(xs[0]-1, xs[-1]+1)
    ax1.scatter(xs, ys, s=20)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['layer_size'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    ax2.scatter(xs, ys, s=20)
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Accuracy')

    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [-t['result']['loss'] for t in trials.trials]

    ax3.scatter(xs, ys, s=20)
    ax3.set_xlabel('learning_rate')
    ax3.set_ylabel('Accuracy')
    plt.show()

    plt.show()



if __name__=='__main__':
    main()