"""
Python version of a dense NN to solve the MNIST challenge, keeping it nice and simple.
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras import layers, losses

## Load our data
print("Loading data")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

## Normalise and reshape our data into a 784-length vector
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

## Create our model
model = Sequential()
# ! Do we need to add an extra 784 neuron layer to duplicate my implementation?
model.add(
    layers.Dense(
        784, 
        input_shape=(784,),
        use_bias=True,
        activation=layers.Activation("relu")
    )
)
model.add(
    layers.Dense(
        128, 
        input_shape=(784,),
        use_bias=True,
        activation=layers.Activation("relu")
    )
)
model.add(
    layers.Dense(
        64, 
        input_shape=(128,),
        use_bias=True,
        activation=layers.Activation("relu")
    )
)
model.add(
    layers.Dense(
        10, 
        input_shape=(64,),
        use_bias=True,
        activation=layers.Activation("relu")
    )
)
model.summary()

## Compile model
model.compile(
    loss=losses.MeanSquaredError,
    optimizer='adam',
    metrics=['accuracy']
)

## Fit the model, emulating how my version works
model.fit(
    X_train, 
    y_train, 
    batch_size=1, 
    epochs=3,
    verbose=1
)

## Evaluate the model
model.evaluate(
    X_test,
    y_test,
    batch_size=1,
    verbose=1
)