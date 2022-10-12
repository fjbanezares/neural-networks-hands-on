# keras hello-world

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras
print(tf.__version__)
```

We define the simplest neural network with only one neuron.

For this we build the model in Python using Keras

1. loss function we use is MSE in each iteration we will check the error this way
2. optimizer algorithm is Stochastic Gradient Descent, in each iteration we advance towards the minimum loss


```python
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')
```

Now we define the dataset, in this case a few points on the line y=2x-1

```python
# Declare model inputs and outputs for training
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```

With the data and the model, we now train the algorithm 500 times

```python
# Train the model
model.fit(xs, ys, epochs=500)
```

Now with the model we are going to make a prediction

```python
# Make a prediction
print(model.predict([10.0]))
```

neural networks deal with probabilities.

With 6 data points, the network guess highly probable 2x+1 but not in certain.

With millions of data points just would estimate the probability closer to 1.