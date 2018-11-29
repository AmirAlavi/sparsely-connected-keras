# import pdb; pdb.set_trace()

import numpy as np
np.random.seed(42)
from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.utils import to_categorical, plot_model

from sparsely_connected_keras import Sparse
input_dim = 3
hidden_size = 5
# adjacency matrix specifiying the connections between
# the input layer and the hidden layer (not a dense MLP)
adj = np.array([[0, 1, 1, 0, 0],
                [1, 1, 0, 0, 1],
                [1, 0, 1, 0, 1]])
print("Adjacency matrix:")
print(adj)
print()
orig = adj
sparse_layer = Sparse(adjacency_mat=adj, kernel_initializer='random_uniform', bias_initializer='ones')
inputs = Input(shape=(input_dim,))
x = sparse_layer(inputs)
model = Model(inputs=inputs, outputs=x)
model.compile(
    loss='mse',
    optimizer='sgd')
print()
print(model.summary())
print()

# Sample inputs
num_samples = 100
X = np.random.rand(num_samples, input_dim)
y = np.random.rand(num_samples, hidden_size)
print("Sparse weights before training:")
names = [weight.name for weight in model.layers[1].weights]
for n, v in zip(names, model.layers[1].get_weights()):
    print(n)
    print(v)
print()
