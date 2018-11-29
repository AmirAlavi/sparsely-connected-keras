# import pdb; pdb.set_trace()

import numpy as np
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
sparse_layer = Sparse(adjacency_mat=adj)
inputs = Input(shape=(input_dim,))
x = sparse_layer(inputs)
model = Model(inputs=inputs, outputs=x)
#plot_model(model, to_file="sparse_test_architecture.png", show_shapes=True)
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
model.fit(X, y, epochs=5, batch_size=10)
print("Sparse weights after training:")
for n, v in zip(names, model.layers[1].get_weights()):
    print(n)
    print(v)
print()
model.fit(X, y, epochs=5, batch_size=10)
print("Sparse weights after even more training:")
for n, v in zip(names, model.layers[1].get_weights()):
    print(n)
    print(v)
print()
print("Check outputs of predict...")
print(model.predict(X[:10]))
model.save("testing_sparse_save.h5")
print("\nSAVING MODEL, DELETING IN-MEMORY MODEL\n")
del model
del adj
print("LOADING MODEL FROM DISK")
print()
model = load_model("testing_sparse_save.h5", custom_objects={'Sparse': Sparse})
print(model.summary())
print()
print("Sparse weights after loading more training:")
for n, v in zip(names, model.layers[1].get_weights()):
    print(n)
    print(v)
print()
print(model.layers[1].adjacency_mat)
print()
print(orig == model.layers[1].adjacency_mat)
