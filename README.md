# sparsely-connected-keras
Sparsely-connected layers for Keras

Install with pip:
```
pip install sparsely-connected-keras
```

Import and use in your code:
```
from sparsely_connected_keras import Sparse
sparse_layer = Sparse(adjacency_mat=adj)
```

Note: for deserializing a Keras model that contains a Sparse layer,
you must specify `custom_objects`:
```
model = load_model("example.h5", custom_objects={"Sparse": Sparse})
```
