import numpy as np
from keras import backend as K
from keras.engine import InputSpec
from keras.initializers import Constant
from keras.layers import Dense


class Sparse(Dense):
    def __init__(self,
                 adjacency_mat=None,
                 # Specifies which inputs (rows) are connected to which outputs
                 # (columns)
                 *args,
                 **kwargs):
        self.adjacency_mat = adjacency_mat
        if adjacency_mat is not None:
            units = adjacency_mat.shape[1]
        super().__init__(units=units, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        connection_vector = (
            np.sum(self.adjacency_mat, axis=0) > 0
        ).astype(int)
        if np.sum(connection_vector) < self.adjacency_mat.shape[1]:
            print('Warning: not all nodes in the Sparse layer are ' +
                  'connected to inputs! These nodes will always have zero ' +
                  'output.')

        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=lambda shape: K.constant(
                self.adjacency_mat) *
            self.kernel_initializer(shape),

            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=lambda shape: K.constant(connection_vector) *
                self.bias_initializer(shape),

                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

            self.bias_adjacency_tensor = self.add_weight(
                shape=(self.units,),
                initializer=Constant(connection_vector),
                name='bias_adjacency_matrix',
                trainable=False)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        # Ensure we set weights to zero according to adjancency matrix
        self.adjacency_tensor = self.add_weight(
            shape=(input_dim, self.units),
            initializer=Constant(self.adjacency_mat),
            name='adjacency_matrix',
            trainable=False)
        self.built = True

    def call(self, inputs):
        output = self.kernel * self.adjacency_tensor
        output = K.dot(inputs, output)
        if self.use_bias:
            output = K.bias_add(output, self.bias_adjacency_tensor * self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def count_params(self):
        num_weights = 0
        if self.use_bias:
            bias_weights = np.sum(
                (np.sum(self.adjacency_mat, axis=0) > 0).astype(int))
            num_weights += bias_weights
        num_weights += np.sum(self.adjacency_mat)
        return num_weights

    def get_config(self):
        config = {
            'adjacency_mat': self.adjacency_mat.tolist()
        }
        base_config = super().get_config()
        base_config.pop('units', None)
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        adjacency_mat_as_list = config['adjacency_mat']
        config['adjacency_mat'] = np.array(adjacency_mat_as_list)
        return cls(**config)
