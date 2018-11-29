import numpy as np
from keras import backend as K
from keras.engine import InputSpec
from keras.initializers import VarianceScaling, Constant, _compute_fans
from keras.layers import Dense

# NOTE on kernel initializers: the purpose of the SparseLayer is to sparsely
# connect the input nodes to the layer, by setting certain weights to be zero.
# This is achieved by multiplying a constant binary 'adjacency matrix' by the
# kernel matrix, in every execution of __call__ prior to taking the dot product.
# Because of this, you might be tempted to think that it is ok if the kernel matrix
# has non-zero values where the adjacency matrix has a zero value (since the elementwise
# multiplication of W with the adjanceny matrix will zero-out those values anyways).
# However, for things like regularization, the values actually do matter. By using
# a sparse layer, we are effectivly simplifying the model, and regularization methods
# should take this into account correctly. If we have non-zero values sitting
# around in the kernel where they shouldn't, regularizers would incorrectly penalize
# us for those weights.
# Thus, we should guarantee that those values are indeed zero. The only way to do this
# is to initially set them to zero. If we initialize them to zero, they will stay at zero.
# Thus, we need some custom initializers.

class SparseGlorotInitializer(VarianceScaling):
    def __init__(self, adjacency_mat=None, *args, **kwargs):
        self.adjacency_mat = adjacency_mat
        super().__init__(*args, **kwargs)

    def __call__(self, shape, dtype=None):
        # The only difference between this and the built-in VarianceScaling is that
        # we multiply element-wise by our adjacency matrix as the final step.
        fan_in, fan_out = _compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        final = None
        if self.distribution == 'normal':
            mean = 0.0
            stddev = np.sqrt(scale)
            normal_mat = np.random.normal(loc=mean, scale=stddev, size=shape)
            clipped = np.clip(normal_mat, mean - 2 * stddev, mean + 2 * stddev)
            #return clipped * self.adjacency_mat
            return clipped
        else:
            limit = np.sqrt(3. * scale)
            dense_initial = np.random.uniform(
                low=-limit, high=limit, size=shape)
            print("using uniform dist")
            #return dense_initial * self.adjacency_mat
            return dense_initial

    def get_config(self):
        config = {
            'adjacency_mat': self.adjacency_mat.tolist()
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        adjacency_mat_as_list = config['adjacency_mat']
        config['adjacency_mat'] = np.array(adjacency_mat_as_list)
        super().from_config(cls, config)
        # return cls(**config)

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

        connection_vector = (np.sum(self.adjacency_mat, axis=0) > 0).astype(int)
        if np.sum(connection_vector) < self.adjacency_mat.shape[1]:
            print('Warning: not all nodes in the Sparse layer are ' +
                  'connected to inputs! These nodes will always have zero ' +
                  'output.')

        print("initializer: ", self.kernel_initializer)
        self.kernel = self.add_weight(
            shape=(
                input_dim,
                self.units),
            initializer=lambda shape: K.constant(self.adjacency_mat) * self.kernel_initializer(shape),
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        print(type(self.kernel))
        self.kernel = self.kernel * self.adjacency_mat
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=lambda shape: K.constant(connection_vector) * self.bias_initializer(shape),
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
            shape=(
                input_dim,
                self.units),
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
            bias_weights = np.sum((np.sum(self.adjacency_mat, axis=0) > 0).astype(int))
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
