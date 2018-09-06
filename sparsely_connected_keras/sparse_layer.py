import numpy as np
from keras import backend as K
from keras.engine import InputSpec
from keras.initializers import Initializer, VarianceScaling, _compute_fans
from keras.layers import Dense


# For now, only Glorot initializers are supported for the weight matrix of a
# Sparse layer. Whatever the user specifies for 'kernel_initializer' is
# ignored.
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
            return clipped * self.adjacency_mat
        else:
            limit = np.sqrt(3. * scale)
            dense_initial = np.random.uniform(
                low=-limit, high=limit, size=shape)
            return dense_initial * self.adjacency_mat

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


class AdjacencyInitializer(Initializer):
    def __init__(self, adjacency_mat=1):
        # Default value is 1 which translates to a dense (fully connected)
        # layer
        self.adjacency_mat = adjacency_mat

    def __call__(self, shape, dtype=None):
        return K.constant(self.adjacency_mat, shape=shape, dtype=dtype)

    def get_config(self):
        return {'adjacency_mat': self.adjacency_mat}


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

        self.kernel = self.add_weight(
            shape=(
                input_dim,
                self.units),
            initializer=SparseGlorotInitializer(
                adjacency_mat=self.adjacency_mat,
                scale=1.,
                mode='fan_avg',
                distribution='uniform'),
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        # Ensure we set weights to zero according to adjancency matrix
        self.adjacency_tensor = self.add_weight(
            shape=(
                input_dim,
                self.units),
            initializer=AdjacencyInitializer(
                self.adjacency_mat),
            name='adjacency_matrix',
            trainable=False)
        self.built = True

    def call(self, inputs):
        output = self.kernel * self.adjacency_tensor
        output = K.dot(inputs, output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def count_params(self):
        num_weights = 0
        if self.use_bias:
            num_weights += self.units
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
