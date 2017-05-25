import tensorflow as tf
from .supervised import SupervisedModel


class LinearRegressor(SupervisedModel):
    """Docstring for LinearRegressor. """

    def __init__(self, dims_in, dims_out, **kwargs):
        """@todo: to be defined1.

        :param input_dims: @todo
        :param batch_size: @todo

        """
        super().__init__((dims_in,), (dims_out,), **kwargs)
        self['slope'] = tf.Variable(name='slope', dtype=self.dtype,
                                    initial_value=tf.random_normal((dims_in, dims_out)))
        self['intercept'] = tf.Variable(name='intercept', dtype=self.dtype,
                                        initial_value=tf.zeros((dims_out,)))
        self.build()

    def predictor(self, x):
        return tf.matmul(x, self['slope']) + self['intercept']

    def loss_function(self, y_ref, y_pred):
        return tf.reduce_mean(tf.square(y_ref - y_pred))
