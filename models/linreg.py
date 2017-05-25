import tensorflow as tf
import numpy as np
from collections import namedtuple


def _check_init(method):
    def check(self, *args, **kwargs):
        if not self.is_init:
            self.init()
        return method(self, *args, **kwargs)
    return check


class LinearRegressor(object):
    """Docstring for LinearRegressor. """

    def __init__(self, dims_in, dims_out):
        """@todo: to be defined1.

        :param input_dims: @todo
        :param batch_size: @todo

        """
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.dtype = tf.float32

        self.weights = {
            'slope': tf.Variable(name='slope', initial_value=tf.random_normal((dims_in, dims_out))),
            'intercept': tf.Variable(name='intercept', initial_value=tf.zeros((dims_out,))),
        }
        self.expressions = dict()
        self.build()

    @property
    def sess(self):
        return tf.get_default_session()

    @property
    def is_init(self):
        return all(self.sess.run(tf.is_variable_initialized(w))
                   for w in self.weights.values())

    def build(self):
        x = tf.placeholder(self.dtype, shape=(None, self.dims_in),
                           name='x')
        self.expressions['input'] = x
        y_pred = tf.matmul(x, self.weights['slope']) + self.weights['intercept']
        y_ref = tf.placeholder(self.dtype, shape=(None, self.dims_out),
                               name='y_ref')
        self.expressions['prediction'] = y_pred
        self.expressions['reference'] = y_ref

        self.expressions['loss'] = tf.reduce_mean(tf.square(y_ref - y_pred))

        optimizier = tf.train.AdamOptimizer()
        self.program = optimizier.minimize(self.expressions['loss'])

    def init(self):
        print("Initializing variables")
        self.sess.run(tf.global_variables_initializer())

    @_check_init
    def predict(self, x):
        feed_dict = {self.expressions['input']: x}
        return self.sess.run(self.expressions['prediction'], feed_dict=feed_dict)

    @_check_init
    def evaluate(self, x, y):
        feed_dict = {self.expressions['input']: x,
                     self.expressions['reference']: y}
        return self.sess.run(self.expressions['loss'], feed_dict=feed_dict)

    @_check_init
    def train(self, x, y, epochs=1, batch_size=None):

        for epoch in range(epochs):
            feed_dict = {self.expressions['input']: x,
                         self.expressions['reference']: y}
            _, loss = self.sess.run([self.program, self.expressions['loss']],
                                    feed_dict=feed_dict)
            print(f"Finished {epoch}/{epochs} with loss {loss}")


if __name__ == '__main__':
    import matplotlib.pyplot as pl
    SIGMA = 0.05
    SAMPLES = 100

    slope, intercept = np.random.randn(2)
    x_train = np.random.uniform(low=-1.0, high=1.0, size=(SAMPLES))
    y_train = slope * x_train + intercept
    y_train += SIGMA * np.random.randn(*y_train.shape)
    x_test = np.linspace(min(x_train), max(x_train), 100, endpoint=True)
    y_test = slope * x_test + intercept

    model = LinearRegressor(dims_in=1, dims_out=1)
    with tf.Session().as_default():
        model.train(x_train[:, None], y_train[:, None], epochs=10000)
        y_pred = model.predict(x_test[:, None])
        print('Predicted loss:', model.evaluate(x_train[:, None], y_train[:, None]))

    pl.scatter(x_train, y_train)
    pl.plot(x_test, y_test)
    pl.plot(x_test, y_pred)
    pl.show()
