from abc import abstractmethod

import progressbar as pb
import tensorflow as tf
import numpy as np


# DECORATORS
def _check_init(method):
    def check(self, *args, **kwargs):
        if not self.is_init:
            self.init()
        return method(self, *args, **kwargs)
    return check


def chunks(array, n):
    """Returns an iterator over successive chunks of `array` of size `n`"""
    for i in range(0, len(array), n):
        yield array[i:i + n]


# CLASSES
class SupervisedModel(object):
    """Docstring for SupervisedModel. """

    def __init__(self, input_shape, output_shape, dtype=tf.float32,
                 optimizer=tf.train.AdamOptimizer()):
        """@todo: to be defined1.

        :param input_shape: @todo
        :param output_shape: @todo
        :param optimizier: @todo

        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dtype = dtype
        self.optimizer = optimizer
        self._vars = dict()
        self._init_session = None

    @abstractmethod
    def predictor(self, x):
        pass

    @abstractmethod
    def loss_function(self, y_ref, y_pred):
        pass

    @property
    def is_init(self):
        return self._init_session is self.session

    def __setitem__(self, key, value):
        self._vars[key] = value

    def __getitem__(self, key):
        return self._vars[key]

    def build(self):
        self['input'] = tf.placeholder(self.dtype, name='input',
                                       shape=(None,) + self.input_shape)
        self['prediction'] = self.predictor(self['input'])
        self['reference'] = tf.placeholder(self.dtype, name='reference',
                                           shape=(None,) + self.input_shape)
        self['loss'] = self.loss_function(self['reference'], self['prediction'])
        self['optimization'] = self.optimizer.minimize(self['loss'])

    def init(self):
        self.session.run(tf.global_variables_initializer())
        self._init_session = self.session

    @property
    def session(self):
        return tf.get_default_session()

    @_check_init
    def predict(self, x):
        feed_dict = {self['input']: x}
        return self.session.run(self['prediction'], feed_dict=feed_dict)

    @_check_init
    def evaluate(self, x, y):
        feed_dict = {self['input']: x, self['reference']: y}
        return self.session.run(self['loss'], feed_dict=feed_dict)

    @_check_init
    def train(self, x, y, epochs=1, batch_size=None):
        progress = pb.ProgressBar(max_value=epochs)
        indices = np.arange(len(x))
        batch_size = batch_size if batch_size is not None else len(x)
        np.random.shuffle(indices)

        for epoch in range(epochs):
            for batch_idx in chunks(indices, batch_size):
                feed_dict = {self['input']: x[batch_idx],
                             self['reference']: y[batch_idx]}
                _, loss = self.session.run([self['optimization'], self['loss']],
                                        feed_dict=feed_dict)
            progress.update(epoch)

        progress.finish()
