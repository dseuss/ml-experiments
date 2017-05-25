from abc import abstractmethod

import progressbar as pb
import tensorflow as tf


## DECORATORS
def _check_init(method):
    def check(self, *args, **kwargs):
        if not self.is_init:
            self.init()
        return method(self, *args, **kwargs)
    return check


## CLASSES
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
        self.is_init = False
        self.optimizer = optimizer
        self._vars = dict()

    @abstractmethod
    def predictor(self, x):
        pass

    @abstractmethod
    def loss_function(self, y_ref, y_pred):
        pass

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
        self.sess.run(tf.global_variables_initializer())
        self.is_init = True

    @property
    def sess(self):
        return tf.get_default_session()

    @_check_init
    def predict(self, x):
        feed_dict = {self['input']: x}
        return self.sess.run(self['prediction'], feed_dict=feed_dict)

    @_check_init
    def evaluate(self, x, y):
        feed_dict = {self['input']: x, self['reference']: y}
        return self.sess.run(self['loss'], feed_dict=feed_dict)

    @_check_init
    def train(self, x, y, epochs=1):
        progress = pb.ProgressBar(max_value=epochs)

        for epoch in range(epochs):
            feed_dict = {self['input']: x, self['reference']: y}
            _, loss = self.sess.run([self['optimization'], self['loss']],
                                    feed_dict=feed_dict)
            progress.update(epoch)

        progress.finish()
