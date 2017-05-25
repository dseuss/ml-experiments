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

    def _setup_progressbar(self, epochs, metrics, validation=False):
        widgets = [pb.SimpleProgress(), '    ']
        for metric in metrics:
            widgets += [pb.DynamicMessage(metric), '   ']

        if validation:
            for metric in metrics:
                widgets += [pb.DynamicMessage('val_' + metric), '   ']

        progress = pb.ProgressBar(max_value=epochs, widgets=widgets,
                                  term_with=60)
        return progress

    @_check_init
    def train(self, x, y, epochs=1, batch_size=None, metrics=['loss'],
              validation_data=None):
        indices = np.arange(len(x))
        batch_size = batch_size if batch_size is not None else len(x)
        np.random.shuffle(indices)
        nr_chucks = len(list(chunks(indices, batch_size)))
        val_ops = [self[key] for key in metrics]

        if validation_data is not None:
            x_test, y_test = validation_data
            with_validation = True
        else:
            with_validation = False

        for epoch in range(epochs):
            progress = self._setup_progressbar(nr_chucks, metrics,
                                               validation=with_validation)

            for batch_nr, batch_idx in enumerate(chunks(indices, batch_size)):
                feed_dict = {self['input']: x[batch_idx],
                             self['reference']: y[batch_idx]}
                _, *metric = self.session.run([self['optimization']] + val_ops,
                                              feed_dict=feed_dict)
                for key, value in zip(metrics, metric):
                    progress.dynamic_messages[key] = value
                progress.update(batch_nr)

            if with_validation:
                feed_dict = {self['input']: x_test,
                                self['reference']: y_test}
                metric = self.session.run(val_ops, feed_dict=feed_dict)
                for key, value in zip(metrics, metric):
                    progress.dynamic_messages['val_' + key] = value

            progress.finish()
