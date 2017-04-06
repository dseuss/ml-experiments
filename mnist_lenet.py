#!/usr/bin/env python
# encoding: utf-8


import h5py
import keras
from keras.datasets import mnist
from keras.optimizers import Adam
from models import lenet

keras.backend.set_image_dim_ordering('th')


NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255).astype('float32')
x_test = (x_test / 255).astype('float32')
y_train = keras.utils.np_utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.np_utils.to_categorical(y_test, NUM_CLASSES)

model = lenet.generate(x_train.shape[1:], NUM_CLASSES)
optimizier = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizier,
              metrics=['accuracy'])

for super_epoch in range(20):
    model.fit(x_train[:, None, :, :], y_train, batch_size=128, nb_epoch=10,
              verbose=1, validation_data=(x_test[:, None, :, :], y_test))

    with h5py.File('lenet.h5', 'a') as output:
        try:
            del output[str(super_epoch)]
        except KeyError:
            pass
        model.save_weights_to_hdf5_group(output.create_group(str(super_epoch)))
