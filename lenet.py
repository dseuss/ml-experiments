# encoding: utf-8

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Lambda
from keras.optimizers import Adam
import h5py


keras.backend.set_image_dim_ordering('th')


def generate_model(figsize, nr_classes, cunits=[20, 50], fcunits=[500]):
    model = Sequential()
    cunits = list(cunits)

    model.add(Convolution2D(cunits[0], 5, 5, border_mode='same', activation='relu',
                            input_shape=(1,) + figsize))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layers
    for nr_units in cunits[1:]:
        model.add(Convolution2D(nr_units, 5, 5, border_mode='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    for nr_units in fcunits:
        model.add(Dense(nr_units, activation='relu'))

    # Output layer
    model.add(Dense(nr_classes, activation='softmax'))

    return model


NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train / 255).astype('float32')
x_test = (x_test / 255).astype('float32')
y_train = keras.utils.np_utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.np_utils.to_categorical(y_test, NUM_CLASSES)

model = generate_model(x_train.shape[1:], NUM_CLASSES)
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


