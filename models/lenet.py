# encoding: utf-8

from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential


def generate(figsize, nr_classes, cunits=[20, 50], fcunits=[500]):
    model = Sequential()
    cunits = list(cunits)

    model.add(Convolution2D(cunits[0], 5, 5, border_mode='same',
                            activation='relu', input_shape=(1,) + figsize))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layers
    for nr_units in cunits[1:]:
        model.add(Convolution2D(nr_units, 5, 5, border_mode='same',
                                activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    for nr_units in fcunits:
        model.add(Dense(nr_units, activation='relu'))

    # Output layer
    model.add(Dense(nr_classes, activation='softmax'))

    return model
