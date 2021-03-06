# encoding: utf-8

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.backend import image_data_format


def generate(figsize, nr_classes, cunits=[20, 50], fcunits=[500]):
    model = Sequential()
    cunits = list(cunits)
    input_shape = figsize + (1,) if image_data_format == 'channels_last' \
        else (1,) + figsize

    model.add(Conv2D(cunits[0], (5, 5), padding='same',
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Convolutional layers
    for nr_units in cunits[1:]:
        model.add(Conv2D(nr_units, (5, 5), padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    for nr_units in fcunits:
        model.add(Dense(nr_units, activation='relu'))

    # Output layer
    activation = 'softmax' if nr_classes > 1 else 'sigmoid'
    model.add(Dense(nr_classes, activation=activation))

    return model
