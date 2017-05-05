import sys

import numpy as np
from keras import backend
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from models import vgg_notop


def preprocess(x, y):
    sel = np.ravel((y == 3) + (y == 5))
    x, y = x[sel].astype(np.double), y[sel]

    # normalize to zero-mean pixels
    pixel_means = np.mean(x, axis=(0, 1, 2), keepdims=True)
    x -= pixel_means

    # RGB -> BGR
    x = x[..., ::-1]

    if backend.image_data_format == 'channels_first':
        x = np.transpose(x, (0, 3, 1, 2))

    y = (y == 3).astype(np.int).ravel()
    return x, y


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    return (x_train, y_train), (x_test, y_test)


def generate_vgg_reduced(img_shape=(32, 32, 3)):
    print("Generating new VGG-style model")
    conv_params=dict(activation='relu', padding='same')
    model = Sequential()

    model.add(Conv2D(32, (3, 3), name='block1_conv1', input_shape=img_shape,
                     **conv_params))
    model.add(Conv2D(32, (3, 3), name='block1_conv2', **conv_params))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(Conv2D(64, (3, 3), name='block2_conv1', **conv_params))
    model.add(Conv2D(64, (3, 3), name='block2_conv2', **conv_params))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(Conv2D(128, (3, 3), name='block3_conv1', **conv_params))
    model.add(Conv2D(128, (3, 3), name='block3_conv2', **conv_params))
    model.add(Conv2D(128, (3, 3), name='block3_conv3', **conv_params))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def generate_vgg16_transfer(img_shape=(32, 32, 3)):
    print("Using imagenet pretrained VGG16 model")
    base_model = vgg_notop.VGG16(weights='imagenet', input_shape=img_shape)
    x = base_model.get_layer('block3_pool').output
    x = Flatten(name='flatten')(x)

    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)

    prediction = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.inputs, outputs=prediction)
    return model


def train_reduced():
    (x_train, y_train), (x_test, y_test) = load_data()
    backend.set_image_data_format('channels_last')

    model = generate_vgg_reduced()
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    save_callback = ModelCheckpoint('cifar10_reduced.h5', verbose=1, period=10)
    tb_callback = TensorBoard()
    lr_callback = ReduceLROnPlateau(factor=0.05, verbose=1, cooldown=5)

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=64),
                        validation_data=(x_test, y_test),
                        verbose=1, epochs=1000, steps_per_epoch=256,
                        callbacks=[save_callback, tb_callback, lr_callback]
                        )


def train_transfer():
    (x_train, y_train), (x_test, y_test) = load_data()
    backend.set_image_data_format('channels_last')
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    model = generate_vgg16_transfer()

    save_callback = ModelCheckpoint('cifar10_transfer.h5', verbose=1, period=10)
    tb_callback = TensorBoard()
    lr_callback = ReduceLROnPlateau(factor=0.05, verbose=1, cooldown=5)

    for layer in model.layers:
        if 'conv' in layer.name:
            print("Setting %s to untrainable" % layer.name)
            layer.trainable = False
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=64),
                        validation_data=(x_test, y_test),
                        verbose=1, epochs=50, steps_per_epoch=256,
                        )
    model.save_weights('cifar10_transfer_pretrain.h5')

    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=Adam(lr=0.0001))
    model.summary()
    model.load_weights('cifar10_transfer_pretrain.h5')

    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=64),
                        validation_data=(x_test, y_test),
                        verbose=1, epochs=1000, steps_per_epoch=256,
                        callbacks=[save_callback, tb_callback, lr_callback]
                        )

MODELS = {
    'reduced': train_reduced,
    'transfer': train_transfer
}

if __name__ == '__main__':
    MODELS.get(sys.argv[1])()
