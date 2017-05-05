import numpy as np
from keras import backend
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


def preprocess(x, y):
    sel = np.ravel((y == 3) + (y == 5))
    x, y = x[sel].astype(np.double), y[sel]
    x = x / 255.
    y = (y == 3).astype(np.uint8).ravel()
    return x, y


def generate_model(img_shape=(32, 32, 3)):
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

    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train = preprocess(x_train, y_train)
x_test, y_test = preprocess(x_test, y_test)

backend.set_image_data_format('channels_last')
model = generate_model()
model.compile(loss='binary_crossentropy', optimizer='adam',
              metric=['accuracy'])
model.summary()

save_callback = ModelCheckpoint('cifar10.h5', verbose=1, period=10)
tb_callback = TensorBoard()
lr_callback = ReduceLROnPlateau(factor=0.1, verbose=1, cooldown=5)

train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_datagen.fit(x_train)

model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=64),
                    validation_data=(x_test, y_test),
                    verbose=1, epochs=1000, steps_per_epoch=100,
                    callbacks=[save_callback, tb_callback, lr_callback]
                    )
