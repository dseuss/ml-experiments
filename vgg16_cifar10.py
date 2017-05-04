import numpy as np
from keras import backend
from keras.applications.vgg16 import VGG16, preprocess_input, WEIGHTS_PATH_NO_TOP
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file

from models.vgg16_modified import VGG16


TARGETFILE = 'data/vgg16_cifar10.py'


backend.set_image_data_format('channels_last')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
sel = np.ravel((y_train == 3) + (y_train == 5))
x_train, y_train = preprocess_input(x_train[sel].astype(np.double)), y_train[sel]
sel = np.ravel((y_test == 3) + (y_test == 5))
x_test, y_test = preprocess_input(x_test[sel].astype(np.double)), y_test[sel]
y_train = y_train == 3
y_test = y_test == 3

imggen = ImageDataGenerator(rotation_range=20,
                            width_shift_range=0.15,
                            height_shift_range=0.15,
                            shear_range=0.2,
                            fill_mode='constant',
                            cval=0.,
                            zoom_range=0.3,
                            channel_shift_range=0.1)
imggen.fit(x_train)
img_input = Input(shape=(32, 32, 3))

base_model = VGG16(include_top=False, weights='imagenet',
                   input_shape=x_train.shape[1:])

x = base_model.get_layer('block3_pool').output
x = Flatten(name='flatten')(x)
x = Dense(256, name='fc1', activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, name='fc2', activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.inputs, outputs=prediction)

# set the convolutional layers untrainable for now
for layer in base_model.layers:
    layer.trainable = False

optimizer = Adam(lr=1e-3, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
model.fit_generator(imggen.flow(x_train, y_train, batch_size=64),
                    validation_data=(x_test, y_test),
                    steps_per_epoch=100, epochs=50, verbose=True)

# and now train the full network
for layer in model.layers:
    layer.trainable = True

optimizer = Adam(lr=1e-4, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
save_callback = ModelCheckpoint(TARGETFILE, monitor='val_loss',
                                verbose=True, mode='auto', period=1)
model.fit_generator(imggen.flow(x_train, y_train, batch_size=64),
                    validation_data=(x_test, y_test),
                    steps_per_epoch=100, epochs=10000, verbose=True,
                    validation_steps=1000, callbacks=[save_callback])
