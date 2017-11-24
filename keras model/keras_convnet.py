# Python 3.6.0
# tensorflow 1.1.0
# Keras 2.0.4

import os
import os.path as path
import numpy as np
import cv2

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'mnist_convnet'
EPOCHS = 5
BATCH_SIZE = 64


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*28*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    for i in range(len(x_train)):
        x_train[i] = deskew(x_train[i])[:, :, np.newaxis]
    for i in range(len(x_test)):
        x_test[i] = deskew(x_test[i])[:, :, np.newaxis]
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Fully connected layer
    model.add(Dense(512))
    BatchNormalization()
    model.add(Activation('relu'))
    model.add(Dense(10))

    model.add(Activation('softmax'))
    return model


def train(model, train_generator, test_generator):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=EPOCHS,
                        validation_data=test_generator, validation_steps=10000//64)


def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out',
                         MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None,
                              False, 'out/' + MODEL_NAME + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def main():
    if not path.exists('out'):
        os.mkdir('out')

    x_train, y_train, x_test, y_test = load_data()

    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                             height_shift_range=0.08, zoom_range=0.08)
    test_gen = ImageDataGenerator()

    train_generator = gen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    test_generator = test_gen.flow(x_test, y_test, batch_size=BATCH_SIZE)

    model = build_model()

    train(model, train_generator, test_generator)

    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "activation_6/Softmax")


if __name__ == '__main__':
    main()
