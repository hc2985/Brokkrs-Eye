"""
Minimal model.py - Only includes Model_A which is actually used
"""
import keras.backend as K
from tensorflow.keras import backend as k

# Compatibility: use built-in mish activation if tensorflow_addons not available
try:
    import tensorflow_addons as tfa
except ImportError:
    # Create a simple namespace for compatibility
    class TFACompat:
        class activations:
            @staticmethod
            def mish(x):
                import tensorflow as tf
                return tf.keras.activations.mish(x)
    tfa = TFACompat()

import tensorflow as tf

from keras.layers import *
from tensorflow import keras

import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'

def AttLayer(q):
    """Attitude normalization layer"""
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)
    normalized_array = q/tf.norm(q, axis=1, keepdims=True)
    normalized_array = tf.cast(normalized_array, tf.float64)
    w, x, y, z = tf.split(normalized_array, 4, axis=-1)
    roll = tf.math.atan2(2*(w*x + y*z),
                       (1-2*(x*x + y*y)))
    pitch = tf.math.asin(2*(w*y - x*z))
    zero_float64 = tf.constant(0.0, dtype=tf.float64)
    qx = (tf.math.sin(roll/2) * tf.math.cos(pitch/2) * tf.math.cos(zero_float64) -
          tf.math.cos(roll/2) * tf.math.sin(pitch/2) * tf.math.sin(zero_float64))
    qx = tf.reshape(qx, (tf.shape(roll)[0], 1))
    qy = (tf.math.cos(roll/2) * tf.math.sin(pitch/2) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.cos(pitch/2) * tf.math.sin(zero_float64))
    qy = tf.reshape(qy, (tf.shape(roll)[0], 1))
    qz = (tf.math.cos(roll/2) * tf.math.cos(pitch/2) * tf.math.sin(zero_float64) -
          tf.math.sin(roll/2) * tf.math.sin(pitch/2) * tf.math.cos(zero_float64))
    qz = tf.reshape(qz, (tf.shape(roll)[0], 1))
    qw = (tf.math.cos(roll/2) * tf.math.cos(pitch/2) * tf.math.cos(zero_float64) +
          tf.math.sin(roll/2) * tf.math.sin(pitch/2) * tf.math.sin(zero_float64))
    qw = tf.reshape(qw, (tf.shape(roll)[0], 1))
    quat = tf.concat([qw, qx, qy, qz], axis=-1)
    return quat

def Model_A(window_size):
    """Main model used for IMU processing"""
    D = 256
    Gn = 0.25
    acc = Input((window_size, 3), name='Acc')
    Acc = GaussianNoise(Gn)(acc)
    Acc1 = Lambda(lambda x: x[:, :, 0], name='Acc1')(Acc)
    Acc1 = Reshape((Acc1.shape[1], 1), name='ReshapeAcc1')(Acc1)
    Acc2 = Lambda(lambda x: x[:, :, 1], name='Acc2')(Acc)
    Acc2 = Reshape((Acc2.shape[1], 1), name='ReshapeAcc2')(Acc2)
    Acc3 = Lambda(lambda x: x[:, :, 2], name='Acc3')(Acc)
    Acc3 = Reshape((Acc3.shape[1], 1), name='ReshapeAcc3')(Acc3)

    conv1Acc = Conv1D(128, 11, padding="same",
                      activation=tfa.activations.mish, name='Conv1Acc1')(Acc1)
    pool1acc = MaxPooling1D(3, name='MaxPool1Acc')(conv1Acc)

    conv2Acc = Conv1D(128, 11, padding="same",
                      activation=tfa.activations.mish, name='Conv2Acc1')(Acc2)
    pool2acc = MaxPooling1D(3, name='MaxPool2Acc')(conv2Acc)

    conv3Acc = Conv1D(128, 11, padding="same",
                      activation=tfa.activations.mish, name='Conv3Acc1')(Acc3)
    pool3acc = MaxPooling1D(3, name='MaxPool3Acc')(conv3Acc)

    concAcc = concatenate(
        [pool1acc, pool2acc, pool3acc], name='ConcatenateCNN1')

    gyro = Input((window_size, 3), name='Gyro')
    Gyro = GaussianNoise(Gn)(gyro)
    Gyro1 = Lambda(lambda x: x[:, :, 0], name='Gyro1')(Gyro)
    Gyro1 = Reshape((Gyro1.shape[1], 1), name='ReshapeGyro1')(Gyro1)
    Gyro2 = Lambda(lambda x: x[:, :, 1], name='Gyro2')(Gyro)
    Gyro2 = Reshape((Gyro2.shape[1], 1), name='ReshapeGyro2')(Gyro2)
    Gyro3 = Lambda(lambda x: x[:, :, 2], name='Gyro3')(Gyro)
    Gyro3 = Reshape((Gyro3.shape[1], 1), name='ReshapeGyro3')(Gyro3)

    conv1Gyro = Conv1D(128, 11, padding="same",
                       activation=tfa.activations.mish, name='Conv1Gyro1')(Gyro1)
    pool1gyro = MaxPooling1D(3, name='MaxPool1Gyro')(conv1Gyro)

    conv2Gyro = Conv1D(128, 11, padding="same",
                       activation=tfa.activations.mish, name='Conv2Gyro1')(Gyro2)
    pool2gyro = MaxPooling1D(3, name='MaxPool2Gyro')(conv2Gyro)

    conv3Gyro = Conv1D(128, 11, padding="same",
                       activation=tfa.activations.mish, name='Conv3Gyro1')(Gyro3)
    pool3gyro = MaxPooling1D(3, name='MaxPool3Gyro')(conv3Gyro)

    concGyro = concatenate(
        [pool1gyro, pool2gyro, pool3gyro], name='ConcatenateCNN2')
    AGconc = concatenate([Acc, Gyro])
    AGconc = Bidirectional(LSTM(128, return_sequences=True))(AGconc)
    AGconc = Dropout(0.2)(AGconc)
    AGconc = Dense(D, activation=tfa.activations.mish)(AGconc)
    AGconc = Flatten()(AGconc)

    conc = concatenate([concAcc, concGyro], name='ConcatenateCNN')
    conc = Conv1D(128, 11, padding="same",
                  activation=tfa.activations.mish, name='Conv1')(conc)
    conc = GaussianNoise(Gn)(conc)
    conc = Dense(D, activation=tfa.activations.mish)(conc)
    conc = Flatten()(conc)

    fs = Input((1,), name='Fs')
    fsDense = Dense(D, activation=tfa.activations.mish)(fs)

    concat = concatenate([conc, fsDense, AGconc])
    quat = Dense(4, activation="linear", name='Quat')(concat)
    quat = Lambda(lambda x: k.l2_normalize(x, axis=1), name='QuatNorm')(quat)
    quat = Lambda(lambda x: AttLayer(x), name='Attitude')(quat)
    model = keras.Model(inputs=[Acc, Gyro, fs], outputs=[quat])
    model.summary()
    return model
