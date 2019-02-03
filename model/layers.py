#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
# Author: lionel
import tensorflow as tf
from tensorflow import keras


class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1].value, self.attention_size),
                                 initializer="uniform",
                                 trainable=True)

        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(self.attention_size,),
                                 initializer="zeros",
                                 trainable=True)

        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="uniform",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        et = keras.backend.tanh(keras.backend.dot(inputs, self.W) + self.b)

        at = keras.backend.softmax(keras.backend.squeeze(keras.backend.dot(et, self.u), axis=-1))

        if mask is not None:
            at *= keras.backend.cast(mask, keras.backend.floatx)
        atx = keras.backend.expand_dims(at, axis=-1)
        ot = atx * inputs

        output = keras.backend.tanh(ot)
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.attention_size
        return tf.TensorShape(shape)

    def get_config(self):
        config = {'attention_size': self.attention_size}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OutPutLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OutPutLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OutPutLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        result = tf.reduce_sum(tf.multiply(inputs[0], inputs[1]), axis=1)
        a_ = tf.reduce_sum(tf.multiply(inputs[0], inputs[1]), axis=1)
        b_ = tf.reduce_sum(tf.multiply(inputs[0], inputs[1]), axis=1)
        prediction = tf.div(result, tf.multiply(a_, b_))
        prediction = tf.subtract(tf.ones_like(prediction), tf.rint(prediction), name='prediction')
        return prediction

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.attention_size
        return tf.TensorShape(shape)
