"""Backend functions."""

import tensorflow as tf


def safer_sqrt(tensor, epsilon=1e-8):
    "A safer sqrt of a tensor"
    return tf.sqrt(tf.maximum(tensor, epsilon))


def safer_norm(tensor, axis=None, keep_dims=False, epsilon=1e-8):
    "A safer norm of a tensor"
    sq_tensor = tf.square(tensor)
    sum_squares = tf.reduce_sum(sq_tensor, axis=axis, keepdims=keep_dims)
    return tf.sqrt(tf.maximum(sum_squares, epsilon))
