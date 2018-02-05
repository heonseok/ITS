import numpy as np
import tensorflow as tf
import os

# x : input(batch * state_dim)
def linear(x, state_dim, name='linear'):
    tf.set_random_seed(224)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:

        weight = tf.get_variable('weight', [x.get_shape()[-1], state_dim], initializer=tf.random_normal_initializer(stddev=0.1))
        #weight = tf.get_variable('weight', [x.get_shape()[-1], state_dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        #bias = tf.get_variable('bias', [state_dim], initializer=tf.random_normal_initializer(0.1))
        bias = tf.get_variable('bias', [state_dim], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(x, weight) + bias
        return weighted_sum
        
