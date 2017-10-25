import numpy as np
import tensorflow as tf

from replay_memory import *
from environment import *

class DQN(object):
    def __init__(self, args, sess, memory, environment):
        self.args = args
        self.sess = sess
        self.memory = memory
        self.env = environment
        self.num_actions = self.env.num_actions
        self.input_shape = self.memory.state_shape


        self.kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        self.bias_initializer = tf.constant_initializer(0.05)

        self.states = tf.placeholder(tf.float32, [None] + self.input_shape)
        self.actions = tf.placeholder(tf.uint8, [None])
        self.rewards = tf.placeholder(tf.float32, [None])
        self.terminals = tf.placeholder(tf.float32, [None])
        self.max_q = tf.placeholder(tf.float32, [None])

        self.prediction_Q = self.build_network('pred')
        self.target_Q = self.build_network('target')

        self.loss, self.optimizer = self.build_optimizer()

    def build_network(self, name):
        with tf.variable_scope(name):
            fc1 = tf.layers.dense(inputs=self.states, units=10, activation=tf.nn.relu, kernel_initializer=self.kernel_initializer)
            Q = tf.layers.dense(inputs=fc1, units=self.num_actions, activation=None, kernel_initializer=self.kernel_initializer)
            return Q

    def build_optimizer(self):
        target_q = self.rewards + tf.multiply(1-self.terminals, tf.multiply(self.args.discount_factor, self.max_q))

        action_one_hot = tf.one_hot(indices=self.actions, depth=self.num_actions, on_value=1.0, off_value=0.0)
        pred_q = tf.reduce_sum(tf.multiply(self.prediction_Q, action_one_hot), reduction_indices=1)

        loss = tf.reduce_mean(tf.square(pred_q - target_q))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(loss)

        return loss, optimizer

        
    def update_target_network(self):
        copy_op = []
        pred_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for pred_var, target_var in zip(pred_vars, target_vars):
            copy_op.append(target_var.assign(pred_var.value()))
        self.sess.run(copy_op)

    def predict_Q_value(self, state):
        return self.sess.run(self.prediction_Q, feed_dict={self.states: [state]})

    def train_network(self):
        b_prestates, b_actions, b_rewards, b_terminals, b_poststates = self.memory.mini_batch() 

        b_q_poststates = self.sess.run(self.target_Q, feed_dict={self.states : b_poststates})
        b_max_q = np.max(b_q_poststates, axis=1)

        feeds = {self.states : b_prestates, self.actions : b_actions, self.rewards : b_rewards, self.terminals : b_terminals, self.max_q : b_max_q}
        return self.sess.run([self.loss, self.optimizer], feed_dict = feeds)
