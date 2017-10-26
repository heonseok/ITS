import gym
import numpy as np
import random

#from model import *

class Environment(object):
    def __init__(self, args):
        self.args = args

        
class SimpleEnvironment(Environment):
    def __init__(self, args):
        super(SimpleEnvironment, self).__init__(args)
        self.env = gym.make(self.args.env_name)
        self.num_actions = self.env.action_space.n
        self.state_shape = list(self.env.observation_space.shape)

    def new_episode(self):
        return self.env.reset()

    def act(self, action):
        self.state, self.reward, self.terminal, _ = self.env.step(action)

        return self.state, self.reward, self.terminal

    def random_action(self):
        return self.env.action_space.sample()

class DKVMNEnvironment(Environment):
    def __init__(self, args, sess, dkvmn):
        super(DKVMNEnvironment, self).__init__(args)

        self.sess = sess

        self.env = dkvmn 
        dkvmn.args.seq_len = 1
        dkvmn.args.batch_size = 1

        self.env.print_info()
        self.state_shape = self.env.get_value_memory_shape()
        print('State shape')
        print(self.state_shape)
        self.num_actions = self.env.get_n_questions()

    def new_episode(self):
        print('\nnew_episode is not implemented\n')
        return False

    def act(self, action):
        #print('\nact is not implemented\n')

        #action = np.array(action, type=np.int32)
        action = np.array(action)
        action.astype(np.int32)
        action = np.expand_dims(action, axis=0)
        action = np.expand_dims(action, axis=0)
        print('ACTION SHAPE')
        print(action.shape)
        print(action.dtype)

        #self.state = self.sess.run(self.env.updated_value_memory, self.env.value_memory_difference, feed_dict={self.env.q_data_seq:action})
        self.reward = self.sess.run(self.env.value_memory_difference, feed_dict={self.env.q_data_seq:action})
        print('REWARD : ')
        print(reward)


    def random_action(self):
        return random.randrange(1, self.num_actions+1)
        
