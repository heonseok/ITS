import numpy as np
import tensorflow as tf
from tqdm import tqdm

import os 
from logger import *

from dqn import *
from environment import *
from replay_memory import *

import dkvmn_utils 

class DKVMNAgent():
    def __init__(self, args, sess, dkvmn):

        self.args = args
        self.sess = sess

        self.logger = dkvmn_utils.set_logger('DQN', 'policy.log', self.args.logging_level)
        self.logger.debug('Initializing AGENT')

        self.env = DKVMNEnvironment(args, sess, dkvmn, self.logger)
        self.memory = DKVMNMemory(args, self.env.state_shape)

        self.dqn = DQN(self.args, self.sess, self.memory, self.env)

        self.saver = tf.train.Saver()
        self.tb_logger = Logger(os.path.join(self.args.dqn_tb_log_dir, self.model_dir))

        '''
        self.logger.info('Trainalbe_variables of DKVMNAgent')
        for i in tf.trainable_variables():
            if "dkvmn" not in i.op.name:
                print(i.op.name)
        '''

        self.sess.run(tf.global_variables_initializer())
        dkvmn.load()

        self.dqn.update_target_network()

        self.logger.info('#'*120)
        self.logger.info(dkvmn.model_dir)
        self.logger.info('{:<20s}: {}'.format('Policy type', self.args.test_policy_type))


    def train(self):
        self.logger.debug('Agent is training')
        self.episode_count = 0
        best_reward = 0
        self.episode_reward = 0
        action_count = 0 
        episode_rewards = []

        self.logger.debug('===== Start to make random memory =====')
        self.reset_episode()
        for self.step in tqdm(range(1, self.args.max_step+1), ncols=70, initial=0):
            action = self.select_action()
            action_count += 1

            next_state, reward, terminal, mastery_lelvel, pred_prob = self.env.act(action)
            self.memory.add(action, reward, terminal, next_state)

            #self.tb_logger.log_scalar(tag='Episode%d:action'%self.episode_count, value=action, step=action_count)
            
            self.episode_reward += reward 
            if terminal:
                print('Terminal at %dth action' % action_count)
                self.episode_count += 1
                episode_rewards.append(self.episode_reward)
                if self.episode_reward > best_reward:
                    best_reward = self.episode_reward
                #self.tb_logger.log_scalar(tag='reward', value=self.episode_reward, step=self.step)
                #self.tb_logger.log_scalar(tag='{}_terminal_action_count'.format(self.args.test_policy_type), value=action_count, step=self.episode_count)
                self.reset_episode()
                action_count = 0

            if self.step >= self.args.training_start_step:
                if self.step == self.args.training_start_step:
                    self.logger.debug("===== Start to update the network =====")

                if self.step % self.args.train_interval == 0:
                    loss, _ = self.dqn.train_network()

                if self.step % self.args.copy_interval == 0:
                    self.dqn.update_target_network()

                if self.step % self.args.save_interval == 0:
                    self.save()

                if self.step % self.args.show_interval == 0:
                    avg_r = np.mean(episode_rewards)
                    max_r = np.max(episode_rewards)
                    min_r = np.min(episode_rewards)
                    if max_r > best_reward:
                        best_reward = max_r
                    #self.logger.debug('\n[recent %d episodes] avg_r: %.4f, max_r: %d, min_r: %d // Best: %d' % (len(episode_rewards), avg_r, max_r, min_r, best_reward))
                    episode_rewards = []
    


    def play(self, load=True):
        if load:
            if not self.load():
                exit()
        
        '''
        self.tr_vrbs = tf.trainable_variables()
        for i in self.tr_vrbs:
            if "dqn/pred" in i.op.name:
                print(i.op.name)
                print(i.shape)
                weights = self.sess.run(i)
                print(np.sum(weights))
                #for j in weights:
                    #print(np.sum(j))
        '''

        best_reward = 0
        action_count = 0
        action_count_list = [] 
    
        episode_access = []
        episode_correct_count = []
        episode_correct_rate = []

        episode_pos_terminal = list()
        episode_neg_terminal = list()
        episode_non_terminal = list()

        self.reset_episode()
        for episode in range(self.args.num_test_episode):
            current_reward = 0

            terminal = False
            while not terminal:
                action = self.select_action()
                action_count += 1
                #self.tb_logger.log_scalar(tag='Episode%d:action'%episode, value=action, step=action_count)
                next_state, reward, terminal, mastery_level, pred_prob = self.env.act(action)
                #self.tb_logger.log_scalar(tag='Episode%d:reward'%episode, value=reward, step=action_count)
                self.tb_logger.log_scalar(tag='Episode%d:mastery'%episode, value=np.sum(mastery_level), step=action_count)

                current_reward += reward
                if terminal:
                    action_count_list.append(action_count)
                    access, correct_count, correct_rate, pos_terminal, neg_terminal, non_terminal = self.reset_episode()
                    episode_access.append(access)
                    episode_correct_count.append(correct_count)
                    episode_correct_rate.append(correct_rate)

                    episode_pos_terminal.append(pos_terminal)
                    episode_neg_terminal.append(neg_terminal)
                    episode_non_terminal.append(non_terminal)

                    #print(pred_prob)

                    action_count = 0
                    break

            if current_reward > best_reward:
                best_reward = current_reward

            #self.logger.info('<%d> Current episode reward: %f' % (episode, current_reward))
            #self.logger.info('Best episode reward: %f' % (best_reward))

        #print(action_count_list) 
        action_count_avg = np.average(np.array(action_count_list))
        access_avg = np.average(np.array(episode_access))
        correct_count_avg = np.average(np.array(episode_correct_count))
        correct_rate_avg = np.average(np.array(episode_correct_rate))

        pos_terminal_avg = np.average(episode_pos_terminal)
        neg_terminal_avg = np.average(episode_neg_terminal)
        non_terminal_avg = np.average(episode_non_terminal)

        self.logger.info('{:<20s}: {:>.2f}'.format('Access avg', access_avg))
        self.logger.info('{:<20s}: {:>.2f}'.format('Correct count avg', correct_count_avg))
        self.logger.info('{:<20s}: {:>.2f}'.format('Correct rate avg', correct_rate_avg))
        self.logger.info('{:<20s}: {:>.2f}'.format('Action count avg', action_count_avg))
        self.logger.info('{:<20s}: {:>.2f}'.format('Pos terminal avg', pos_terminal_avg))
        self.logger.info('{:<20s}: {:>.2f}'.format('Neg terminal avg', neg_terminal_avg))
        self.logger.info('{:<20s}: {:>.2f}'.format('Non terminal avg', non_terminal_avg))

        self.logger.info('\n')


    def select_action(self):
        if self.args.dqn_train:
            self.eps = np.max([self.args.eps_min, self.args.eps_init - (self.args.eps_init - self.args.eps_min)*(float(self.step)/float(self.args.max_exploration_step))])
        elif self.args.dqn_test:
            self.eps = self.args.eps_test

        if self.args.test_policy_type == 'random' or np.random.rand() < self.eps:
            action = self.env.random_action()
        elif self.args.test_policy_type == 'prob_max' or self.args.test_policy_type == 'prob_min':
            action = self.env.baseline_action()
        elif self.args.test_policy_type == 'dqn':

            self.q = self.dqn.predict_Q_value(np.squeeze(self.env.state))[0]

            ## Msking problems for higher than 0.9
            self.q = self.env.mask_actions(self.q)

            action = np.argmax(self.q)
            
        return action 

    def write_log(self, episode_count, episode_reward):
        if not os.path.exists('./train.csv'):
            train_log = open('./train.csv', 'w')
            train_log.write('episode\t, total reward\n')
        else:
            train_log = open('./train.csv', 'a')
            train_log.write(str(episode_count) + '\t' + str(episode_reward) +'\n')
        
    @property
    def model_dir(self):
        return '{}_Terminal_'.format(self.env.env.model_dir, self.args.terminal_condition_type)
        #return '{}_State_{}_Reward_{}'.format(self.env.env.model_dir, self.args.state_type, self.args.reward_type)
        #return '{}_{}batch'.format(self.args.env_name, self.args.batch_size_dqn)
            
            
    def save(self):
        checkpoint_dir = os.path.join(self.args.dqn_checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, str(self.step)))
        self.logger.debug('*** Save at %d steps' % self.step)

    def load(self):
        self.logger.debug('Loading DQN checkpoint ...')
        checkpoint_dir = os.path.join(self.args.dqn_checkpoint_dir, self.model_dir)
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_model = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
            self.logger.debug('Success to load %s' % checkpoint_model)
            return True
        else:
            self.logger.info('Fail to find a checkpoint')
            return False

    def reset_episode(self):
        self.env.episode_step = 0
        self.episode_reward = 0

        return self.env.new_episode()

